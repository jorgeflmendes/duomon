from __future__ import annotations

import socket
import subprocess
from pathlib import Path

from ..core.env import env_enabled
from .benchmark_context import *


_SHOWDOWN_PROCESS: subprocess.Popen[Any] | None = None
_ROOT = Path(__file__).resolve().parents[2]


def _env_enabled(name: str, default: str = "0") -> bool:
    return env_enabled(name, default)


def _showdown_running(host: str = "127.0.0.1", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.35):
            return True
    except OSError:
        return False


def ensure_showdown_server(timeout: float = 45.0) -> None:

    if not _env_enabled("DUOMON_AUTOSTART_SHOWDOWN", "1"):
        return
    if _showdown_running():
        return

    global _SHOWDOWN_PROCESS
    npm = "npm.cmd" if os.name == "nt" else "npm"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    _SHOWDOWN_PROCESS = subprocess.Popen(
        [npm, "start", "--", "--no-security"],
        cwd=str(_ROOT / "showdown-server"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=creationflags,
    )
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if _showdown_running():
            print(f"[showdown] status=running pid={_SHOWDOWN_PROCESS.pid}")
            return
        time.sleep(0.50)
    raise RuntimeError(
        "Pokemon Showdown did not start at ws://localhost:8000/showdown/websocket "
        f"within {timeout:.0f}s."
    )


def _benchmark_logs_ctde_examples(config: AgentConfig) -> bool:
    if "DUOMON_BENCHMARK_LOG_CTDE_EXAMPLES" in os.environ:
        return _env_enabled("DUOMON_BENCHMARK_LOG_CTDE_EXAMPLES", "0")
    return bool(getattr(config, "log_ctde_joint_examples", True))


def _multi_battle_format(config: AgentConfig) -> str:
    if bool(getattr(config, "fixed_ally_team_enabled", False)):
        return str(
            getattr(config, "fixed_ally_battle_format", "")
            or "gen9duomonfixedalliesmultirandombattle"
        )
    return "gen9multirandombattle"


def _read_config_team_text(config: AgentConfig, role: str) -> str:
    inline = str(getattr(config, f"fixed_ally_team_{role}", "") or "").strip()
    if inline:
        return inline
    path = str(getattr(config, f"fixed_ally_team_{role}_path", "") or "").strip()
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _fixed_ally_team_hash(config: AgentConfig) -> str:
    if not bool(getattr(config, "fixed_ally_team_enabled", False)):
        return ""
    p1_text = "\n".join(
        line.rstrip() for line in _read_config_team_text(config, "p1").splitlines()
    ).strip()
    p3_text = "\n".join(
        line.rstrip() for line in _read_config_team_text(config, "p3").splitlines()
    ).strip()
    optimize_leads = bool(getattr(config, "fixed_ally_optimize_leads_enabled", False))
    mirror = bool(getattr(config, "mirror_opponent_team_enabled", False))
    lead_mode = "leadopt" if optimize_leads else "fixedlead"
    if mirror:
        digest = hashlib.sha256(
            f"mode\nmirror_opponents_{lead_mode}\n\np1\n{p1_text}\n\np3\n{p3_text}\n".encode(
                "utf-8"
            )
        ).hexdigest()
        return digest[:12]
    digest = hashlib.sha256(
        f"mode\n{lead_mode}\n\np1\n{p1_text}\n\np3\n{p3_text}\n".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _set_default_training_artifacts(config: AgentConfig) -> None:
    defaults = {
        "model_path": ("DUOMON_MODEL_PATH", "multi_independent_value_model.json"),
        "ctde_joint_reranker_path": (
            "DUOMON_CTDE_JOINT_RERANKER_PATH",
            "ctde_joint_reranker.json",
        ),
        "ctde_joint_dataset_path": (
            "DUOMON_CTDE_JOINT_DATASET_PATH",
            "ctde_joint_examples.jsonl",
        ),
        "ctde_outcomes_path": ("DUOMON_CTDE_OUTCOMES_PATH", "ctde_outcomes.jsonl"),
        "ctde_split_path": ("DUOMON_CTDE_SPLIT_PATH", "ctde_split.json"),
    }
    for field, (env_name, default_name) in defaults.items():
        if env_name not in os.environ:
            setattr(config, field, training_path(config, default_name))


def apply_data_hygiene_namespace(config: AgentConfig) -> str:

    team_hash = _fixed_ally_team_hash(config)
    explicit_namespace = os.environ.get("DUOMON_DATA_NAMESPACE", "").strip()
    namespace = explicit_namespace or (f"fixed_allies_{team_hash}" if team_hash else "")
    setattr(config, "fixed_ally_team_hash", team_hash)
    setattr(config, "data_namespace", namespace)
    if not namespace or _env_enabled("DUOMON_DISABLE_DATA_NAMESPACE", "0"):
        return namespace

    base_output_dir = os.environ.get("DUOMON_OUTPUT_DIR", "outputs")
    base_training_dir = os.environ.get("DUOMON_MODEL_DIR", "learned_weights")
    config.output_dir = os.path.join(base_output_dir, "runs", namespace)
    config.training_dir = os.path.join(base_training_dir, namespace)
    config.replay_path = output_path(config, "multi_independent_replays.jsonl")
    config.metrics_path = output_path(config, "multi_independent_metrics.jsonl")
    _set_default_training_artifacts(config)
    return namespace


def _result_label(result: Dict[str, Any]) -> str:
    if result.get("p1_won"):
        return "win"
    if result.get("p1_lost"):
        return "loss"
    if result.get("error"):
        return "error"
    return "unfinished"


async def _wait_logged_in(*players: Player, timeout: float = 20.0) -> None:
    async def wait_one(player: Player) -> None:
        try:
            wait_for_login = getattr(player.ps_client, "wait_for_login", None)
            if callable(wait_for_login):
                await asyncio.wait_for(wait_for_login(wait_for=int(timeout)), timeout=timeout + 2.0)
            else:
                await asyncio.wait_for(player.ps_client.logged_in.wait(), timeout=timeout)
        except Exception as exc:
            raise TimeoutError(
                f"{player.username} did not log in to the local Showdown server within {timeout:.0f}s. "
                "Verify that Pokemon Showdown is running at ws://localhost:8000/showdown/websocket."
            ) from exc

    await asyncio.gather(*(wait_one(p) for p in players))


async def _wait_for_latest_battle(
    player: Player, format_id: str, previous_tags: set[str], timeout: float = 30.0
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for tag, battle in player.battles.items():
            if tag not in previous_tags and format_id in tag and not battle.finished:
                return battle
        await asyncio.sleep(0.10)
    raise TimeoutError(f"No new {format_id!r} battle appeared for {player.username}.")


async def _wait_for_battle_tag(player: Player, battle_tag: str, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        battle = player.battles.get(battle_tag)
        if battle is not None:
            return battle
        await asyncio.sleep(0.10)
    raise TimeoutError(f"{player.username} did not join room {battle_tag!r} within {timeout:.0f}s.")


async def _send_multi_invites(host: Player, battle_tag: str, p3: Player, p4: Player) -> None:
    for user, role in [(p3.username, "p3"), (p4.username, "p4")]:
        await host.ps_client.send_message(f"/addplayer {user}, {role}", room=battle_tag)


async def _accept_multi_invites(host: Player, p3: Player, p4: Player) -> None:


    host_id = to_id_str(host.username)
    for player in (p3, p4):
        await player.ps_client.set_team(player.get_next_team())
        await player.ps_client.send_message(f"/accept {host_id}")


def _multi_winner_ids_from_replay(battle: Any) -> set[str]:
    replay = safe_getattr(battle, "_replay_data", []) or []
    for event in reversed(replay):
        if not isinstance(event, list) or len(event) < 3:
            continue
        if str(event[1]) != "win":
            continue
        raw = str(event[2] or "")
        if not raw:
            return set()
        winners = {
            to_id_str(part.strip()) for part in raw.replace(",", "&").split("&") if part.strip()
        }
        return {w for w in winners if w}
    return set()


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
