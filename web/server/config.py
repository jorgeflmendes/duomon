from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SHOWDOWN_SERVER_ROOT = ROOT / "showdown-server"
DEFAULT_TEAM_P1 = ROOT / "teams" / "duomon_ally_p1.txt"
DEFAULT_TEAM_P3 = ROOT / "teams" / "duomon_ally_p3.txt"
NODE_EXE = os.environ.get("NODE_EXE", "node")
RANDOM_ALLY_TEAM_FORMAT = "gen9multirandombattle"


def _repo_path_from_env(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default))
    return path if path.is_absolute() else ROOT / path


OUTPUT_ROOT = _repo_path_from_env("DUOMON_OUTPUT_DIR", "outputs")
TRAINING_ROOT = _repo_path_from_env(
    "DUOMON_MODEL_DIR",
    "learned_weights",
)
DEMO_TEAM_ROOT = OUTPUT_ROOT / "demo_teams"


def _coerce_team_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("text") or value.get("team") or ""
    text = str(value or "").strip()
    if text.startswith("{") and "'value'" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                text = str(
                    parsed.get("value") or parsed.get("text") or parsed.get("team") or text
                ).strip()
        except (SyntaxError, ValueError):
            pass
    return text


def _looks_like_exported_set(block: str) -> bool:
    lines = [line.strip() for line in str(block or "").splitlines() if line.strip()]
    if not lines:
        return False
    return any(line.startswith("- ") for line in lines) and any(
        line.startswith("Ability:") or line.startswith("Level:") or " @ " in line for line in lines
    )


def _normalize_team_text(value: Any, max_sets: int = 3) -> str:
    text = _coerce_team_text(value)
    if not text or ("|" in text and "\n" not in text):
        return text
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    pokemon_blocks = [block for block in blocks if _looks_like_exported_set(block)]
    if pokemon_blocks:
        return "\n\n".join(pokemon_blocks[: max(1, int(max_sets))])
    return text


def _read_text_file(path: Path) -> str:
    try:
        return _normalize_team_text(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""


def _payload_enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _write_demo_team(name: str, team_text: str, fallback_path: Path) -> Path:
    text = _normalize_team_text(team_text) or _read_text_file(fallback_path)
    DEMO_TEAM_ROOT.mkdir(parents=True, exist_ok=True)
    path = DEMO_TEAM_ROOT / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _team_namespace(
    p1_text: str,
    p3_text: str,
    mirror_opponents: bool = False,
    optimize_leads: bool = True,
) -> str:
    p1_norm = "\n".join(line.rstrip() for line in p1_text.strip().splitlines()).strip()
    p3_norm = "\n".join(line.rstrip() for line in p3_text.strip().splitlines()).strip()
    lead_mode = "leadopt" if optimize_leads else "fixedlead"
    if not mirror_opponents:
        digest = hashlib.sha256(
            f"mode\n{lead_mode}\n\np1\n{p1_norm}\n\np3\n{p3_norm}\n".encode("utf-8")
        ).hexdigest()[:12]
        return f"fixed_allies_{digest}"
    digest = hashlib.sha256(
        f"mode\nmirror_opponents_{lead_mode}\n\np1\n{p1_norm}\n\np3\n{p3_norm}\n".encode("utf-8")
    ).hexdigest()[:12]
    return f"fixed_allies_mirror_{digest}"


def _split_exported_team(exported: str) -> tuple[str, str]:
    sets = [
        block.strip()
        for block in re.split(r"\n\s*\n", str(exported or "").strip())
        if block.strip()
    ]
    if len(sets) < 6:
        raise ValueError(f"Pokemon Showdown returned {len(sets)} sets; expected at least 6.")
    return "\n\n".join(sets[:3]), "\n\n".join(sets[3:6])


def _generate_showdown_team(format_id: str = RANDOM_ALLY_TEAM_FORMAT) -> str:
    generated = subprocess.run(
        [NODE_EXE, "pokemon-showdown", "generate-team", format_id],
        cwd=str(SHOWDOWN_SERVER_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    exported = subprocess.run(
        [NODE_EXE, "pokemon-showdown", "export-team"],
        cwd=str(SHOWDOWN_SERVER_ROOT),
        input=generated.stdout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return _normalize_team_text(exported.stdout, max_sets=3)


def _generate_random_ally_teams() -> Dict[str, str]:
    p1_text = _generate_showdown_team()
    p3_text = _generate_showdown_team()
    return {
        "fixed_ally_team_enabled": True,
        "ally_p1_team": p1_text,
        "ally_p3_team": p3_text,
        "source": f"pokemon-showdown {RANDOM_ALLY_TEAM_FORMAT}",
    }


def _fixed_team_env(payload: Dict[str, Any]) -> Dict[str, str]:
    if not _payload_enabled(payload.get("fixed_ally_team_enabled"), True):
        return {}
    mirror_opponents = _payload_enabled(payload.get("mirror_opponent_team_enabled"), False)
    optimize_leads = _payload_enabled(payload.get("fixed_ally_optimize_leads_enabled"), True)
    p1_text = _normalize_team_text(payload.get("ally_p1_team")) or _read_text_file(DEFAULT_TEAM_P1)
    p3_text = _normalize_team_text(payload.get("ally_p3_team")) or _read_text_file(DEFAULT_TEAM_P3)
    namespace = _team_namespace(p1_text, p3_text, mirror_opponents, optimize_leads)
    training_root = TRAINING_ROOT / namespace
    results_root = OUTPUT_ROOT / "runs" / namespace
    p1_path = _write_demo_team("ally_p1.txt", p1_text, DEFAULT_TEAM_P1)
    p3_path = _write_demo_team("ally_p3.txt", p3_text, DEFAULT_TEAM_P3)
    return {
        "DUOMON_FIXED_ALLY_TEAMS": "1",
        "DUOMON_FIXED_ALLY_BATTLE_FORMAT": "gen9duomonfixedalliesmultirandombattle",
        "DUOMON_FIXED_ALLY_TEAM_P1_PATH": str(p1_path),
        "DUOMON_FIXED_ALLY_TEAM_P3_PATH": str(p3_path),
        "DUOMON_MIRROR_OPPONENT_TEAM_ENABLED": "1" if mirror_opponents else "0",
        "DUOMON_FIXED_ALLY_OPTIMIZE_LEADS": "1" if optimize_leads else "0",
        "DUOMON_DATA_NAMESPACE": namespace,
        "DUOMON_CTDE_JOINT_DATASET_PATH": str(training_root / "ctde_joint_examples.jsonl"),
        "DUOMON_CTDE_OUTCOMES_PATH": str(training_root / "ctde_outcomes.jsonl"),
        "DUOMON_CTDE_SPLIT_PATH": str(training_root / "ctde_split.json"),
        "DUOMON_CTDE_OUT_DIR": str(training_root / "outcome_ctde_100_each"),
        "DUOMON_CTDE_RESULTS_DIR": str(results_root),
    }


def _path_has_records(path_value: str) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("r", encoding="utf-8") as handle:
            return any(bool(line.strip()) for line in handle)
    except OSError:
        return False


def _dataset_has_training_scope(path_value: str, train_opponent: str) -> bool:
    wanted = {
        "simple": {"vs_simpleheuristics"},
        "abyssal": {"vs_abyssal"},
        "both": {"vs_simpleheuristics", "vs_abyssal"},
    }.get(str(train_opponent or "both").lower(), {"vs_simpleheuristics", "vs_abyssal"})
    found: set[str] = set()
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                benchmark_type = ""
                try:
                    record = json.loads(line)
                    benchmark_type = str(record.get("benchmark_type") or "")
                except json.JSONDecodeError:
                    pass
                for benchmark in wanted:
                    if (
                        benchmark_type == benchmark
                        or f'"benchmark_type": "{benchmark}"' in line
                        or f'"benchmark_type":"{benchmark}"' in line
                    ):
                        found.add(benchmark)
                if wanted.issubset(found):
                    return True
    except OSError:
        return False
    return False


def _reset_ctde_training_inputs(env: Dict[str, str]) -> None:
    for key in (
        "DUOMON_CTDE_JOINT_DATASET_PATH",
        "DUOMON_CTDE_OUTCOMES_PATH",
        "DUOMON_CTDE_SPLIT_PATH",
    ):
        raw_path = str(env.get(key) or "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
