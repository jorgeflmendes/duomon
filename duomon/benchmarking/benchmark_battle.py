from __future__ import annotations

from .benchmark_context import *
from .benchmark_setup import *


async def run_multi_battle(
    p1: Player,
    p2: Player,
    p3: Player,
    p4: Player,
    timeout_seconds: float = 180.0,
    log_context: str = "",
) -> Dict[str, Any]:
    before = set(p1.battles.keys())
    prefix = f"[battle] {log_context} | " if log_context else "[battle] "

    print(
        f"{prefix}step=1/5 action=create_challenge host={p1.username} opponent={p2.username}",
        flush=True,
    )
    battle_task = asyncio.create_task(p1.battle_against(p2, n_battles=1))

    try:
        battle = await _wait_for_latest_battle(p1, p1.format, before, timeout=90.0)
        print(
            f"{prefix}step=2/5 action=room_created battle={battle.battle_tag}",
            flush=True,
        )

        print(
            f"{prefix}step=3/5 action=verify_extra_players players={p3.username},{p4.username}",
            flush=True,
        )
        await _wait_logged_in(p3, p4, timeout=45.0)

        print(
            f"{prefix}step=4/5 action=invite_extra_players battle={battle.battle_tag}",
            flush=True,
        )
        await _send_multi_invites(p1, battle.battle_tag, p3, p4)
        await asyncio.sleep(0.75)
        await _accept_multi_invites(p1, p3, p4)


        await _wait_for_battle_tag(p3, battle.battle_tag, timeout=90.0)
        await _wait_for_battle_tag(p4, battle.battle_tag, timeout=90.0)
        print(
            f"{prefix}step=5/5 action=battle_running battle={battle.battle_tag}",
            flush=True,
        )

        await asyncio.wait_for(battle_task, timeout=timeout_seconds)
    except Exception:
        if not battle_task.done():
            battle_task.cancel()
            try:
                await battle_task
            except BaseException:
                pass
        raise

    winners = _multi_winner_ids_from_replay(battle)
    ally_ids = {to_id_str(p1.username), to_id_str(p3.username)}
    opp_ids = {to_id_str(p2.username), to_id_str(p4.username)}


    fallback_won = bool(safe_getattr(battle, "won", False))
    fallback_lost = bool(safe_getattr(battle, "lost", False))
    team_won = bool(winners & ally_ids) if winners else fallback_won
    team_lost = bool(winners & opp_ids) if winners else fallback_lost

    result = {
        "battle_tag": battle.battle_tag,
        "p1": p1.username,
        "p2": p2.username,
        "p3": p3.username,
        "p4": p4.username,
        "finished": bool(battle.finished),

        "p1_won": team_won,
        "p1_lost": team_lost,
        "winner_ids": sorted(winners),
    }
    terminal_reward = 1.0 if team_won else -1.0 if team_lost else 0.0
    for ally_player in (p1, p3):
        apply_reward = getattr(ally_player, "apply_external_terminal_reward", None)
        if callable(apply_reward):
            try:
                apply_reward(battle.battle_tag, terminal_reward)
            except Exception as exc:
                logger.info(
                    f"Could not apply multi terminal reward for {ally_player.username}: {exc}"
                )
    return result


def append_ctde_outcomes(
    config: AgentConfig,
    results: Sequence[Dict[str, Any]],
    phase: str,
    source_result_path: str,
) -> int:
    outcomes_path = str(getattr(config, "ctde_outcomes_path", "") or "")
    if not outcomes_path:
        return 0
    written = 0
    ensure_parent_dir(outcomes_path)
    with open(outcomes_path, "a", encoding="utf-8") as handle:
        for result in results:
            tag = str(result.get("battle_tag") or "")
            if not tag or result.get("finished") is not True:
                continue
            record = {
                "battle_tag": tag,
                "p1_won": bool(result.get("p1_won", False)),
                "p1_lost": bool(result.get("p1_lost", False)),
                "finished": bool(result.get("finished", False)),
                "benchmark_type": result.get("benchmark_type", ""),
                "opponent_kind": result.get("opponent_kind", ""),
                "phase": phase,
                "source_result_path": source_result_path,
                "data_namespace": getattr(config, "data_namespace", ""),
                "fixed_ally_team_hash": getattr(config, "fixed_ally_team_hash", ""),
            }
            handle.write(json.dumps(json_safe(record)) + "\n")
            written += 1
    return written


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
