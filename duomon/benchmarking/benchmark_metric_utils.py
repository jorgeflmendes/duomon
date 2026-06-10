from __future__ import annotations

from .benchmark_metrics_context import *
from .benchmark_metric_definitions import *


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _wilson_interval(wins: int, total: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    phat = float(wins) / float(total)
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denom
    margin = z * ((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) ** 0.5 / denom
    return {
        "low": 100.0 * max(0.0, centre - margin),
        "high": 100.0 * min(1.0, centre + margin),
    }


def _result_won(row: Dict[str, Any]) -> bool:
    return bool(row.get("p1_won", row.get("won", False)))


def _result_lost(row: Dict[str, Any]) -> bool:
    return bool(row.get("p1_lost", row.get("lost", False)))


def _is_fixed_ally_row(row: Dict[str, Any]) -> bool:
    if "fixed_ally_team_enabled" in row:
        return bool(row.get("fixed_ally_team_enabled"))
    tag = str(row.get("battle_tag") or "")
    return "fixedallies" in tag


def _opponent_kind(row: Dict[str, Any]) -> str:
    return str(
        row.get("opponent_kind")
        or str(row.get("benchmark_type") or "").replace("vs_", "")
        or "unknown"
    )


def load_turn_rows_for_results(
    results: Sequence[Dict[str, Any]], root_dir: str = "."
) -> List[Dict[str, Any]]:
    wanted_tags = {str(row.get("battle_tag") or "") for row in results if row.get("battle_tag")}
    if not wanted_tags:
        return []
    paths = []
    for row in results:
        raw_path = str(row.get("replay_log_path") or row.get("replay_path") or "")
        if not raw_path:
            continue
        path = raw_path if os.path.isabs(raw_path) else os.path.join(root_dir, raw_path)
        paths.append(os.path.normpath(path))
    turn_rows: List[Dict[str, Any]] = []
    for path in sorted(set(paths)):
        for record in iter_jsonl(path) or []:
            if str(record.get("battle_tag") or "") in wanted_tags and "coordination" in record:
                turn_rows.append(record)
    return turn_rows


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
