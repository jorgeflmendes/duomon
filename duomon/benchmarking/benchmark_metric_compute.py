from __future__ import annotations

from .benchmark_metrics_context import *
from .benchmark_metric_analysis import *


def compute_benchmark_metrics(
    results: Sequence[Dict[str, Any]],
    turn_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rows = list(results)
    turn_rows = list(turn_rows or [])
    base = _base_result_summary(rows)
    turn = _turn_summary(turn_rows)
    risk_turn = _compute_risk_decision_metrics(turn_rows)

    non_fixed_rows = [row for row in rows if not _is_fixed_ally_row(row)]
    non_fixed_summary = _base_result_summary(non_fixed_rows) if non_fixed_rows else None
    fixed_rows = len(rows) - len(non_fixed_rows)

    elapsed_values = [
        _safe_float(row.get("elapsed_seconds"), 0.0)
        for row in rows
        if _safe_float(row.get("elapsed_seconds"), 0.0) > 0.0
    ]
    avg_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0
    cost = (
        avg_elapsed
        if avg_elapsed > 0.0
        else max(1.0, float(turn.get("average_turns_per_battle", 0.0) or 0.0))
    )
    efficiency_value = float(base["winrate_finished"]) / cost if cost else 0.0
    efficiency_unit = "pts/sec" if avg_elapsed > 0.0 else "pts/turn"

    metrics = {
        "win_rate": {
            **METRIC_DEFINITIONS["win_rate"],
            "value": float(base["winrate_finished"]),
            "unit": "%",
            "detail": f"{base['wins']}/{base['finished'] or base['total']} wins",
            "sample": f"n={base['finished'] or base['total']} finished battles",
            "confidence": f"95% CI {base['winrate_ci_low']:.1f}-{base['winrate_ci_high']:.1f}%",
            "ci_low": float(base["winrate_ci_low"]),
            "ci_high": float(base["winrate_ci_high"]),
            "available": base["total"] > 0,
        },
        "conflict": {
            **METRIC_DEFINITIONS["conflict"],
            "value": float(turn["conflict_rate"]),
            "unit": "%",
            "detail": f"{turn['conflict_turns']}/{turn['analysed_turns']} battle-turns",
            "sample": f"{turn['turn_rows']} agent-turn logs",
            "confidence": "Lower is better",
            "reasons": turn.get("conflict_reasons", {}),
            "available": turn["analysed_turns"] > 0,
        },
        "consistency": {
            **METRIC_DEFINITIONS["consistency"],
            "value": float(turn["consistency_rate"]),
            "unit": "%",
            "detail": f"{turn['consistent_planned_slots']}/{turn['planned_slots']} planned slots",
            "sample": f"{turn['analysed_turns']} battle-turns analysed",
            "confidence": "Higher is better",
            "available": turn["planned_slots"] > 0,
        },
        "generalization": {
            **METRIC_DEFINITIONS["generalization"],
            "value": float(non_fixed_summary["winrate_finished"]) if non_fixed_summary else None,
            "unit": "%",
            "detail": (
                f"{non_fixed_summary['wins']}/{non_fixed_summary['finished'] or non_fixed_summary['total']} non-fixed wins"
                if non_fixed_summary
                else f"not measured in this run ({fixed_rows} fixed-team battles)"
            ),
            "sample": (
                f"n={non_fixed_summary['finished'] or non_fixed_summary['total']} non-fixed battles"
                if non_fixed_summary
                else "requires a random-allies benchmark run"
            ),
            "confidence": (
                f"95% CI {non_fixed_summary['winrate_ci_low']:.1f}-{non_fixed_summary['winrate_ci_high']:.1f}%"
                if non_fixed_summary
                else "N/A for fixed-team-only run"
            ),
            "available": bool(non_fixed_summary and non_fixed_summary["total"]),
        },
        "efficiency": {
            **METRIC_DEFINITIONS["efficiency"],
            "value": float(efficiency_value),
            "unit": efficiency_unit,
            "detail": (
                f"avg latency {avg_elapsed:.2f}s"
                if avg_elapsed > 0.0
                else f"avg battle length {turn['average_turns_per_battle']:.1f} turns"
            ),
            "sample": "uses elapsed_seconds when present; otherwise replay turn count",
            "confidence": f"cost={cost:.2f} {('seconds' if avg_elapsed > 0.0 else 'turns')}",
            "available": base["total"] > 0 and cost > 0.0,
        },

        "average_turns": {
            **METRIC_DEFINITIONS["average_turns"],
            "value": float(turn["average_turns_per_battle"]),
            "unit": "turns",
            "detail": f"avg {turn['average_turns_per_battle']:.1f} turns/battle",
            "sample": f"{turn['turn_rows']} agent-turn logs",
            "confidence": "raw count from turn logs",
            "available": turn["turn_rows"] > 0,
        },
        "average_decision_latency": {
            **METRIC_DEFINITIONS["average_decision_latency"],
            "value": avg_elapsed if avg_elapsed > 0.0 else None,
            "unit": "s/battle",
            "detail": (
                f"avg {avg_elapsed:.3f}s per battle"
                if avg_elapsed > 0.0
                else "elapsed_seconds not recorded"
            ),
            "sample": f"{len(elapsed_values)} battles with timing",
            "confidence": "wall clock, includes server round-trip",
            "available": avg_elapsed > 0.0,
        },
        "illegal_rejected_move_rate": _compute_illegal_rejected_rate(turn_rows),
        "ctde_top1_alignment": _compute_ctde_alignment(turn_rows),
        "model_inference_latency": _compute_model_latency(turn_rows),
        "risky_no_protect": _compute_risky_no_protect_metric(turn_rows),
        "primary_threat_coverage": _compute_primary_threat_coverage_metric(turn_rows),
    }
    return {
        "definitions": METRIC_DEFINITIONS,
        "metrics": metrics,
        "raw": {
            **base,
            **turn,
            **risk_turn,
            "fixed_team_battles": fixed_rows,
            "non_fixed_team_battles": len(non_fixed_rows),
            "average_elapsed_seconds": avg_elapsed,
        },
    }


def compute_metrics_by_opponent(
    results: Sequence[Dict[str, Any]],
    turn_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    rows_by_opponent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    turns_by_benchmark: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        rows_by_opponent[_opponent_kind(row)].append(row)
    for row in turn_rows or []:
        benchmark = str(row.get("benchmark_type") or "")
        turns_by_benchmark[benchmark.replace("vs_", "")].append(row)
    return {
        opponent: compute_benchmark_metrics(rows, turns_by_benchmark.get(opponent, []))
        for opponent, rows in rows_by_opponent.items()
    }


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
