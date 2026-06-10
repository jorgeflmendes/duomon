from __future__ import annotations

from .benchmark_metrics_context import *
from .benchmark_metric_turns import *


def _compute_illegal_rejected_rate(
    turn_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total = 0
    rejected = 0
    for row in turn_rows or []:
        total += 1
        coordination = row.get("coordination") or {}
        if not isinstance(coordination, dict):
            coordination = {}
        veto_reason = str(coordination.get("veto_reason") or "").strip()
        is_repeat = bool(coordination.get("repeated_order"))
        is_rejected = bool(coordination.get("move_rejected"))
        is_illegal = bool(coordination.get("illegal_move"))
        if veto_reason not in {"", "none"} or is_repeat or is_rejected or is_illegal:
            rejected += 1
    rate = _rate(rejected, total)
    return {
        **METRIC_DEFINITIONS["illegal_rejected_move_rate"],
        "value": rate,
        "unit": "%",
        "detail": f"{rejected}/{total} turns with rejected/illegal orders",
        "sample": f"{total} agent-turn logs",
        "confidence": "Lower is better",
        "available": total > 0,
    }


def _compute_ctde_alignment(
    turn_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total = 0
    aligned = 0
    for row in turn_rows or []:
        diagnostics = row.get("decision_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        ctde_top1 = diagnostics.get("ctde_top1_matched")
        if ctde_top1 is None:
            continue
        total += 1
        if bool(ctde_top1):
            aligned += 1
    rate = _rate(aligned, total)
    return {
        **METRIC_DEFINITIONS["ctde_top1_alignment"],
        "value": rate if total > 0 else None,
        "unit": "%",
        "detail": f"{aligned}/{total} turns with CTDE top-1 matching chosen pair",
        "sample": f"{total} turns with CTDE scores",
        "confidence": "Requires ctde_top1_matched field in decision_diagnostics",
        "available": total > 0,
    }


def _compute_model_latency(
    turn_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    latencies: List[float] = []
    for row in turn_rows or []:
        lat = row.get("inference_latency_ms")
        if lat is None:
            diagnostics = row.get("decision_diagnostics")
            if isinstance(diagnostics, dict):
                lat = diagnostics.get("inference_latency_ms")
        if lat is not None:
            v = _safe_float(lat, -1.0)
            if v >= 0.0:
                latencies.append(v)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        **METRIC_DEFINITIONS["model_inference_latency"],
        "value": avg_lat if latencies else None,
        "unit": "ms/turn",
        "detail": f"avg {avg_lat:.2f}ms over {len(latencies)} scored turns",
        "sample": f"{len(latencies)} turns with inference timing",
        "confidence": "Requires inference_latency_ms in turn logs",
        "available": len(latencies) > 0,
    }


def _selected_defensive_gate(row: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = row.get("decision_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    gates = diagnostics.get("selected_gates")
    if not isinstance(gates, dict):
        return {}
    defensive = gates.get("defensive")
    return defensive if isinstance(defensive, dict) else {}


def _selected_risk_context(row: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = row.get("decision_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    context = diagnostics.get("risk_context")
    return context if isinstance(context, dict) else {}


def _compute_risk_decision_metrics(
    turn_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total = 0
    high_risk = 0
    risky_no_protect = 0
    threat_covered = 0
    protect_lines = 0
    no_protect_available = 0
    risk_sum = 0.0
    risky_reasons = {
        "soft_penalty_risky_action_without_protect",
        "penalize_non_defensive_action_under_ko_risk",
        "penalize_low_hp_action_without_threat_removal",
    }
    for row in turn_rows or []:
        defensive = _selected_defensive_gate(row)
        context = _selected_risk_context(row)
        if not defensive and not context:
            continue
        total += 1
        risk = _safe_float(context.get("risk", defensive.get("risk", 0.0)), 0.0)
        risk_sum += risk
        reason = str(defensive.get("reason") or "")
        preempts = bool(defensive.get("preempts_primary_threat"))
        protect = reason.startswith("protect_")
        if protect:
            protect_lines += 1
        if not bool(context.get("protect_candidate_available", context.get("protect_available", True))):
            no_protect_available += 1
        if risk < 0.66:
            continue
        high_risk += 1
        if preempts or reason in {
            "risk_covered_by_attack",
            "reward_primary_threat_removal_under_risk",
        }:
            threat_covered += 1
        elif not protect and reason in risky_reasons:
            risky_no_protect += 1
        elif not protect and not preempts:
            risky_no_protect += 1

    return {
        "diagnostic_turns": total,
        "average_predicted_risk": risk_sum / total if total else 0.0,
        "high_risk_decisions": high_risk,
        "risky_no_protect_decisions": risky_no_protect,
        "primary_threat_covered_decisions": threat_covered,
        "protect_lines": protect_lines,
        "no_protect_available_decisions": no_protect_available,
        "risky_no_protect_rate": _rate(risky_no_protect, high_risk),
        "primary_threat_coverage_rate": _rate(threat_covered, high_risk),
        "protect_line_rate": _rate(protect_lines, total),
        "no_protect_available_rate": _rate(no_protect_available, total),
    }


def _compute_risky_no_protect_metric(turn_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    risk = _compute_risk_decision_metrics(turn_rows)
    high_risk = int(risk["high_risk_decisions"])
    risky = int(risk["risky_no_protect_decisions"])
    return {
        **METRIC_DEFINITIONS["risky_no_protect"],
        "value": float(risk["risky_no_protect_rate"]) if high_risk else None,
        "unit": "%",
        "detail": f"{risky}/{high_risk} high-risk decisions",
        "sample": f"{risk['diagnostic_turns']} decision diagnostics",
        "confidence": f"avg predicted risk {risk['average_predicted_risk']:.3f}",
        "available": high_risk > 0,
    }


def _compute_primary_threat_coverage_metric(turn_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    risk = _compute_risk_decision_metrics(turn_rows)
    high_risk = int(risk["high_risk_decisions"])
    covered = int(risk["primary_threat_covered_decisions"])
    return {
        **METRIC_DEFINITIONS["primary_threat_coverage"],
        "value": float(risk["primary_threat_coverage_rate"]) if high_risk else None,
        "unit": "%",
        "detail": f"{covered}/{high_risk} high-risk decisions",
        "sample": f"{risk['diagnostic_turns']} decision diagnostics",
        "confidence": f"protect lines {risk['protect_line_rate']:.1f}%",
        "available": high_risk > 0,
    }


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
