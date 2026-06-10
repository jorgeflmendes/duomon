from __future__ import annotations

from .benchmark_metrics_context import *
from .benchmark_metric_utils import *


def _base_result_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    finished = sum(1 for row in rows if row.get("finished"))
    wins = sum(1 for row in rows if _result_won(row))
    losses = sum(1 for row in rows if _result_lost(row))
    errors = sum(1 for row in rows if row.get("error"))
    interval = _wilson_interval(wins, finished or total)
    return {
        "total": total,
        "finished": finished,
        "wins": wins,
        "losses": losses,
        "errors": errors,
        "winrate_all": _rate(wins, total),
        "winrate_finished": _rate(wins, finished),
        "winrate_ci_low": interval["low"],
        "winrate_ci_high": interval["high"],
    }


def _message_has_conflict(row: Dict[str, Any]) -> bool:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    for message in messages:
        if not isinstance(message, dict):
            continue
        speech_act = str(message.get("speech_act") or "").lower()
        if speech_act in {"reject", "veto", "counter"}:
            return True
    return False


def _selected_signature(row: Dict[str, Any]) -> str:
    diagnostics = (
        row.get("decision_diagnostics") if isinstance(row.get("decision_diagnostics"), dict) else {}
    )
    return str(diagnostics.get("selected_signature") or "")


def _shared_plan_signatures(row: Dict[str, Any]) -> List[str]:
    diagnostics = (
        row.get("decision_diagnostics") if isinstance(row.get("decision_diagnostics"), dict) else {}
    )
    plan = (
        diagnostics.get("shared_joint_plan")
        if isinstance(diagnostics.get("shared_joint_plan"), dict)
        else {}
    )
    signatures = [
        str(plan.get("local_signature") or ""),
        str(plan.get("partner_signature") or ""),
    ]
    return [signature for signature in signatures if signature]


def _message_selected_labels(row: Dict[str, Any]) -> List[str]:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    if not messages or not isinstance(messages[0], dict):
        return []
    own_agent = str(messages[0].get("agent") or "")
    labels: List[str] = []
    for message in messages:
        if not isinstance(message, dict) or str(message.get("agent") or "") != own_agent:
            continue
        selected = (message.get("content") or {}).get("selected") or {}
        if not isinstance(selected, dict):
            continue
        label = str(selected.get("label") or "")
        if label:
            labels.append(label)
    return labels


def _norm_action_text(value: str) -> str:
    return str(value or "").lower().replace(" ", "").replace("_", "").replace("-", "")


def _has_shared_plan(row: Dict[str, Any], coordination: Dict[str, Any]) -> bool:
    protocol = str(coordination.get("protocol_used") or "").lower()
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    has_selected_message = any(
        isinstance(message, dict) and bool((message.get("content") or {}).get("selected"))
        for message in messages
    )
    diagnostics = (
        row.get("decision_diagnostics") if isinstance(row.get("decision_diagnostics"), dict) else {}
    )
    return (
        "shared" in protocol or has_selected_message or bool(diagnostics.get("shared_joint_plan"))
    )


def _turn_summary(turn_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    turns = len(turn_rows)
    if not turns:
        return {
            "turn_rows": 0,
            "analysed_turns": 0,
            "conflict_turns": 0,
            "conflict_rate": 0.0,
            "conflict_reasons": {},
            "planned_slots": 0,
            "consistent_planned_slots": 0,
            "consistency_rate": 0.0,
            "average_turns_per_battle": 0.0,
        }
    rows_by_turn: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    max_turn_by_battle: Dict[str, int] = defaultdict(int)
    for row in turn_rows:
        battle_tag = str(row.get("battle_tag") or "")
        turn = int(row.get("turn") or 0)
        rows_by_turn[(battle_tag, turn)].append(row)
        max_turn_by_battle[battle_tag] = max(max_turn_by_battle[battle_tag], turn)

    conflict_turns = 0
    planned_slots = 0
    consistent_planned_slots = 0
    conflict_reasons: Dict[str, int] = defaultdict(int)
    for _turn_key, rows in rows_by_turn.items():
        turn_has_conflict = False
        seen_signatures: set[str] = set()
        planned_signatures: set[str] = set()
        for row in rows:
            selected_signature = _selected_signature(row)
            if selected_signature:
                seen_signatures.add(selected_signature)
            planned_signatures.update(_shared_plan_signatures(row))

        for row in rows:
            coordination = row.get("coordination") or {}
            if not isinstance(coordination, dict):
                coordination = {}
            message_conflict = bool(coordination.get("message_conflict"))
            veto_reason = str(coordination.get("veto_reason") or "none").lower()
            partner_risk = _safe_float(coordination.get("partner_damage_risk"), 0.0)
            double_switch = bool(coordination.get("double_switch"))
            message_level_conflict = message_conflict or _message_has_conflict(row)
            row_conflict = (
                message_level_conflict
                or veto_reason not in {"", "none"}
                or partner_risk >= 0.25
                or double_switch
            )
            if row_conflict:
                turn_has_conflict = True
                if message_level_conflict:
                    conflict_reasons["message_reject"] += 1
                if veto_reason not in {"", "none"}:
                    conflict_reasons[f"veto:{veto_reason}"] += 1
                if partner_risk >= 0.25:
                    conflict_reasons["partner_damage_risk"] += 1
                if double_switch:
                    conflict_reasons["double_switch"] += 1

            if _has_shared_plan(row, coordination):
                planned_slots += 1
                selected_signature = _selected_signature(row)
                row_plan_signatures = _shared_plan_signatures(row) or list(planned_signatures)
                action_text = _norm_action_text(str(row.get("action") or ""))
                selected_labels = [
                    _norm_action_text(label) for label in _message_selected_labels(row)
                ]
                if row_conflict:
                    pass
                elif selected_signature and selected_signature in row_plan_signatures:
                    consistent_planned_slots += 1
                elif (
                    selected_signature
                    and planned_signatures
                    and selected_signature in planned_signatures
                ):
                    consistent_planned_slots += 1
                elif selected_labels and any(
                    label and label in action_text for label in selected_labels
                ):
                    consistent_planned_slots += 1
                elif not row_conflict and bool(coordination.get("message_agreement")):
                    consistent_planned_slots += 1

        conflict_turns += int(turn_has_conflict)

    battle_turns = [turn for tag, turn in max_turn_by_battle.items() if tag and turn > 0]
    analysed_turns = len(rows_by_turn)
    return {
        "turn_rows": turns,
        "analysed_turns": analysed_turns,
        "conflict_turns": conflict_turns,
        "conflict_rate": _rate(conflict_turns, analysed_turns),
        "conflict_reasons": dict(
            sorted(conflict_reasons.items(), key=lambda item: item[1], reverse=True)[:6]
        ),
        "planned_slots": planned_slots,
        "consistent_planned_slots": consistent_planned_slots,
        "consistency_rate": _rate(consistent_planned_slots, planned_slots),
        "average_turns_per_battle": sum(battle_turns) / len(battle_turns) if battle_turns else 0.0,
    }


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
