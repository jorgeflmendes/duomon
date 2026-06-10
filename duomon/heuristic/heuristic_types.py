from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import AgentConfig
from ..shared import (
    HISTORICAL_POKEMON_MOVE_DICT,
    DoubleBattle,
    GenData,
    Move,
    SingleBattleOrder,
)
from .heuristic_constants import *
from .heuristic_safe import *
from .heuristic_moves import *
from .heuristic_damage import *


@dataclass
class SlotAction:
    slot: int
    kind: str
    order: SingleBattleOrder
    label: str
    move: Optional[Any] = None
    target: Optional[Any] = None
    target_position: Optional[int] = None
    switch: Optional[Any] = None


@dataclass
class JointAction:
    left: SlotAction
    right: SlotAction
    value: float = 0.0
    search_score: float = 0.0
    soft_penalty: float = 0.0
    compatibility: float = 0.0
    communication_score: float = 0.0
    protocol_used: str = "independent-local"
    protocol_reason: str = "pure-local-baseline"
    messages: Optional[List[Dict[str, Any]]] = None

    @property
    def label(self) -> str:
        return f"L:{self.left.label} | R:{self.right.label}"


@dataclass
class CoordinationMetrics:
    focus_fire: bool = False
    split_targets: bool = False
    double_switch: bool = False
    damage_sum: float = 0.0
    damage_max: float = 0.0
    ko_prob_sum: float = 0.0
    ko_prob_max: float = 0.0
    double_protect: bool = False
    helping_hand_combo: bool = False
    speed_control_combo: bool = False
    redirection_combo: bool = False
    partner_damage_risk: float = 0.0
    support_without_attack: bool = False
    message_agreement: bool = False
    message_conflict: bool = False
    useful_focus_fire: bool = False
    overkill_risk: float = 0.0
    saved_partner_attempt: bool = False
    threat_response: bool = False
    plan_consistency: bool = False
    communication_gain: float = 0.0
    veto_reason: str = "none"
    resolved_by: str = "none"
    protocol_used: str = "independent-local"
    protocol_reason: str = "pure-local-baseline"
    independent_score: float = 0.0
    aggressive_score: float = 0.0
    threat_score: float = 0.0
    selected_score: float = 0.0
    offensive_opportunity: bool = False
    high_threat_state: bool = False
    protocol_bandit_bonus: float = 0.0
    protocol_raw_score: float = 0.0
    protocol_adjusted_score: float = 0.0
    protocol_reward: float = 0.0
    support_score: float = 0.0
    anti_protect_score: float = 0.0
    speed_score: float = 0.0
    role_left: str = "flex"
    role_right: str = "flex"
    role_confidence: float = 0.0
    role_consistency: float = 0.0
    intent_consistency: float = 0.0
    teammate_alignment: float = 0.0


@dataclass
class ThreatEstimate:
    slot_threat: Dict[int, float]
    slot_ko_risk: Dict[int, float]
    threatening_opp: Dict[int, str]
    threatening_opp_ref: Dict[int, Any]
    global_target: str
    global_target_ref: Optional[Any]
    global_pressure: float
    spread_threat: float


    slot_combined_ko: Dict[int, float] = field(default_factory=dict)

    slot_2hko_risk: Dict[int, float] = field(default_factory=dict)

    slot_priority_threat: Dict[int, float] = field(default_factory=dict)


MULTI_INTENT_BLACKBOARD: Dict[str, Dict[str, Dict[str, Any]]] = {}
MULTI_SHORT_MEMORY: Dict[str, Dict[str, Any]] = {}


def _role_num(role: Optional[str]) -> int:
    try:
        return int(str(role or "")[1])
    except Exception:
        return 0


def _multi_slot_letter(role: Optional[str]) -> str:
    return "a" if _role_num(role) in {1, 2} else "b"


def _multi_same_side(role_a: Optional[str], role_b: Optional[str]) -> bool:
    a = _role_num(role_a)
    b = _role_num(role_b)
    return bool(a and b and (a % 2) == (b % 2))


def _multi_partner_role(role: Optional[str]) -> Optional[str]:
    return {"p1": "p3", "p3": "p1", "p2": "p4", "p4": "p2"}.get(str(role or ""))


def _multi_opponent_roles(role: Optional[str]) -> List[str]:
    return {
        "p1": ["p2", "p4"],
        "p3": ["p2", "p4"],
        "p2": ["p1", "p3"],
        "p4": ["p1", "p3"],
    }.get(str(role or ""), [])


def _multi_coordination_key(battle: DoubleBattle, role: Optional[str]) -> str:
    battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
    turn = int(safe_getattr(battle, "turn", 0) or 0)
    side = "odd" if _role_num(role) in {1, 3} else "even"
    return f"{battle_key}:{turn}:{side}"


def _multi_memory_key(battle: DoubleBattle, role: Optional[str]) -> str:
    battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
    side = "odd" if _role_num(role) in {1, 3} else "even"
    return f"{battle_key}:{side}"


def _fresh_multi_short_memory() -> Dict[str, Any]:
    return {
        "ally_facts": {},
        "enemy_facts": {},
        "enemy_events": [],
        "partner_reports": {},
        "predictions": {},
        "team_plan": None,
        "commitments": {},
        "focus_target": None,
        "focus_target_slot": None,
        "focus_ttl": 0,
        "updated_at": time.time(),
    }


def _get_multi_short_memory(battle: DoubleBattle, role: Optional[str] = None) -> Dict[str, Any]:
    key = _multi_memory_key(battle, role or _multi_side_role(battle))
    mem = MULTI_SHORT_MEMORY.setdefault(key, _fresh_multi_short_memory())
    mem["updated_at"] = time.time()
    return mem


def _multi_side_role(battle: DoubleBattle) -> Optional[str]:
    return safe_getattr(battle, "player_role", None)


def compute_coordination_metrics(battle: DoubleBattle, action: JointAction) -> CoordinationMetrics:
    metrics = CoordinationMetrics(
        protocol_used=action.protocol_used, protocol_reason=action.protocol_reason
    )
    messages = action.messages or []
    metrics.message_agreement = any(
        str(m.get("speech_act", "")).lower() in {"accept", "commit"}
        for m in messages
        if isinstance(m, dict)
    )
    metrics.message_conflict = any(
        str(m.get("speech_act", "")).lower() == "reject" for m in messages if isinstance(m, dict)
    )
    metrics.plan_consistency = any(
        str(m.get("speech_act", "")).lower() == "commit" for m in messages if isinstance(m, dict)
    )
    left_mid = move_id(action.left.move) if action.left.move is not None else ""
    right_mid = move_id(action.right.move) if action.right.move is not None else ""





    partner_selected: Dict[str, Any] = {}
    own_agent = (
        str(messages[0].get("agent", "")) if messages and isinstance(messages[0], dict) else ""
    )
    for m in messages:
        if not isinstance(m, dict):
            continue
        if str(m.get("agent", "")) == own_agent:
            continue
        sel = (m.get("content") or {}).get("selected") or {}
        if not sel:
            continue

        partner_selected = sel

    partner_kind = str(partner_selected.get("kind") or "")
    partner_mid = str(partner_selected.get("move_id") or "")
    partner_target_slot = partner_selected.get("target_slot")
    partner_spread = bool(partner_selected.get("spread"))

    def target_slot_from_action(sa: SlotAction) -> Optional[int]:
        if sa.target is not None:
            for idx, opp in enumerate(battle.opponent_active_pokemon[:2]):
                if opp is sa.target:
                    return idx
        try:
            position = int(sa.target_position or 0)
            if position == int(getattr(battle, "OPPONENT_1_POSITION", 1)):
                return 0
            if position == int(getattr(battle, "OPPONENT_2_POSITION", 2)):
                return 1
        except Exception:
            return None
        return None

    metrics.double_switch = action.left.kind == "switch" and (
        action.right.kind == "switch" or partner_kind == "switch"
    )
    metrics.double_protect = left_mid in PROTECT_MOVES and (
        right_mid in PROTECT_MOVES or partner_mid in PROTECT_MOVES
    )

    targets = []
    for sa in [action.left, action.right]:
        if (
            sa.kind == "move"
            and sa.move is not None
            and sa.target is not None
            and move_base_power(sa.move) > 0
            and _target_is_opponent(battle, sa.target)
        ):
            targets.append(sa.target)
    if len(targets) >= 2:
        metrics.focus_fire = targets[0] is targets[1]
        metrics.split_targets = targets[0] is not targets[1]
    elif (
        len(targets) == 1
        and partner_target_slot is not None
        and action.left.kind == "move"
        and action.left.move is not None
        and move_base_power(action.left.move) > 0
    ):

        try:
            own_target_slot = target_slot_from_action(action.left)
            if own_target_slot is not None:
                if partner_spread or bool(
                    action.left.move is not None and is_spread_move(action.left.move)
                ):
                    metrics.focus_fire = True
                else:
                    metrics.focus_fire = int(partner_target_slot) == int(own_target_slot)
                    metrics.split_targets = not metrics.focus_fire
        except Exception:
            pass
    elif (
        not targets
        and partner_target_slot is not None
        and action.left.kind == "move"
        and action.left.move is not None
        and move_base_power(action.left.move) > 0
    ):
        try:
            own_target_slot = target_slot_from_action(action.left)
            if own_target_slot is not None:
                if partner_spread or bool(
                    action.left.move is not None and is_spread_move(action.left.move)
                ):
                    metrics.focus_fire = True
                else:
                    metrics.focus_fire = int(partner_target_slot) == int(own_target_slot)
                    metrics.split_targets = not metrics.focus_fire
        except Exception:
            pass

    damage_ratios: List[float] = []
    ko_probs: List[float] = []
    for sa in [action.left, action.right]:
        if sa.kind != "move" or sa.move is None or move_base_power(sa.move) <= 0:
            continue
        attacker = battle.active_pokemon[sa.slot] if sa.slot < len(battle.active_pokemon) else None
        if attacker is None:
            continue
        spread = is_spread_move(sa.move)
        if spread:
            visible_opponents = active_alive_mons(battle.opponent_active_pokemon)
            if not visible_opponents:
                dmg_ratio = blind_positional_damage_ratio(sa.move, attacker, spread=True)
                for _ in (0, 1):
                    damage_ratios.append(dmg_ratio)
                    ko_probs.append(blind_ko_probability_from_ratio(dmg_ratio))
            for opp in visible_opponents:
                hp = estimated_max_hp_points(opp)
                dmg = approximate_damage_points(sa.move, attacker, opp, spread=True)
                damage_ratios.append(dmg / max(1.0, hp))
                ko_probs.append(estimated_ko_probability(sa.move, attacker, opp, spread=True))
        elif sa.target is not None and _target_is_opponent(battle, sa.target):
            hp = estimated_max_hp_points(sa.target)
            dmg = approximate_damage_points(sa.move, attacker, sa.target, spread=False)
            damage_ratios.append(dmg / max(1.0, hp))
            ko_probs.append(estimated_ko_probability(sa.move, attacker, sa.target, spread=False))

    if partner_kind == "move":
        for value in (partner_selected.get("damage_by_slot", {}) or {}).values():
            try:
                damage_ratios.append(float(value))
            except Exception:
                continue
        for value in (partner_selected.get("ko_by_slot", {}) or {}).values():
            try:
                ko_probs.append(float(value))
            except Exception:
                continue

    metrics.damage_sum = float(np.sum(damage_ratios)) if damage_ratios else 0.0
    metrics.damage_max = float(np.max(damage_ratios)) if damage_ratios else 0.0
    metrics.ko_prob_sum = float(np.sum(ko_probs)) if ko_probs else 0.0
    metrics.ko_prob_max = float(np.max(ko_probs)) if ko_probs else 0.0

    partner_attacks = partner_kind == "move" and float(partner_selected.get("bp", 0.0)) > 0.0
    left_attacks = action.left.move is not None and move_base_power(action.left.move) > 0
    right_attacks = action.right.move is not None and move_base_power(action.right.move) > 0
    metrics.helping_hand_combo = (
        left_mid in HELPING_HAND_MOVES and (right_attacks or partner_attacks)
    ) or ((right_mid in HELPING_HAND_MOVES or partner_mid in HELPING_HAND_MOVES) and left_attacks)
    metrics.speed_control_combo = (
        left_mid in SPEED_CONTROL_MOVES and (right_attacks or partner_attacks)
    ) or ((right_mid in SPEED_CONTROL_MOVES or partner_mid in SPEED_CONTROL_MOVES) and left_attacks)
    metrics.redirection_combo = (
        left_mid in REDIRECTION_MOVES and (right_attacks or partner_attacks)
    ) or ((right_mid in REDIRECTION_MOVES or partner_mid in REDIRECTION_MOVES) and left_attacks)
    metrics.partner_damage_risk = estimate_partner_damage_risk(battle, action)
    metrics.support_without_attack = joint_support_without_attack(action)
    return metrics


def estimate_partner_damage_risk(battle: DoubleBattle, action: JointAction) -> float:
    risk = 0.0
    for sa in [action.left, action.right]:
        if sa.kind != "move" or sa.move is None:
            continue
        attacker = battle.active_pokemon[sa.slot] if sa.slot < len(battle.active_pokemon) else None
        if move_target_type(sa.move) in {"allAdjacent", "all"}:
            for ally in active_alive_mons(battle.active_pokemon):
                if ally is not attacker and damage_multiplier(ally, sa.move) > 0:
                    risk += damage_multiplier(ally, sa.move)
        elif (
            sa.target is not None
            and _target_is_ally(battle, sa.target)
            and move_base_power(sa.move) > 0
        ):
            ally_ratio = _advanced_damage_ratio(battle, sa.move, attacker, sa.target)
            activation_discount = 0.35 if _move_hits_ally_activation(sa.move, sa.target) else 1.0
            risk += activation_discount * utility_damage_ratio(ally_ratio, cap=1.50)
    return risk


def joint_support_without_attack(action: JointAction) -> bool:
    has_move = False
    has_damage = False
    for sa in [action.left, action.right]:
        if sa.kind == "move" and sa.move is not None:
            has_move = True
            if move_base_power(sa.move) > 0:
                has_damage = True
    return has_move and not has_damage


def action_uses_tera(action: SlotAction) -> bool:
    try:
        text = str(
            safe_getattr(action.order, "message", None)
            or safe_getattr(action.order, "order", None)
            or action.order
        ).lower()
        return "terastallize" in text or " tera" in text
    except Exception:
        return False





__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
