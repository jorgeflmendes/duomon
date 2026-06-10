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
from .heuristic_threat import *


class TacticalKnowledgeEvaluator:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.threat_model = OpponentThreatModel()

    def slot_action_bonus(self, battle: DoubleBattle, slot: int, action: SlotAction) -> float:
        if not self.config.tactical_knowledge_enabled:
            return 0.0
        threat = self.threat_model.analyze(battle)
        if action.kind == "switch":
            return self.config.switch_tactical_weight * self._switch_bonus(
                battle, slot, action, threat
            )
        if action.kind != "move" or action.move is None:
            return 0.0
        mid = move_id(action.move)
        if move_base_power(action.move) > 0:
            return self._damaging_move_bonus(battle, slot, action, threat)
        if mid in PROTECT_MOVES:
            return self.config.protect_tactical_weight * self._protect_bonus(battle, slot, threat)
        if mid in FAKE_OUT_MOVES:
            return self.config.fake_out_tactical_weight * self._fake_out_bonus(
                battle, slot, action, threat
            )
        if mid in SPEED_CONTROL_MOVES:
            return self.config.speed_control_tactical_weight * self._speed_control_bonus(
                battle, action, threat
            )
        if mid in REDIRECTION_MOVES:
            return self.config.redirection_tactical_weight * self._redirection_bonus(
                battle, slot, threat
            )
        if mid in STATUS_CONTROL_MOVES:
            return self.config.status_control_tactical_weight * self._status_control_bonus(
                battle, slot, action, threat
            )
        if mid in HELPING_HAND_MOVES:
            return self.config.helping_hand_tactical_weight * self._helping_hand_bonus(battle, slot)
        return 0.0

    def _damaging_move_bonus(
        self, battle: DoubleBattle, slot: int, action: SlotAction, threat: ThreatEstimate
    ) -> float:
        attacker = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        move = action.move
        if attacker is None or move is None:
            return 0.0
        bonus = 0.0
        if is_spread_move(move):
            ratios = []
            visible_opponents = active_alive_mons(battle.opponent_active_pokemon)
            if not visible_opponents:
                blind_ratio = blind_positional_damage_ratio(move, attacker, spread=True)
                bonus += self.config.spread_pressure_weight * 2.0 * blind_ratio
                if move_accuracy(move) >= 0.95:
                    bonus += 0.22
                return bonus
            for opp in visible_opponents:
                mult = damage_multiplier(opp, move)
                if mult == 0:
                    continue
                ratios.append(_advanced_damage_ratio(battle, move, attacker, opp, spread=True))
                if mult >= 2:
                    bonus += 0.20
            if ratios:
                bonus += self.config.spread_pressure_weight * float(np.sum(ratios))
            if move_target_type(move) in {"allAdjacent", "all"}:
                for ally in active_alive_mons(battle.active_pokemon):
                    if ally is not attacker and damage_multiplier(ally, move) > 0:
                        bonus -= 0.85 * damage_multiplier(ally, move)
            return bonus
        if action.target is None:
            return bonus
        if _target_is_ally(battle, action.target):
            ally_ratio = _advanced_damage_ratio(battle, move, attacker, action.target)
            if _move_hits_ally_activation(move, action.target):
                bonus += 0.62
                if ally_ratio <= 0.05:
                    bonus += 0.30
                elif ally_ratio <= 0.25:
                    bonus += 0.12
                bonus -= 0.85 * utility_damage_ratio(ally_ratio, cap=1.35)
                if safe_hp_fraction(action.target) < 0.45 and ally_ratio > 0.05:
                    bonus -= 0.65
                return bonus
            return -2.75 - 1.45 * utility_damage_ratio(ally_ratio, cap=1.35)
        mult = damage_multiplier(action.target, move)
        ratio = _advanced_damage_ratio(battle, move, attacker, action.target)
        ko = _advanced_ko_probability(battle, move, attacker, action.target)
        if mult == 0:
            return -2.20
        if mult >= 4:
            bonus += 0.75 * self.config.type_effectiveness_weight
        elif mult >= 2:
            bonus += 0.50 * self.config.type_effectiveness_weight
        elif mult <= 0.25:
            bonus -= 0.70 * self.config.type_effectiveness_weight
        elif mult < 1:
            bonus -= 0.35 * self.config.type_effectiveness_weight
        bonus += self.config.ko_pressure_weight * ko
        if ratio >= 0.95:
            bonus += 0.55
        elif ratio >= 0.70:
            bonus += 0.25
        target_label = safe_species(action.target)
        if target_label == threat.global_target:
            bonus += 0.22
        if target_label in threat.threatening_opp.values():
            bonus += 0.28
        return bonus

    def _protect_bonus(self, battle: DoubleBattle, slot: int, threat: ThreatEstimate) -> float:
        my_mon = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if my_mon is None:
            return 0.0
        hp = safe_hp_fraction(my_mon)
        slot_threat = threat.slot_threat.get(slot, 0.0)
        ko_risk = threat.slot_ko_risk.get(slot, 0.0)
        combined_ko = threat.slot_combined_ko.get(slot, 0.0)
        two_hko = threat.slot_2hko_risk.get(slot, 0.0)
        priority_threat = threat.slot_priority_threat.get(slot, 0.0)
        partner_can_punish = 0.0
        danger = threat.threatening_opp_ref.get(slot)
        if danger is not None:
            partner_can_punish = self._best_slot_damage_into_target(battle, 1 - slot, danger)
        value = 0.0
        risk = max(slot_threat, ko_risk, combined_ko, 0.65 * two_hko)
        if risk >= 0.65 or ko_risk >= 0.40 or combined_ko >= 0.30 or hp < 0.35:
            value += 0.80
        if two_hko >= 0.85 and hp < 0.70:
            value += 0.40
        if priority_threat and hp < 0.55:
            value += 0.30
        if partner_can_punish >= 0.55:
            value += 0.45
        elif partner_can_punish >= 0.35 and risk >= 0.45:
            value += 0.25
        if risk < 0.30 and hp > 0.65:
            value -= 0.20
        return value

    def _fake_out_bonus(
        self, battle: DoubleBattle, slot: int, action: SlotAction, threat: ThreatEstimate
    ) -> float:
        target = action.target
        if target is None:
            return 0.0
        if "ghost" in _mon_type_names(target):
            return -1.30
        if _psychic_terrain_blocks_priority(battle, target):
            return -0.70
        target_label = safe_species(target)
        value = 0.35
        if target_label == threat.global_target:
            value += 0.35
        if target_label in threat.threatening_opp.values():
            value += 0.45
        if int(safe_getattr(battle, "turn", 0) or 0) <= 2:
            value += 0.25
        value += min(0.35, 0.35 * self._best_slot_damage_into_target(battle, 1 - slot, target))
        return value

    def _speed_control_bonus(
        self, battle: DoubleBattle, action: SlotAction, threat: ThreatEstimate
    ) -> float:
        needed = self._speed_control_needed(battle)
        if needed <= 0:
            return -0.10
        value = needed
        mid = move_id(action.move)
        if mid == "trickroom":
            my_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.active_pokemon)] or [100]
            )
            opp_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon)] or [100]
            )
            value += 0.45 if my_speed < opp_speed else -0.35
        if mid in {"icywind", "electroweb", "bulldoze", "stringshot"}:
            value += 0.20
        if max(threat.slot_threat.values()) >= 0.55:
            value += 0.20
        return value

    def _redirection_bonus(self, battle: DoubleBattle, slot: int, threat: ThreatEstimate) -> float:
        redirector = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if redirector is None:
            return 0.0
        partner_threat = threat.slot_threat.get(1 - slot, 0.0)
        partner_ko = threat.slot_ko_risk.get(1 - slot, 0.0)
        if partner_threat < 0.50 and partner_ko < 0.35:
            return -0.05
        return 0.35 + 0.55 * max(partner_threat, partner_ko) + 0.20 * safe_hp_fraction(redirector)

    def _status_control_bonus(
        self, battle: DoubleBattle, slot: int, action: SlotAction, threat: ThreatEstimate
    ) -> float:
        move = action.move
        target = action.target
        if move is None or target is None or not _target_is_opponent(battle, target):
            return -0.20
        mid = move_id(move)
        if not self._status_control_target_is_valid(battle, target, mid):
            return -1.80

        target_label = safe_species(target)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        target_threat = max(
            float(threat.global_pressure if target_label == threat.global_target else 0.0),
            max(
                (
                    float(threat.slot_threat.get(idx, 0.0))
                    for idx, label in threat.threatening_opp.items()
                    if label == target_label
                ),
                default=0.0,
            ),
        )
        partner_damage = self._best_slot_damage_into_target(battle, 1 - slot, target)
        speed_penalty = (
            0.25 if active is not None and safe_speed(active) + 8 < safe_speed(target) else 0.0
        )

        if mid in SLEEP_CONTROL_MOVES:
            value = 1.15 + 0.35 * float(turn <= 3)
            value += 0.80 * min(1.0, target_threat)
            value += 0.35 * min(1.0, partner_damage)
            value -= speed_penalty
            if target_label == threat.global_target:
                value += 0.35
            return value
        if mid in {"taunt", "encore", "disable"}:
            return 0.45 + 0.45 * min(1.0, target_threat) + 0.20 * float(turn <= 3)
        if mid in {"thunderwave", "glare", "nuzzle"}:
            return 0.35 + 0.75 * self._speed_control_needed(battle)
        if mid in {"willowisp", "toxic"}:
            return 0.25 + 0.35 * min(1.0, target_threat)
        if mid == "yawn":
            return 0.25 + 0.25 * min(1.0, target_threat)
        return 0.0

    @staticmethod
    def _status_control_target_is_valid(battle: DoubleBattle, target: Any, mid: str) -> bool:
        if target is None or is_fainted(target):
            return False
        if _mon_status_name(target):
            return False
        target_types = set(_mon_type_names(target))
        item = _mon_item_name(target)
        ability = _mon_ability_name(target)
        if mid in SLEEP_CONTROL_MOVES or mid == "yawn":
            if "electricterrain" in _battle_field_names(battle) and _is_grounded_approx(target):
                return False
            if "mistyterrain" in _battle_field_names(battle) and _is_grounded_approx(target):
                return False
            if ability in {"insomnia", "vitalspirit", "sweetveil", "comatose"}:
                return False
            if mid in {"spore", "sleeppowder", "grasswhistle"}:
                if (
                    "grass" in target_types
                    or ability in {"overcoat", "safetygoggles"}
                    or item == "safetygoggles"
                ):
                    return False
        if mid in {"thunderwave", "glare", "nuzzle"}:
            if "ground" in target_types and mid in {"thunderwave", "nuzzle"}:
                return False
            if "electric" in target_types:
                return False
        if mid == "willowisp" and "fire" in target_types:
            return False
        if mid == "toxic" and ("poison" in target_types or "steel" in target_types):
            return False
        return True

    def _helping_hand_bonus(self, battle: DoubleBattle, slot: int) -> float:
        partner_slot = 1 - slot
        partner = (
            battle.active_pokemon[partner_slot]
            if partner_slot < len(battle.active_pokemon)
            else None
        )
        if partner is None or is_fainted(partner):
            return -0.50



        partner_move_iter = normalize_slot_list(
            safe_getattr(battle, "available_moves", []), partner_slot
        )
        partner_moves = list(partner_move_iter) if partner_move_iter else []
        if not partner_moves:
            partner_moves = OpponentThreatModel._predicted_moves_for_opp(partner)
        best_gain = 0.0
        for move in partner_moves:
            if move_base_power(move) <= 0:
                continue
            spread = is_spread_move(move)
            for target in active_alive_mons(battle.opponent_active_pokemon):
                base = _advanced_damage_ratio(battle, move, partner, target, spread=spread)
                boosted = _advanced_damage_ratio(
                    battle, move, partner, target, spread=spread, helping_hand=True
                )
                best_gain = max(
                    best_gain, 1.0 if base < 0.90 <= boosted else max(0.0, boosted - base)
                )
        return best_gain - 0.08

    def _switch_bonus(
        self, battle: DoubleBattle, slot: int, action: SlotAction, threat: ThreatEstimate
    ) -> float:
        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        candidate = action.switch
        if active is None or candidate is None:
            return 0.0
        current_threat = threat.slot_threat.get(slot, 0.0)
        ko_risk = max(
            threat.slot_ko_risk.get(slot, 0.0),
            threat.slot_combined_ko.get(slot, 0.0),
            0.70 * threat.slot_2hko_risk.get(slot, 0.0),
        )
        active_hp = safe_hp_fraction(active)

        if current_threat < 0.45 and ko_risk < 0.35 and active_hp > 0.55:
            return -0.55

        candidate_risk = 0.0
        known = 0
        for opp in active_alive_mons(battle.opponent_active_pokemon):
            for move in OpponentThreatModel._predicted_moves_for_opp(opp):
                if move_base_power(move) <= 0:
                    continue
                known += 1
                candidate_risk += _advanced_damage_ratio(battle, move, opp, candidate)
        candidate_risk /= max(1, known)

        defensive = 0.0
        defensive_n = 0
        for opp in active_alive_mons(battle.opponent_active_pokemon):
            for move in OpponentThreatModel._predicted_moves_for_opp(opp):
                if move_base_power(move) <= 0:
                    continue
                mult = damage_multiplier(candidate, move)
                if mult <= 0:
                    defensive += 2.0
                elif mult < 1.0:
                    defensive += 1.0 - mult
                else:
                    defensive -= 0.4 * (mult - 1.0)
                defensive_n += 1
        defensive_score = defensive / max(1, defensive_n)

        offensive = 0.0
        cand_moves = safe_getattr(candidate, "moves", {}) or {}
        move_iter = cand_moves.values() if isinstance(cand_moves, dict) else cand_moves
        for move in move_iter:
            if move_base_power(move) <= 0:
                continue
            for opp in active_alive_mons(battle.opponent_active_pokemon):
                mult = damage_multiplier(opp, move)
                if mult >= 2.0:
                    offensive = max(offensive, 0.8)
                elif mult >= 1.0:
                    offensive = max(offensive, 0.3)
        return (
            current_threat
            + 0.85 * ko_risk
            - candidate_risk
            + 0.30 * safe_hp_fraction(candidate)
            + 0.55 * defensive_score
            + offensive
            - 0.20
        )

    @staticmethod
    def _speed_control_needed(battle: DoubleBattle) -> float:
        my_active = active_alive_mons(battle.active_pokemon)
        opp_active = active_alive_mons(battle.opponent_active_pokemon)
        if not my_active or not opp_active:
            return 0.0
        checks = 0
        bad = 0
        for mine in my_active:
            for opp in opp_active:
                checks += 1
                if safe_speed(opp) > safe_speed(mine):
                    bad += 1
        return bad / max(1, checks)

    @staticmethod
    def _best_slot_damage_into_target(battle: DoubleBattle, slot: int, target: Any) -> float:
        attacker = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if attacker is None or target is None:
            return 0.0
        best = 0.0
        for move in normalize_slot_list(safe_getattr(battle, "available_moves", []), slot):
            if move_base_power(move) <= 0:
                continue
            spread = is_spread_move(move)
            best = max(best, _advanced_damage_ratio(battle, move, attacker, target, spread=spread))
        return best





__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
