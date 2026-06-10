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
from .heuristic_types import *


class OpponentThreatModel:
    @staticmethod
    def _predicted_moves_for_opp(opp: Any) -> List[Any]:
        observed_raw = safe_getattr(opp, "moves", {}) or {}
        observed = (
            list(observed_raw.values()) if isinstance(observed_raw, dict) else list(observed_raw)
        )
        predicted: List[Any] = [move for move in observed if move is not None]

        seen = {move_id(move) for move in predicted if move is not None}
        historical = HISTORICAL_POKEMON_MOVE_DICT.get(species_key(opp), {}) or {}
        if isinstance(historical, dict):
            ranked = sorted(
                historical.values(),
                key=lambda item: float(item[3])
                if isinstance(item, list) and len(item) > 3
                else 0.0,
                reverse=True,
            )
            for item in ranked:
                if not isinstance(item, list) or not item:
                    continue
                mid = move_id(item[0])
                if not mid or mid in seen:
                    continue
                try:
                    predicted.append(Move(item[0], gen=9))
                    seen.add(mid)
                except Exception:
                    continue
                if len(predicted) >= 4:
                    break
        return predicted

    def analyze(self, battle: DoubleBattle) -> ThreatEstimate:
        slot_threat = {0: 0.0, 1: 0.0}
        slot_ko_risk = {0: 0.0, 1: 0.0}
        threatening_opp: Dict[int, str] = {0: "none", 1: "none"}
        threatening_opp_ref: Dict[int, Any] = {0: None, 1: None}
        opp_pressure: Dict[str, float] = {}
        opp_ref_by_name: Dict[str, Any] = {}
        spread_threat = 0.0



        slot_best_per_opp: Dict[int, List[Tuple[float, float, bool]]] = {0: [], 1: []}

        my_slots = [
            (i, m)
            for i, m in enumerate(battle.active_pokemon[:2])
            if m is not None and not is_fainted(m)
        ]
        for opp in active_alive_mons(battle.opponent_active_pokemon):
            opp_name = safe_species(opp)
            opp_ref_by_name[opp_name] = opp
            move_list = self._predicted_moves_for_opp(opp)
            best_for_opp = 0.0


            opp_slot_best: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0), 1: (0.0, 0.0)}
            opp_slot_priority: Dict[int, bool] = {0: False, 1: False}

            for move in move_list:
                if move_base_power(move) <= 0:
                    continue
                is_spread = is_spread_move(move)
                priority_flag = float(move_priority(move)) > 0
                if is_spread:
                    current_spread = 0.0
                    for slot, mine in my_slots:
                        ratio = approximate_damage_points(move, opp, mine, spread=True) / max(
                            1.0, estimated_max_hp_points(mine)
                        )
                        current_spread += ratio
                        hp_frac_mine = safe_hp_fraction(mine)
                        effective_vs_me = ratio / max(0.01, hp_frac_mine)
                        ko_risk_here = _ko_prob_from_effective(effective_vs_me)
                        if ratio > slot_threat[slot]:
                            slot_threat[slot] = ratio
                            slot_ko_risk[slot] = ko_risk_here
                            threatening_opp[slot] = opp_name
                            threatening_opp_ref[slot] = opp
                        if ratio > opp_slot_best[slot][0]:
                            opp_slot_best[slot] = (
                                ratio,
                                ko_risk_here,
                            )
                        if priority_flag:
                            opp_slot_priority[slot] = True
                        best_for_opp = max(best_for_opp, ratio)
                    spread_threat = max(spread_threat, current_spread)
                else:
                    for slot, mine in my_slots:
                        ratio = approximate_damage_points(move, opp, mine, spread=False) / max(
                            1.0, estimated_max_hp_points(mine)
                        )
                        hp_frac_mine = safe_hp_fraction(mine)
                        effective_vs_me = ratio / max(0.01, hp_frac_mine)
                        ko_risk_here = _ko_prob_from_effective(effective_vs_me)
                        if ratio > slot_threat[slot]:
                            slot_threat[slot] = ratio
                            slot_ko_risk[slot] = ko_risk_here
                            threatening_opp[slot] = opp_name
                            threatening_opp_ref[slot] = opp
                        if ratio > opp_slot_best[slot][0]:
                            opp_slot_best[slot] = (
                                ratio,
                                ko_risk_here,
                            )
                        if priority_flag:
                            opp_slot_priority[slot] = True
                        best_for_opp = max(best_for_opp, ratio)
            for slot in (0, 1):
                if opp_slot_best[slot][0] > 0.0:
                    slot_best_per_opp[slot].append(
                        (opp_slot_best[slot][0], opp_slot_best[slot][1], opp_slot_priority[slot])
                    )
            opp_pressure[opp_name] = utility_damage_ratio(best_for_opp, cap=1.35) + 0.35 * (
                1.0 - safe_hp_fraction(opp)
            )





        slot_combined_ko: Dict[int, float] = {0: 0.0, 1: 0.0}
        slot_2hko_risk: Dict[int, float] = {0: 0.0, 1: 0.0}
        slot_priority_threat: Dict[int, float] = {0: 0.0, 1: 0.0}
        for slot, mine in my_slots:
            opp_stats = slot_best_per_opp.get(slot, [])
            if not opp_stats:
                continue
            survive = 1.0
            sum_ratio = 0.0
            any_priority = False
            for ratio, ko, has_priority in opp_stats:
                survive *= max(0.0, 1.0 - ko)
                sum_ratio += ratio
                if has_priority:
                    any_priority = True
            slot_combined_ko[slot] = 1.0 - survive

            hp_frac = safe_hp_fraction(mine)
            two_turn_damage = sum_ratio * 2.0
            if hp_frac > 0:
                slot_2hko_risk[slot] = min(1.0, two_turn_damage / max(0.05, hp_frac))
            slot_priority_threat[slot] = 1.0 if any_priority else 0.0

        if opp_pressure:
            global_target = max(opp_pressure, key=opp_pressure.get)
            global_pressure = opp_pressure[global_target]
            global_target_ref = opp_ref_by_name.get(global_target)
        else:
            global_target, global_pressure, global_target_ref = "none", 0.0, None

        return ThreatEstimate(
            slot_threat,
            slot_ko_risk,
            threatening_opp,
            threatening_opp_ref,
            global_target,
            global_target_ref,
            float(global_pressure),
            float(spread_threat),
            slot_combined_ko=slot_combined_ko,
            slot_2hko_risk=slot_2hko_risk,
            slot_priority_threat=slot_priority_threat,
        )


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
