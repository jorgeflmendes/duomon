from __future__ import annotations

from .multi_agent_context import *


class MultiAgentResponsePredictionMixin:
    def _predict_opponent_responses(self, battle: MultiBattle) -> List[Dict[str, Any]]:
        responses: List[Dict[str, Any]] = []
        for opp_slot, opp in enumerate(battle.opponent_active_pokemon[:2]):
            if opp is None or is_fainted(opp):
                continue
            if self.config.benchmark_type == "vs_abyssal":
                responses.extend(self._predict_abyssal_responses_for_slot(battle, opp_slot, opp))
                continue
            if self.config.benchmark_type == "vs_simpleheuristics":
                responses.extend(
                    self._predict_simpleheuristic_responses_for_slot(battle, opp_slot, opp)
                )
                continue
            maxpower_responses = self._predict_maxpower_responses_for_slot(battle, opp_slot, opp)
            for response in maxpower_responses:
                item = dict(response)
                item["weight"] = 0.40 * float(item.get("weight", 1.0) or 1.0)
                responses.append(item)
            response = self._predict_simpleheuristic_response_for_slot(battle, opp_slot, opp)
            if response is not None:
                item = dict(response)
                item["weight"] = 0.60 * float(item.get("weight", 1.0) or 1.0)
                responses.append(item)
        return responses

    def _weighted_response_candidates(
        self,
        candidates: List[Tuple[float, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        try:
            top_k = max(1, int(getattr(self.config, "opponent_response_top_k", 1) or 1))
        except Exception:
            top_k = 1
        margin = max(0.0, float(getattr(self.config, "opponent_response_margin", 0.0) or 0.0))
        best_score = float(candidates[0][0])
        selected = [
            item
            for item in candidates[:top_k]
            if margin <= 0.0 or float(item[0]) >= best_score - margin
        ]
        if not selected:
            selected = [candidates[0]]
        if len(selected) == 1:
            item = dict(selected[0][1])
            item["weight"] = float(item.get("weight", 1.0) or 1.0)
            return [item]
        temp = max(0.05, float(getattr(self.config, "opponent_response_softmax_temp", 1.0) or 1.0))
        raw_weights = [
            math.exp(max(-40.0, min(0.0, (float(score) - best_score) / temp)))
            for score, _ in selected
        ]
        total = sum(raw_weights) or 1.0
        weighted: List[Dict[str, Any]] = []
        for raw, (_score, response) in zip(raw_weights, selected):
            item = dict(response)
            base_weight = float(item.get("weight", 1.0) or 1.0)
            item["weight"] = base_weight * raw / total
            weighted.append(item)
        return weighted

    def _predict_maxpower_responses_for_slot(
        self, battle: MultiBattle, opp_slot: int, opp: Any
    ) -> List[Dict[str, Any]]:
        damaging = [
            move
            for move in OpponentThreatModel._predicted_moves_for_opp(opp)
            if move_base_power(move) > 0
        ]
        if not damaging:
            return []

        best_score = max(
            move_base_power(move) + 5.0 * move_accuracy(move) + 2.0 * move_priority(move)
            for move in damaging
        )
        best_moves = [
            move
            for move in damaging
            if abs(
                (move_base_power(move) + 5.0 * move_accuracy(move) + 2.0 * move_priority(move))
                - best_score
            )
            <= 1e-9
        ]
        if not best_moves:
            return []

        best_moves.sort(key=move_id)
        chosen = best_moves[0]
        if is_spread_move(chosen):
            return [
                self._response_from_move(
                    battle, opp_slot, opp, chosen, target_slot=None, weight=1.0
                )
            ]

        responses = []
        target_slots = [
            slot
            for slot, mine in enumerate(battle.active_pokemon[:2])
            if mine is not None and not is_fainted(mine)
        ]
        if not target_slots:
            return []
        weight = 1.0 / len(target_slots)
        for target_slot in target_slots:
            responses.append(
                self._response_from_move(
                    battle, opp_slot, opp, chosen, target_slot=target_slot, weight=weight
                )
            )
        return responses

    def _predict_simpleheuristic_response_for_slot(
        self,
        battle: MultiBattle,
        opp_slot: int,
        opp: Any,
    ) -> Optional[Dict[str, Any]]:
        responses = self._predict_simpleheuristic_responses_for_slot(battle, opp_slot, opp)
        return responses[0] if responses else None

    def _predict_simpleheuristic_responses_for_slot(
        self,
        battle: MultiBattle,
        opp_slot: int,
        opp: Any,
    ) -> List[Dict[str, Any]]:
        candidate_responses: List[Tuple[float, Dict[str, Any]]] = []
        move_pool = OpponentThreatModel._predicted_moves_for_opp(opp)
        for move in move_pool:
            if move is None:
                continue
            if is_spread_move(move):
                response = self._response_from_move(
                    battle, opp_slot, opp, move, target_slot=None, weight=1.0
                )
                candidate_responses.append(
                    (self._simpleheuristic_response_score(battle, opp, response), response)
                )
                continue
            legal_targets = [
                slot
                for slot, mine in enumerate(battle.active_pokemon[:2])
                if mine is not None and not is_fainted(mine)
            ]
            if move_base_power(move) <= 0 and not legal_targets:
                legal_targets = [0]
            if not legal_targets:
                continue
            for target_slot in legal_targets:
                response = self._response_from_move(
                    battle, opp_slot, opp, move, target_slot=target_slot, weight=1.0
                )
                candidate_responses.append(
                    (self._simpleheuristic_response_score(battle, opp, response), response)
                )

        return self._weighted_response_candidates(candidate_responses)

    def _predict_abyssal_responses_for_slot(
        self,
        battle: MultiBattle,
        opp_slot: int,
        opp: Any,
    ) -> List[Dict[str, Any]]:
        candidate_responses: List[Tuple[float, Dict[str, Any]]] = []
        move_pool = OpponentThreatModel._predicted_moves_for_opp(opp)
        for move in move_pool:
            if move is None:
                continue
            if move_base_power(move) <= 0:
                response = self._response_from_move(
                    battle, opp_slot, opp, move, target_slot=0, weight=1.0
                )
                candidate_responses.append(
                    (self._abyssal_response_score(battle, opp, response), response)
                )
                continue
            if is_spread_move(move) or move_target_type(move) in NO_EXPLICIT_TARGETS:
                response = self._response_from_move(
                    battle, opp_slot, opp, move, target_slot=None, weight=1.0
                )
                candidate_responses.append(
                    (self._abyssal_response_score(battle, opp, response), response)
                )
                continue
            legal_targets = [
                slot
                for slot, mine in enumerate(battle.active_pokemon[:2])
                if mine is not None and not is_fainted(mine)
            ]
            for target_slot in legal_targets:
                response = self._response_from_move(
                    battle, opp_slot, opp, move, target_slot=target_slot, weight=1.0
                )
                candidate_responses.append(
                    (self._abyssal_response_score(battle, opp, response), response)
                )
        return self._weighted_response_candidates(candidate_responses)

    def _abyssal_response_score(
        self, battle: MultiBattle, opp: Any, response: Dict[str, Any]
    ) -> float:
        move = response.get("move", None)
        if move is None:
            return -10.0
        mid = move_id(move)
        if move_base_power(move) <= 0:
            turn = int(safe_getattr(battle, "turn", 0) or 0)
            if mid in PROTECT_MOVES:
                return 1.25 if safe_hp_fraction(opp) < 0.35 else (-0.55 if turn <= 1 else -0.20)
            if mid in FAKE_OUT_MOVES and turn <= 2:
                return 1.75
            if mid in SPEED_CONTROL_MOVES:
                enemy_speed = np.mean(
                    [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon)]
                    or [100]
                )
                ally_speed = np.mean(
                    [safe_speed(m) for m in active_alive_mons(battle.active_pokemon)] or [100]
                )
                return 1.20 if enemy_speed + 10 < ally_speed else 0.25
            if mid in HAZARD_MOVES:
                return -0.85
            if mid in SETUP_MOVES:
                return 0.20 if turn <= 1 and safe_hp_fraction(opp) > 0.85 else -0.55
            return -0.35

        damage_by_slot = response.get("damage_by_slot", {}) or {}
        ko_by_slot = response.get("ko_by_slot", {}) or {}
        damage_sum = utility_damage_sum(damage_by_slot.values(), cap=1.25)
        ko_sum = float(sum(float(v) for v in ko_by_slot.values()))
        mult = 0.0
        low_hp = 0.0
        for slot in response.get("target_slots", []) or []:
            mine = (
                battle.active_pokemon[int(slot)] if int(slot) < len(battle.active_pokemon) else None
            )
            if mine is None:
                continue
            mult = max(mult, damage_multiplier(mine, move))
            low_hp += max(0.0, 1.0 - safe_hp_fraction(mine))
        return (
            5.40 * damage_sum
            + 4.10 * ko_sum
            + 0.85 * min(4.0, mult)
            + 0.55 * float(has_type(opp, move))
            + 0.42 * move_accuracy(move)
            + 0.28 * move_priority(move)
            + 0.75 * low_hp
            + 0.012 * min(150.0, move_base_power(move))
        )

    def _simpleheuristic_response_score(
        self, battle: MultiBattle, opp: Any, response: Dict[str, Any]
    ) -> float:
        move = response.get("move", None)
        if move is None:
            return -10.0
        if bool(getattr(self.config, "simpleheuristics_exact_response_score", False)):
            return self._exact_simpleheuristic_response_score(battle, opp, response)
        mid = move_id(move)
        priority = float(response.get("priority", 0.0))
        accuracy = move_accuracy(move)
        damage_by_slot = response.get("damage_by_slot", {})
        ko_by_slot = response.get("ko_by_slot", {})

        if move_base_power(move) > 0:
            damage_sum = utility_damage_sum(damage_by_slot.values(), cap=1.25)
            ko_sum = float(sum(float(v) for v in ko_by_slot.values()))
            focus_hp = 0.0
            for slot in response.get("target_slots", []):
                mine = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
                focus_hp += max(0.0, 1.0 - safe_hp_fraction(mine))
            return (
                5.10 * damage_sum
                + 3.90 * ko_sum
                + 0.32 * priority
                + 0.15 * accuracy
                + 0.45 * focus_hp
            )

        turn = int(safe_getattr(battle, "turn", 0) or 0)
        if mid in FAKE_OUT_MOVES:
            value = 2.35 if turn <= 2 else -1.50
            target_slot = response.get("target_slots", [None])[0]
            if isinstance(target_slot, int):
                mine = (
                    battle.active_pokemon[target_slot]
                    if target_slot < len(battle.active_pokemon)
                    else None
                )
                threat = (
                    self.slot_agent._best_available_attack_score(battle)
                    if mine is not None
                    else 0.0
                )
                value += 0.45 * threat
            return value
        if mid in PROTECT_MOVES:
            own_hp = safe_hp_fraction(opp)
            return 0.55 if own_hp < 0.35 else -0.40
        if mid in SPEED_CONTROL_MOVES:
            my_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon)] or [100]
            )
            opp_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.active_pokemon)] or [100]
            )
            return 1.20 if my_speed < opp_speed else 0.15
        if mid in REDIRECTION_MOVES:
            return 0.70
        if mid in HELPING_HAND_MOVES:
            return 0.55
        return -0.35

    @staticmethod
    def _exact_simpleheuristic_response_score(
        battle: MultiBattle, opp: Any, response: Dict[str, Any]
    ) -> float:
        move = response.get("move", None)
        if move is None:
            return -10.0
        bp = move_base_power(move)
        if bp <= 0:
            mid = move_id(move)
            turn = int(safe_getattr(battle, "turn", 0) or 0)
            if mid in HAZARD_MOVES:
                return 0.0
            if mid in SETUP_MOVES and safe_hp_fraction(opp) >= 0.999:
                return 0.0
            if mid in FAKE_OUT_MOVES and turn <= 2:
                return 0.01
            return -1.0

        target_slots = [
            int(slot)
            for slot in (response.get("target_slots", []) or [])
            if isinstance(slot, int) and 0 <= int(slot) < len(battle.active_pokemon)
        ]
        if not target_slots:
            return -10.0

        category = move_category(move)
        best = -1e9
        for slot in target_slots:
            target = battle.active_pokemon[slot]
            if target is None or is_fainted(target):
                continue
            try:
                physical_ratio = SimpleHeuristicsPlayer._stat_estimation(opp, "atk") / max(
                    1.0,
                    SimpleHeuristicsPlayer._stat_estimation(target, "def"),
                )
                special_ratio = SimpleHeuristicsPlayer._stat_estimation(opp, "spa") / max(
                    1.0,
                    SimpleHeuristicsPlayer._stat_estimation(target, "spd"),
                )
            except Exception:
                physical_ratio = safe_stat(opp, "atk", 100) / max(
                    1.0, safe_stat(target, "def", 100)
                )
                special_ratio = safe_stat(opp, "spa", 100) / max(1.0, safe_stat(target, "spd", 100))
            stat_ratio = physical_ratio if category == "physical" else special_ratio
            score = (
                bp
                * (1.5 if has_type(opp, move) else 1.0)
                * stat_ratio
                * move_accuracy(move)
                * move_expected_hits(move)
                * damage_multiplier(target, move)
            )
            if (
                move_target_type(move) not in {"normal", "any"}
                and len(active_alive_mons(battle.active_pokemon[:2])) >= 2
            ):
                score *= 1.5
            best = max(best, float(score))
        return best if math.isfinite(best) else -10.0

    def _response_from_move(
        self,
        battle: MultiBattle,
        opp_slot: int,
        opp: Any,
        move: Any,
        target_slot: Optional[int],
        weight: float,
    ) -> Dict[str, Any]:
        target_slots = (
            [
                slot
                for slot, mine in enumerate(battle.active_pokemon[:2])
                if mine is not None and not is_fainted(mine)
            ]
            if is_spread_move(move)
            else ([] if target_slot is None else [target_slot])
        )
        damage_by_slot: Dict[int, float] = {}
        ko_by_slot: Dict[int, float] = {}
        if move_base_power(move) > 0:
            for slot in target_slots:
                mine = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
                if mine is None:
                    continue
                spread = is_spread_move(move)
                ratio = self._scaled_opponent_damage_ratio(
                    _advanced_damage_ratio(battle, move, opp, mine, spread=spread)
                )
                damage_by_slot[slot] = ratio
                ko_by_slot[slot] = self._ko_probability_from_scaled_ratio(ratio, mine)
        return {
            "opp_slot": opp_slot,
            "opp_species": safe_species(opp),
            "move": move,
            "move_id": move_id(move),
            "priority": float(move_priority(move)),
            "attacker_speed": float(safe_speed(opp)),
            "target_slots": target_slots,
            "damage_by_slot": damage_by_slot,
            "ko_by_slot": ko_by_slot,
            "weight": float(weight),
        }

    def _response_prevention_value(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> float:
        prevented = 0.0
        target_slots = [int(slot) for slot in response.get("target_slots", [])]
        for slot in target_slots:
            if self._slot_is_protected(slot, my_summary, partner_summary):
                prevented += 0.90 * utility_damage_ratio(
                    response.get("damage_by_slot", {}).get(slot, 0.0), cap=1.25
                )
                prevented += 0.65 * float(response.get("ko_by_slot", {}).get(slot, 0.0))

        opp_slot = response.get("opp_slot", None)
        if isinstance(opp_slot, int):
            if self._opponent_is_fakeouted(opp_slot, my_summary, partner_summary):
                prevented += 1.05 + 0.75 * utility_damage_sum(
                    response.get("damage_by_slot", {}).values(), cap=1.25
                )
            elif self._opponent_is_preemptively_removed(
                opp_slot, battle, my_summary, partner_summary, response
            ):
                prevented += 1.20 + 0.95 * utility_damage_sum(
                    response.get("damage_by_slot", {}).values(), cap=1.25
                )
        return prevented

    def _remaining_response_cost(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> float:
        if self._response_fully_neutralized(battle, my_summary, partner_summary, response):
            return 0.0

        cost = 0.0
        damage_by_slot = response.get("damage_by_slot", {})
        ko_by_slot = response.get("ko_by_slot", {})
        for slot in response.get("target_slots", []):
            slot = int(slot)
            mine = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
            hp_weight = 1.0 + max(0.0, 0.60 - safe_hp_fraction(mine))
            cost += hp_weight * utility_damage_ratio(damage_by_slot.get(slot, 0.0), cap=1.25)
            cost += (1.20 + 0.50 * (1.0 - safe_hp_fraction(mine))) * float(
                ko_by_slot.get(slot, 0.0)
            )
        return cost

    def _response_fully_neutralized(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> bool:
        opp_slot = response.get("opp_slot", None)
        if isinstance(opp_slot, int):
            if self._opponent_is_fakeouted(opp_slot, my_summary, partner_summary):
                return True
            if self._opponent_is_preemptively_removed(
                opp_slot, battle, my_summary, partner_summary, response
            ):
                return True
        target_slots = response.get("target_slots", []) or []
        if not target_slots:
            return False
        return all(
            self._slot_is_protected(int(slot), my_summary, partner_summary) for slot in target_slots
        )

    @staticmethod
    def _slot_is_protected(
        slot: int, my_summary: Dict[str, Any], partner_summary: Dict[str, Any]
    ) -> bool:
        for summary in (my_summary, partner_summary):
            if int(summary.get("slot", -1)) == slot and summary.get("move_id", "") in PROTECT_MOVES:
                return True
        return False

    @staticmethod
    def _opponent_is_fakeouted(
        opp_slot: int, my_summary: Dict[str, Any], partner_summary: Dict[str, Any]
    ) -> bool:
        for summary in (my_summary, partner_summary):
            if summary.get("move_id", "") not in FAKE_OUT_MOVES:
                continue
            if summary.get("target_slot", None) == opp_slot:
                return True
        return False

    def _opponent_is_preemptively_removed(
        self,
        opp_slot: int,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> bool:
        total = 0.0
        for summary in (my_summary, partner_summary):
            total += self._preemptive_damage_into_opp_slot(summary, opp_slot, response)
        opp = (
            battle.opponent_active_pokemon[opp_slot]
            if opp_slot < len(battle.opponent_active_pokemon)
            else None
        )
        threshold = self._current_hp_ko_threshold(opp, fallback=0.95, margin=0.92)
        return total >= threshold

    @staticmethod
    def _preemptive_damage_into_opp_slot(
        summary: Dict[str, Any],
        opp_slot: int,
        response: Dict[str, Any],
    ) -> float:
        if summary.get("kind") != "move" or float(summary.get("bp", 0.0)) <= 0:
            return 0.0
        my_priority = float(summary.get("priority", 0.0))
        opp_priority = float(response.get("priority", 0.0))
        my_speed = float(summary.get("attacker_speed", 0.0))
        opp_speed = float(response.get("attacker_speed", 0.0))
        if my_priority < opp_priority:
            return 0.0
        if my_priority == opp_priority and my_speed < opp_speed:
            return 0.0
        damage_by_slot = summary.get("damage_by_slot", {})
        return utility_damage_ratio(damage_by_slot.get(opp_slot, 0.0))

    def _threat_resolution_bonus(
        self,
        threat: ThreatEstimate,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        my_target: Optional[str],
        partner_target: Optional[str],
        my_damage: Dict[str, Any],
        partner_damage: Dict[str, Any],
        my_is_support: bool,
        partner_is_support: bool,
    ) -> float:
        bonus = 0.0
        primary_threat = threat.global_target if threat.global_target != "none" else None
        high_threat = max(
            (utility_damage_ratio(value, cap=1.35) for value in threat.slot_threat.values()),
            default=0.0,
        )
        own_threat = utility_damage_ratio(threat.slot_threat.get(0, 0.0), cap=1.35)

        if primary_threat:
            combined_primary = utility_damage_ratio(
                my_damage.get(primary_threat, 0.0)
            ) + utility_damage_ratio(partner_damage.get(primary_threat, 0.0))
            if combined_primary >= 1.0:
                bonus += 1.25 + 0.75 * min(
                    1.0, utility_damage_ratio(threat.global_pressure, cap=1.35)
                )
            elif combined_primary >= 0.7:
                bonus += 0.55

            if my_target == primary_threat and partner_target == primary_threat:
                bonus += 0.30
            elif my_target == primary_threat or partner_target == primary_threat:
                bonus += 0.15

        if high_threat >= 0.65 and my_is_support and partner_is_support:
            bonus -= 1.10
        elif high_threat >= 0.65 and my_is_support and partner_target != primary_threat:
            bonus -= 0.45

        if move_id(my_summary.get("move", None)) in PROTECT_MOVES:
            if own_threat >= 0.75 and partner_target == primary_threat:
                bonus += 0.55

        return bonus


__all__ = ["MultiAgentResponsePredictionMixin"]
