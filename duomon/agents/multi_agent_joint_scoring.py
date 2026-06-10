from __future__ import annotations

from .multi_agent_context import *


class MultiAgentJointScoringMixin:
    def _shared_joint_pair_score(
        self,
        battle: MultiBattle,
        threat: ThreatEstimate,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        local_score: float,
        local_rank: int,
        partner_rank: int,
    ) -> Tuple[float, Dict[str, Any]]:
        partner_score = float(partner_summary.get("score", 0.0) or 0.0)
        partner_prior = max(-8.0, min(16.0, partner_score))
        partner_weight = float(getattr(self.config, "shared_partner_score_weight", 0.28) or 0.28)
        pair_bonus = partner_weight * partner_prior
        own_damage = self._combined_damage_by_opp_slot(my_summary, partner_summary)
        live_slots = self._live_or_blind_opp_slots(battle, own_damage)

        capped_pressure = sum(
            min(1.15, max(0.0, float(own_damage.get(slot, 0.0)))) for slot, _ in live_slots
        )
        pair_bonus += 1.10 * capped_pressure
        total_damage = sum(max(0.0, float(value)) for value in own_damage.values())
        if total_damage < 0.05:
            pair_bonus -= 2.50
            if bool(my_summary.get("support", False)):
                pair_bonus -= 4.00
            if bool(partner_summary.get("support", False)):
                pair_bonus -= 1.25

        ko_slots: List[int] = []
        for slot, opp in live_slots:
            combined = float(own_damage.get(slot, 0.0))
            if combined <= 0.0:
                continue
            hp_fraction = safe_hp_fraction(opp) if opp is not None else 1.0
            ko_threshold = self._current_hp_ko_threshold(opp, fallback=0.95, margin=0.92)
            effective = combined / max(0.05, hp_fraction)
            if combined >= ko_threshold:
                ko_slots.append(int(slot))
                pair_bonus += 3.10 + 0.95 * max(0.0, 1.0 - hp_fraction) + 0.55 * min(1.4, effective)
            elif combined >= 0.78 * max(0.05, hp_fraction):
                pair_bonus += 1.35 + 0.45 * max(0.0, 1.0 - hp_fraction) + 0.30 * min(1.2, effective)
            elif combined >= 0.48 * max(0.05, hp_fraction):
                pair_bonus += 0.62
            if hp_fraction <= 0.38 and combined >= 0.55:
                pair_bonus += 0.70
            overkill = combined - max(1.05 * hp_fraction, ko_threshold + 0.35)
            if overkill > 0.0:
                pair_bonus -= 0.75 * min(2.5, overkill)

        threat_slot: Optional[int] = None
        threat_value = 0.0
        for slot, _opp in live_slots:
            value = utility_damage_ratio(threat.slot_threat.get(slot, 0.0), cap=1.35)
            if value > threat_value:
                threat_slot = int(slot)
                threat_value = value
        if threat_slot is not None and threat_value >= 0.55:
            threat_damage = float(own_damage.get(threat_slot, 0.0))
            pair_bonus += threat_value * min(2.40, 2.00 * threat_damage)
            threat_opp = (
                battle.opponent_active_pokemon[threat_slot]
                if threat_slot < len(battle.opponent_active_pokemon)
                else None
            )
            if threat_damage >= self._current_hp_ko_threshold(
                threat_opp, fallback=0.95, margin=0.92
            ):
                pair_bonus += 1.65 * threat_value
            elif ko_slots and threat_damage < 0.35:
                pair_bonus -= 2.20 * threat_value

        my_target_slot = my_summary.get("target_slot")
        partner_target_slot = partner_summary.get("target_slot")
        try:
            my_target_slot = int(my_target_slot) if my_target_slot is not None else None
        except Exception:
            my_target_slot = None
        try:
            partner_target_slot = (
                int(partner_target_slot) if partner_target_slot is not None else None
            )
        except Exception:
            partner_target_slot = None

        my_damage_total = float(my_summary.get("damage_sum", 0.0) or 0.0)
        partner_damage_total = float(partner_summary.get("damage_sum", 0.0) or 0.0)
        my_damage_by_slot = {
            int(slot): utility_damage_ratio(value)
            for slot, value in (my_summary.get("damage_by_slot", {}) or {}).items()
        }
        partner_damage_by_slot = {
            int(slot): utility_damage_ratio(value)
            for slot, value in (partner_summary.get("damage_by_slot", {}) or {}).items()
        }
        local_damage_target = (
            float(my_damage_by_slot.get(my_target_slot, 0.0)) if my_target_slot is not None else 0.0
        )
        partner_damage_target = (
            float(partner_damage_by_slot.get(partner_target_slot, 0.0))
            if partner_target_slot is not None
            else 0.0
        )
        my_is_spread = bool(my_summary.get("spread", False))
        partner_is_spread = bool(partner_summary.get("spread", False))
        split_target_value = 0.0
        split_ko_count = 0
        split_pressure_count = 0
        if my_target_slot is not None and my_target_slot == partner_target_slot:
            focused = float(own_damage.get(my_target_slot, 0.0))
            focused_opp = (
                battle.opponent_active_pokemon[my_target_slot]
                if my_target_slot < len(battle.opponent_active_pokemon)
                else None
            )
            focused_hp = max(
                0.05, safe_hp_fraction(focused_opp) if focused_opp is not None else 1.0
            )
            focused_ko_threshold = self._current_hp_ko_threshold(
                focused_opp, fallback=0.95, margin=0.92
            )
            if focused >= focused_ko_threshold:
                pair_bonus += 0.85
            elif focused < 0.35 * focused_hp and not ko_slots:
                pair_bonus -= 0.45
            same_single_target = not my_is_spread and not partner_is_spread
            my_focused_damage = my_damage_by_slot.get(my_target_slot, 0.0)
            partner_focused_damage = partner_damage_by_slot.get(my_target_slot, 0.0)
            solo_kill_already_covered = (
                max(my_focused_damage, partner_focused_damage) >= focused_ko_threshold
            )
            other_live_slots = [
                slot
                for slot, _opp in live_slots
                if slot != my_target_slot and float(own_damage.get(slot, 0.0)) < 0.35
            ]
            if same_single_target and solo_kill_already_covered:


                waste = max(0.0, focused - 1.05)
                pair_bonus -= 3.20 * min(1.80, waste)
                if min(my_focused_damage, partner_focused_damage) >= 0.25 * focused_hp:
                    pair_bonus -= 1.25
                if other_live_slots:
                    pair_bonus -= 1.65
            if (
                focused > focused_hp + 0.35
                and max(my_focused_damage, partner_focused_damage) >= focused_ko_threshold
                and min(my_focused_damage, partner_focused_damage) >= 0.25 * focused_hp
            ):
                pair_bonus -= 1.05 * min(2.5, focused - focused_hp)
            if threat_slot is not None and threat_slot != my_target_slot and focused > 1.20:
                ignored_threat_damage = float(own_damage.get(threat_slot, 0.0))
                if ignored_threat_damage < 0.25:
                    pair_bonus -= 1.35 * threat_value
        elif my_target_slot is not None and partner_target_slot is not None:
            visible_two = len([opp for _slot, opp in live_slots if opp is not None]) >= 2
            if visible_two and not my_is_spread and not partner_is_spread:
                per_actor = [
                    (my_target_slot, float(my_damage_by_slot.get(my_target_slot, 0.0) or 0.0)),
                    (
                        partner_target_slot,
                        float(partner_damage_by_slot.get(partner_target_slot, 0.0) or 0.0),
                    ),
                ]
                effective_values: List[float] = []
                ko_count = 0
                pressure_count = 0
                covered_threat = 0.0
                for slot, damage in per_actor:
                    opp = (
                        battle.opponent_active_pokemon[slot]
                        if slot < len(battle.opponent_active_pokemon)
                        else None
                    )
                    hp = max(0.05, safe_hp_fraction(opp) if opp is not None else 1.0)
                    effective = damage / hp
                    effective_values.append(effective)
                    if damage >= self._current_hp_ko_threshold(opp, fallback=0.95, margin=0.92):
                        ko_count += 1
                    if effective >= 0.42:
                        pressure_count += 1
                    covered_threat += utility_damage_ratio(
                        threat.slot_threat.get(slot, 0.0), cap=1.35
                    ) * min(1.0, effective)
                if pressure_count >= 2:
                    split_target_value += 0.80 + 0.45 * min(2.0, sum(effective_values))
                if ko_count >= 2:
                    split_target_value += 3.80
                elif ko_count == 1 and pressure_count >= 2:
                    split_target_value += 1.65
                if covered_threat >= 0.60:
                    split_target_value += 0.55 * min(2.0, covered_threat)
                if pressure_count <= 1:
                    split_target_value -= 0.55
                pair_bonus += (
                    float(getattr(self.config, "shared_split_target_weight", 1.0) or 0.0)
                    * split_target_value
                )
                split_ko_count = ko_count
                split_pressure_count = pressure_count
                partial_split_penalty = float(
                    getattr(self.config, "shared_partial_split_penalty_weight", 0.0) or 0.0
                )
                if partial_split_penalty > 0.0 and ko_count < 2:
                    pair_bonus -= partial_split_penalty * (1.0 + 0.35 * max(0, 2 - pressure_count))
            elif my_damage_total >= 0.42 and partner_damage_total >= 0.42:
                pair_bonus += 0.38

        my_support = bool(my_summary.get("support", False))
        partner_support = bool(partner_summary.get("support", False))
        my_protect = bool(my_summary.get("protect", False))
        partner_protect = bool(partner_summary.get("protect", False))
        my_mid = str(my_summary.get("move_id", ""))
        partner_mid = str(partner_summary.get("move_id", ""))
        tempo_support_moves = (
            FAKE_OUT_MOVES
            | SPEED_CONTROL_MOVES
            | REDIRECTION_MOVES
            | STATUS_CONTROL_MOVES
            | PIVOT_MOVES
        )
        my_tempo_support = my_mid in tempo_support_moves
        partner_tempo_support = partner_mid in tempo_support_moves
        my_sacrifice = bool(my_summary.get("self_sacrifice", my_mid in SELF_SACRIFICE_MOVES))
        partner_sacrifice = bool(
            partner_summary.get("self_sacrifice", partner_mid in SELF_SACRIFICE_MOVES)
        )
        if my_support and partner_support:
            pair_bonus -= 0.45 if (my_tempo_support or partner_tempo_support) else 2.10
        if my_protect and partner_protect:
            pair_bonus -= 2.75
        if my_support and partner_damage_total < 0.35 and not my_tempo_support:
            pair_bonus -= 0.75
        if partner_support and my_damage_total < 0.35 and not partner_tempo_support:
            pair_bonus -= 0.55
        if my_sacrifice or partner_sacrifice:
            sacrifice_count = int(my_sacrifice) + int(partner_sacrifice)
            decisive = len(ko_slots) >= max(1, sacrifice_count + 1)
            if decisive:
                pair_bonus += 0.45 * len(ko_slots)
            else:
                pair_bonus -= 3.20 * sacrifice_count
                if total_damage < 1.45:
                    pair_bonus -= 1.40 * sacrifice_count

        if my_mid in FAKE_OUT_MOVES and my_target_slot is not None:
            pair_bonus += 1.10 + 0.45 * float(threat.slot_threat.get(my_target_slot, 0.0) >= 0.65)
        if partner_mid in FAKE_OUT_MOVES and partner_target_slot is not None:
            pair_bonus += 0.85 + 0.35 * float(
                threat.slot_threat.get(partner_target_slot, 0.0) >= 0.65
            )

        pair_bonus += 0.70 * self._threat_resolution_bonus(
            threat,
            my_summary,
            partner_summary,
            my_summary.get("target"),
            partner_summary.get("target"),
            dict(my_summary.get("damage", {})),
            dict(partner_summary.get("damage", {})),
            my_support,
            partner_support,
        )
        pair_bonus += 0.55 * self._predicted_response_swing(battle, my_summary, partner_summary)
        pair_bonus += 0.65 * self._paper_rollout_value(battle, my_summary, partner_summary)
        trade_value = self._counterfactual_trade_value(battle, my_summary, partner_summary)
        pair_bonus += 0.95 * trade_value
        survival_value = self._hard_benchmark_survival_value(battle, my_summary, partner_summary)
        pair_bonus += (
            float(getattr(self.config, "hard_benchmark_survival_weight", 0.0) or 0.0)
            * survival_value
        )
        hard_value = self._hard_benchmark_pair_value(battle, threat, my_summary, partner_summary)
        pair_bonus += (
            float(getattr(self.config, "hard_benchmark_pair_weight", 1.0) or 0.0) * hard_value
        )

        if total_damage < 0.35 and not (my_mid in FAKE_OUT_MOVES or partner_mid in FAKE_OUT_MOVES):
            pair_bonus -= 1.35

        best_slot: Optional[int] = None
        best_damage = 0.0
        for slot, value in own_damage.items():
            if float(value) > best_damage:
                best_slot = int(slot)
                best_damage = float(value)
        if (
            my_target_slot is not None
            and partner_target_slot is not None
            and my_target_slot != partner_target_slot
            and split_target_value > 0.0
        ):
            reason = "joint-split"
        else:
            reason = (
                "joint-ko"
                if ko_slots
                else ("joint-focus" if best_damage >= 0.75 else "joint-pressure")
            )
        pair_score = float(local_score + pair_bonus - 0.06 * local_rank - 0.03 * partner_rank)
        details = {
            "reason": reason,
            "pair_score": pair_score,
            "pair_bonus": float(pair_bonus),
            "local_score": float(local_score),
            "partner_score": partner_score,
            "local_label": str(my_summary.get("label", "")),
            "partner_label": str(partner_summary.get("label", "")),
            "local_signature": str(my_summary.get("signature", "")),
            "partner_signature": str(partner_summary.get("signature", "")),
            "local_agent": self.agent_name,
            "partner_agent": str(partner_summary.get("source_agent", "partner")),
            "local_move_id": my_mid,
            "partner_move_id": partner_mid,
            "local_target_slot": my_target_slot,
            "partner_target_slot": partner_target_slot,
            "local_damage_target": float(local_damage_target),
            "partner_damage_target": float(partner_damage_target),
            "local_damage_by_slot": {
                int(slot): float(value) for slot, value in my_damage_by_slot.items()
            },
            "partner_damage_by_slot": {
                int(slot): float(value) for slot, value in partner_damage_by_slot.items()
            },
            "local_damage_total": float(my_damage_total),
            "partner_damage_total": float(partner_damage_total),
            "target_slot": best_slot,
            "target_damage": float(best_damage),
            "ko_slots": [int(slot) for slot in ko_slots],
            "combined_damage_by_slot": {
                int(slot): float(value) for slot, value in own_damage.items()
            },
            "trade_value": float(trade_value),
            "survival_value": float(survival_value),
            "hard_benchmark_value": float(hard_value),
            "split_target_value": float(split_target_value),
            "split_ko_count": int(split_ko_count),
            "split_pressure_count": int(split_pressure_count),
        }
        return pair_score, details

    def _joint_combo_bonus(
        self, battle: MultiBattle, action: SlotAction, partner_candidates: List[Dict[str, Any]]
    ) -> float:
        if action.kind != "move" or action.move is None:
            return 0.0
        my_summary = self._candidate_summary(battle, 0.0, action)
        if my_summary["kind"] != "move":
            return 0.0

        threat = self.slot_agent.tactical.threat_model.analyze(battle)
        best_bonus = 0.0
        my_target = my_summary.get("target")
        my_damage = dict(my_summary.get("damage", {}))
        my_is_support = move_base_power(action.move) <= 0
        for partner in partner_candidates:
            partner_target = partner.get("target")
            partner_damage = dict(partner.get("damage", {}))
            partner_is_support = partner.get("bp", 0.0) <= 0
            bonus = 0.0
            if my_target and partner_target and my_target == partner_target:
                combined = utility_damage_ratio(
                    my_damage.get(my_target, 0.0)
                ) + utility_damage_ratio(partner_damage.get(my_target, 0.0))
                bonus += 0.95 + 1.35 * min(1.5, combined)
                if combined >= 1.0:
                    bonus += 1.15
            elif my_target and partner_target and my_target != partner_target:
                bonus -= 0.18

            if my_is_support and partner_is_support:
                bonus -= 0.80
            elif my_is_support and partner_target and partner.get("score", 0.0) > 0.25:
                bonus += 0.20

            if is_spread_move(action.move):
                for target_name, ratio in my_damage.items():
                    combined = utility_damage_ratio(ratio) + utility_damage_ratio(
                        partner_damage.get(target_name, 0.0)
                    )
                    if combined >= 1.0:
                        bonus += 0.40

            combined_by_slot = self._combined_damage_by_opp_slot(my_summary, partner)
            for opp_slot, combined in combined_by_slot.items():
                if opp_slot >= len(battle.opponent_active_pokemon):
                    opp = None
                else:
                    opp = battle.opponent_active_pokemon[opp_slot]
                if opp is not None and is_fainted(opp):
                    continue
                hp = safe_hp_fraction(opp) if opp is not None else 1.0
                if hp <= 0.50:
                    if combined >= 0.85 * hp:
                        bonus += 1.85 + 1.45 * max(0.0, 1.0 - hp)
                    elif (
                        my_summary.get("target_slot") == opp_slot
                        or partner.get("target_slot") == opp_slot
                    ):
                        bonus += 0.35 * max(0.0, 1.0 - hp)

            bonus += self._threat_resolution_bonus(
                threat,
                my_summary,
                partner,
                my_target,
                partner_target,
                my_damage,
                partner_damage,
                my_is_support,
                partner_is_support,
            )
            bonus += self._predicted_response_swing(battle, my_summary, partner)
            bonus += self._paper_rollout_value(battle, my_summary, partner)
            best_bonus = max(best_bonus, bonus)
        return best_bonus


__all__ = ["MultiAgentJointScoringMixin"]
