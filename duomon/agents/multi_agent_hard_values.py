from __future__ import annotations

from .multi_agent_context import *


class MultiAgentHardValueMixin:
    def _paper_rollout_value(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
    ) -> float:
        own_damage = self._combined_damage_by_opp_slot(my_summary, partner_summary)
        if not own_damage:
            return -0.25 if self._both_low_progress(my_summary, partner_summary) else 0.0

        value = 0.0
        opp_slots = self._live_or_blind_opp_slots(battle, own_damage)
        for slot, opp in opp_slots:
            combined = float(own_damage.get(slot, 0.0))
            if combined <= 0.0:
                continue
            hp_fraction = safe_hp_fraction(opp) if opp is not None else 1.0
            value += 0.95 * min(1.0, combined)
            ttk = self._turns_to_ko_against_hp(combined, hp_fraction)
            if ttk == 1:
                value += 2.60 + 0.45 * max(0.0, 1.0 - hp_fraction)
            elif ttk == 2:
                value += 1.10
            elif ttk == 3:
                value += 0.45
            else:
                value += 0.12
            if combined > hp_fraction + 0.45:
                value -= 0.22 * min(1.0, combined - hp_fraction - 0.45)

        if len([d for d in own_damage.values() if d > 0.0]) >= 2:
            value += 0.30

        responses = self._predict_opponent_responses(battle)
        response_cost = 0.0
        for response in responses:
            response_cost += float(response.get("weight", 1.0)) * self._post_rollout_response_cost(
                battle,
                own_damage,
                my_summary,
                partner_summary,
                response,
            )
        value -= 0.80 * response_cost
        value += self._tempo_denial_bonus(
            battle, own_damage, my_summary, partner_summary, responses
        )

        if self._both_low_progress(my_summary, partner_summary):
            value -= 1.40
        return value

    def _hard_benchmark_survival_value(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
    ) -> float:
        if self.config.benchmark_type != "vs_abyssal":
            return 0.0

        responses = self._predict_opponent_responses(battle)
        if not responses:
            return 0.0

        own_damage = self._combined_damage_by_opp_slot(my_summary, partner_summary)
        own_kos = 0
        for slot, damage in own_damage.items():
            opp = (
                battle.opponent_active_pokemon[int(slot)]
                if int(slot) < len(battle.opponent_active_pokemon)
                else None
            )
            if self._damage_reaches_current_hp(damage, opp, margin=0.92):
                own_kos += 1

        value = 0.25 * own_kos
        unresolved_ko_pressure = 0.0
        unresolved_damage_pressure = 0.0
        for response in responses:
            weight = float(response.get("weight", 1.0) or 1.0)
            target_slots = [
                int(slot) for slot in response.get("target_slots", []) if isinstance(slot, int)
            ]
            if not target_slots:
                continue

            cost = 0.0
            for slot in target_slots:
                mine = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
                hp = max(0.05, safe_hp_fraction(mine))
                damage = utility_damage_ratio(
                    (response.get("damage_by_slot", {}) or {}).get(slot, 0.0), cap=1.35
                )
                ko = float((response.get("ko_by_slot", {}) or {}).get(slot, 0.0) or 0.0)
                effective = damage / hp
                cost += 0.70 * min(2.0, effective) + 1.45 * ko
                if hp <= 0.50:
                    cost += 0.30 * min(1.5, effective) + 0.35 * ko
                if ko >= 0.45 or effective >= 0.92:
                    unresolved_ko_pressure += weight
                unresolved_damage_pressure += weight * min(1.4, effective)

            if self._response_fully_neutralized(battle, my_summary, partner_summary, response):
                value += weight * (0.65 + 0.45 * cost)
                continue

            opp_slot = response.get("opp_slot", None)
            if isinstance(opp_slot, int):
                pressure = float(own_damage.get(opp_slot, 0.0) or 0.0)
                opp = (
                    battle.opponent_active_pokemon[opp_slot]
                    if opp_slot < len(battle.opponent_active_pokemon)
                    else None
                )
                if pressure >= self._current_hp_ko_threshold(opp, fallback=0.95, margin=0.92):

                    cost *= 0.70
                elif pressure >= 0.65 * max(
                    0.05, safe_hp_fraction(opp) if opp is not None else 1.0
                ):
                    cost *= 0.85

            value -= weight * cost

        if unresolved_ko_pressure >= 1.0 and own_kos == 0:
            value -= 0.80
        if unresolved_ko_pressure >= 1.5:
            value -= 0.75
        if unresolved_damage_pressure >= 2.0 and own_kos <= 1:
            value -= 0.45
        if bool(my_summary.get("protect", False)) or bool(partner_summary.get("protect", False)):
            protected_slots = {
                int(summary.get("slot", -1))
                for summary in (my_summary, partner_summary)
                if bool(summary.get("protect", False))
            }
            if protected_slots:
                protected_targeted = any(
                    any(
                        int(slot) in protected_slots
                        for slot in response.get("target_slots", [])
                        if isinstance(slot, int)
                    )
                    for response in responses
                )
                if not protected_targeted:
                    value -= 0.65
        return float(max(-5.0, min(5.0, value)))

    def _hard_benchmark_pair_value(
        self,
        battle: MultiBattle,
        threat: ThreatEstimate,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
    ) -> float:
        if self.config.benchmark_type not in {"vs_simpleheuristics", "vs_abyssal"}:
            return 0.0

        own_damage = self._combined_damage_by_opp_slot(my_summary, partner_summary)
        live_slots = self._live_or_blind_opp_slots(battle, own_damage)
        if not live_slots:
            return -0.25 if self._both_low_progress(my_summary, partner_summary) else 0.0

        value = 0.0
        ko_slots: set[int] = set()
        pressure_slots = 0
        for slot, opp in live_slots:
            damage = max(0.0, float(own_damage.get(slot, 0.0) or 0.0))
            if damage <= 0.0:
                continue
            hp = safe_hp_fraction(opp) if opp is not None else 1.0
            hp = max(0.05, hp)
            effective = damage / hp
            slot_threat = utility_damage_ratio(threat.slot_threat.get(slot, 0.0), cap=1.35)
            value += 1.20 * min(1.8, effective)
            if damage >= 0.92 * hp:
                ko_slots.add(int(slot))
                value += 4.40 + 1.15 * slot_threat + 0.85 * max(0.0, 1.0 - hp)
            elif damage >= 0.70 * hp:
                value += 1.70 + 0.55 * slot_threat
            elif damage >= 0.45 * hp:
                value += 0.70
            if damage >= min(0.40, 0.55 * hp):
                pressure_slots += 1
            if damage > hp + 0.35:
                value -= 0.85 * min(2.0, damage - hp - 0.35)

        if len(live_slots) >= 2 and pressure_slots >= 2:
            value += 0.80
        if len(ko_slots) >= 2:
            value += 2.30

        my_target_slot = self._coerce_slot(my_summary.get("target_slot"))
        partner_target_slot = self._coerce_slot(partner_summary.get("target_slot"))
        my_damage_by_slot = {
            self._coerce_slot(slot): utility_damage_ratio(value)
            for slot, value in (my_summary.get("damage_by_slot", {}) or {}).items()
        }
        partner_damage_by_slot = {
            self._coerce_slot(slot): utility_damage_ratio(value)
            for slot, value in (partner_summary.get("damage_by_slot", {}) or {}).items()
        }

        if my_target_slot is not None and my_target_slot == partner_target_slot:
            opp = (
                battle.opponent_active_pokemon[my_target_slot]
                if my_target_slot < len(battle.opponent_active_pokemon)
                else None
            )
            hp = max(0.05, safe_hp_fraction(opp) if opp is not None else 1.0)
            my_damage = float(my_damage_by_slot.get(my_target_slot, 0.0) or 0.0)
            partner_damage = float(partner_damage_by_slot.get(my_target_slot, 0.0) or 0.0)
            focused = my_damage + partner_damage
            solo_covers = max(my_damage, partner_damage) >= 0.92 * hp
            other_slots = [slot for slot, _opp in live_slots if slot != my_target_slot]
            other_pressure = max(
                (float(own_damage.get(slot, 0.0) or 0.0) for slot in other_slots), default=0.0
            )
            if solo_covers and other_slots and other_pressure < 0.30:
                value -= 3.10
            elif focused >= hp + 0.50 and other_slots and other_pressure < 0.25:
                value -= 1.45

        if bool(my_summary.get("support", False)) and bool(partner_summary.get("support", False)):
            value -= 2.20
        if self._both_low_progress(my_summary, partner_summary):
            value -= 2.40

        responses = self._predict_opponent_responses(battle)
        for response in responses:
            weight = float(response.get("weight", 1.0) or 1.0)
            opp_slot = response.get("opp_slot", None)
            response_cost = self._hard_response_cost(battle, response)
            if isinstance(opp_slot, int) and self._hard_opponent_removed_before_response(
                opp_slot,
                battle,
                my_summary,
                partner_summary,
                response,
            ):
                value += weight * (1.35 + 1.05 * response_cost)
                continue
            if self._response_fully_neutralized_by_protect_or_fakeout(
                my_summary, partner_summary, response
            ):
                value += weight * (0.75 + 0.70 * response_cost)
                continue
            value -= weight * (0.95 * response_cost)

        return float(max(-8.0, min(10.0, value)))

    @staticmethod
    def _coerce_slot(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            slot = int(value)
            return slot if slot in {0, 1} else None
        except Exception:
            return None

    @staticmethod
    def _hard_response_cost(battle: MultiBattle, response: Dict[str, Any]) -> float:
        damage_by_slot = response.get("damage_by_slot", {}) or {}
        ko_by_slot = response.get("ko_by_slot", {}) or {}
        cost = 0.0
        for raw_slot in response.get("target_slots", []) or []:
            slot = MultiAgentHardValueMixin._coerce_slot(raw_slot)
            if slot is None:
                continue
            mine = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
            hp = max(0.05, safe_hp_fraction(mine))
            damage = utility_damage_ratio(damage_by_slot.get(slot, 0.0), cap=1.35)
            effective = damage / hp
            ko = float(ko_by_slot.get(slot, 0.0) or 0.0)
            cost += 1.10 * min(1.8, effective) + 2.35 * ko
        return float(cost)

    def _hard_opponent_removed_before_response(
        self,
        opp_slot: int,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> bool:
        if self._opponent_is_fakeouted(opp_slot, my_summary, partner_summary):
            return True
        opp = (
            battle.opponent_active_pokemon[opp_slot]
            if opp_slot < len(battle.opponent_active_pokemon)
            else None
        )
        hp = max(0.05, safe_hp_fraction(opp) if opp is not None else 1.0)
        damage = 0.0
        for summary in (my_summary, partner_summary):
            damage += self._hard_preemptive_damage_into_opp_slot(summary, opp_slot, response)
        return damage >= 0.92 * hp

    @staticmethod
    def _hard_preemptive_damage_into_opp_slot(
        summary: Dict[str, Any],
        opp_slot: int,
        response: Dict[str, Any],
    ) -> float:
        if summary.get("kind") != "move" or float(summary.get("bp", 0.0) or 0.0) <= 0:
            return 0.0
        damage_by_slot = summary.get("damage_by_slot", {}) or {}
        damage = utility_damage_ratio(damage_by_slot.get(opp_slot, 0.0), cap=1.35)
        if damage <= 0.0:
            return 0.0
        my_priority = float(summary.get("priority", 0.0) or 0.0)
        opp_priority = float(response.get("priority", 0.0) or 0.0)
        if my_priority < opp_priority:
            return 0.0
        if my_priority == opp_priority:
            my_speed = float(summary.get("attacker_speed", 0.0) or 0.0)
            opp_speed = float(response.get("attacker_speed", 0.0) or 0.0)
            if my_speed < opp_speed:
                return 0.0
        return damage

    def _response_fully_neutralized_by_protect_or_fakeout(
        self,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> bool:
        opp_slot = response.get("opp_slot", None)
        if isinstance(opp_slot, int) and self._opponent_is_fakeouted(
            opp_slot, my_summary, partner_summary
        ):
            return True
        target_slots = response.get("target_slots", []) or []
        if not target_slots:
            return False
        return all(
            self._slot_is_protected(int(slot), my_summary, partner_summary) for slot in target_slots
        )


__all__ = ["MultiAgentHardValueMixin"]
