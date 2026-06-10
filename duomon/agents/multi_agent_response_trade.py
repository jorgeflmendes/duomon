from __future__ import annotations

from .multi_agent_context import *


class MultiAgentResponseTradeMixin:
    def _counterfactual_trade_value(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
    ) -> float:
        responses = self._predict_opponent_responses(battle)
        own_damage = self._combined_damage_by_opp_slot(my_summary, partner_summary)
        if not responses:
            return 0.0

        value = 0.0
        own_ko_slots = {
            int(slot)
            for slot, damage in own_damage.items()
            if self._damage_reaches_current_hp(
                damage,
                battle.opponent_active_pokemon[int(slot)]
                if int(slot) < len(battle.opponent_active_pokemon)
                else None,
                margin=0.92,
            )
        }
        value += 0.75 * len(own_ko_slots)

        incoming_damage = 0.0
        incoming_kos = 0.0
        prevented_value = 0.0
        preemptive_removals = 0
        for response in responses:
            weight = float(response.get("weight", 1.0) or 1.0)
            raw_damage = utility_damage_sum(
                (response.get("damage_by_slot", {}) or {}).values(), cap=1.25
            )
            raw_kos = float(sum(float(v) for v in (response.get("ko_by_slot", {}) or {}).values()))
            opp_slot = response.get("opp_slot", None)
            neutralized = self._response_fully_neutralized(
                battle, my_summary, partner_summary, response
            )
            if neutralized:
                prevented_value += weight * (0.90 * raw_damage + 1.75 * raw_kos)
                if isinstance(opp_slot, int) and self._opponent_is_preemptively_removed(
                    opp_slot, battle, my_summary, partner_summary, response
                ):
                    preemptive_removals += 1
                    prevented_value += weight * 1.05
                continue

            remaining = self._remaining_response_cost(battle, my_summary, partner_summary, response)
            incoming_damage += weight * raw_damage
            incoming_kos += weight * raw_kos
            value -= weight * (0.85 * remaining + 0.75 * raw_damage + 2.20 * raw_kos)

            if isinstance(opp_slot, int) and opp_slot not in own_ko_slots and raw_kos >= 0.45:
                value -= weight * 0.90

        value += prevented_value
        value += 0.95 * preemptive_removals

        my_target_slot = my_summary.get("target_slot")
        partner_target_slot = partner_summary.get("target_slot")
        try:
            my_target_slot = int(my_target_slot) if my_target_slot is not None else None
            partner_target_slot = (
                int(partner_target_slot) if partner_target_slot is not None else None
            )
        except Exception:
            my_target_slot = None
            partner_target_slot = None

        if my_target_slot is not None and my_target_slot == partner_target_slot:
            focused_damage = float(own_damage.get(my_target_slot, 0.0))
            focused_opp = (
                battle.opponent_active_pokemon[my_target_slot]
                if my_target_slot < len(battle.opponent_active_pokemon)
                else None
            )
            focused_hp = max(
                0.05, safe_hp_fraction(focused_opp) if focused_opp is not None else 1.0
            )
            if focused_damage > focused_hp + 0.45:
                other_response_pressure = 0.0
                for response in responses:
                    opp_slot = response.get("opp_slot", None)
                    if opp_slot == my_target_slot:
                        continue
                    if self._response_fully_neutralized(
                        battle, my_summary, partner_summary, response
                    ):
                        continue
                    other_response_pressure += float(response.get("weight", 1.0) or 1.0) * (
                        utility_damage_sum(
                            (response.get("damage_by_slot", {}) or {}).values(), cap=1.25
                        )
                        + 1.8 * float(sum((response.get("ko_by_slot", {}) or {}).values()))
                    )
                if other_response_pressure >= 0.55:
                    value -= min(
                        2.20, 0.65 * (focused_damage - focused_hp) + 0.75 * other_response_pressure
                    )

        if bool(my_summary.get("protect", False)) or bool(partner_summary.get("protect", False)):
            protected_slots = [
                int(summary.get("slot", -1))
                for summary in (my_summary, partner_summary)
                if bool(summary.get("protect", False))
            ]
            for response in responses:
                targets = [int(slot) for slot in response.get("target_slots", [])]
                if not any(slot in protected_slots for slot in targets):
                    continue
                raw_kos = float(
                    sum(float(v) for v in (response.get("ko_by_slot", {}) or {}).values())
                )
                opp_slot = response.get("opp_slot", None)
                punish = (
                    float(own_damage.get(int(opp_slot), 0.0)) if isinstance(opp_slot, int) else 0.0
                )
                if raw_kos >= 0.45 and punish >= 0.55:
                    value += 1.35 + 0.65 * min(1.0, punish)
                elif punish < 0.35:
                    value -= 0.75

        if incoming_kos >= 0.85 and preemptive_removals == 0:
            value -= 1.20
        if incoming_damage >= 1.35 and not own_ko_slots:
            value -= 0.65
        return float(value)

    @staticmethod
    def _combined_damage_by_opp_slot(
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        cap: float = 1.35,
    ) -> Dict[int, float]:
        combined: Dict[int, float] = {}
        for summary in (my_summary, partner_summary):
            for raw_slot, ratio in (summary.get("damage_by_slot", {}) or {}).items():
                try:
                    slot = int(raw_slot)
                    combined[slot] = combined.get(slot, 0.0) + utility_damage_ratio(ratio, cap=cap)
                except Exception:
                    continue
        return combined

    @staticmethod
    def _current_hp_ko_threshold(opp: Any, fallback: float = 0.95, margin: float = 0.92) -> float:
        if opp is None:
            return fallback
        hp = safe_hp_fraction(opp)
        if hp <= 0.01:
            return 0.01
        return max(0.04, min(1.05, hp * margin))

    @classmethod
    def _damage_reaches_current_hp(cls, damage_ratio: Any, opp: Any, margin: float = 0.92) -> bool:
        try:
            return float(damage_ratio) >= cls._current_hp_ko_threshold(opp, margin=margin)
        except Exception:
            return False

    @staticmethod
    def _turns_to_ko_against_hp(current_hp_ratio_damage: float, current_hp_fraction: float) -> int:
        if current_hp_ratio_damage <= 0:
            return 99
        hp = max(0.04, float(current_hp_fraction or 1.0))
        return max(1, int(math.ceil(hp / max(0.01, current_hp_ratio_damage))))

    @staticmethod
    def _both_low_progress(my_summary: Dict[str, Any], partner_summary: Dict[str, Any]) -> bool:
        def low(summary: Dict[str, Any]) -> bool:
            if summary.get("kind") != "move":
                return False
            mid = str(summary.get("move_id", ""))
            return float(summary.get("bp", 0.0)) <= 0 and mid not in FAKE_OUT_MOVES

        return low(my_summary) and low(partner_summary)

    def _post_rollout_response_cost(
        self,
        battle: MultiBattle,
        own_damage_by_opp_slot: Dict[int, float],
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        response: Dict[str, Any],
    ) -> float:
        opp_slot = response.get("opp_slot", None)
        if isinstance(opp_slot, int):
            opp = (
                battle.opponent_active_pokemon[opp_slot]
                if opp_slot < len(battle.opponent_active_pokemon)
                else None
            )
            if own_damage_by_opp_slot.get(opp_slot, 0.0) >= self._current_hp_ko_threshold(
                opp, fallback=0.95, margin=0.92
            ):
                if self._opponent_is_preemptively_removed(
                    opp_slot, battle, my_summary, partner_summary, response
                ):
                    return 0.0

        if isinstance(opp_slot, int) and own_damage_by_opp_slot.get(opp_slot, 0.0) >= 1.0:
            if self._opponent_is_preemptively_removed(
                opp_slot, battle, my_summary, partner_summary, response
            ):
                return 0.0

        cost = self._remaining_response_cost(battle, my_summary, partner_summary, response)
        if isinstance(opp_slot, int):
            damage = own_damage_by_opp_slot.get(opp_slot, 0.0)
            if damage >= 0.70:
                cost *= 0.60
            elif damage >= 0.45:
                cost *= 0.82
        return cost

    def _tempo_denial_bonus(
        self,
        battle: MultiBattle,
        own_damage_by_opp_slot: Dict[int, float],
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
        responses: List[Dict[str, Any]],
    ) -> float:
        bonus = 0.0
        for response in responses:
            opp_slot = response.get("opp_slot", None)
            if not isinstance(opp_slot, int):
                continue
            opp = (
                battle.opponent_active_pokemon[opp_slot]
                if opp_slot < len(battle.opponent_active_pokemon)
                else None
            )
            if own_damage_by_opp_slot.get(opp_slot, 0.0) < self._current_hp_ko_threshold(
                opp, fallback=0.95, margin=0.92
            ):
                continue
            if self._opponent_is_preemptively_removed(
                opp_slot, battle, my_summary, partner_summary, response
            ):
                bonus += 0.85 + 0.45 * float(sum(response.get("ko_by_slot", {}).values()))
        return bonus

    def _predicted_response_swing(
        self,
        battle: MultiBattle,
        my_summary: Dict[str, Any],
        partner_summary: Dict[str, Any],
    ) -> float:
        responses = self._predict_opponent_responses(battle)
        if not responses:
            return 0.0

        swing = 0.0
        for response in responses:
            weight = float(response.get("weight", 1.0))
            prevented = self._response_prevention_value(
                battle, my_summary, partner_summary, response
            )
            remaining = self._remaining_response_cost(battle, my_summary, partner_summary, response)
            swing += weight * (prevented - remaining)
        return swing


__all__ = ["MultiAgentResponseTradeMixin"]
