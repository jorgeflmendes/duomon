from __future__ import annotations

from .multi_opponent_context import *
from .multi_opponent_utils import *
from .multi_opponent_basic import *
from .multi_opponent_damage import *
from .multi_opponent_abyssal import *


class DamageRaceMultiSlotAgent(MultiAwarePlayerMixin, Player):

    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                if not switches:
                    return DefaultBattleOrder()
                return max(
                    switches,
                    key=lambda order: IndependentTwoSlotAgent._switch_in_pressure_score(
                        safe_getattr(order, "order", None), battle
                    ),
                )

            active = battle.active_pokemon[0] if battle.active_pokemon else None
            moves = normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
            if active is None or is_fainted(active) or not moves:
                return _multi_safe_random_order(self, battle)

            best: Tuple[float, Optional[Any], int] = (-1e9, None, 0)
            for move in moves:
                positions = _legal_showdown_positions(battle, move, active)
                if is_spread_move(move) or move_target_type(move) in NO_EXPLICIT_TARGETS:
                    score = self._damage_race_score(battle, active, move, None)
                    if score > best[0]:
                        best = (score, move, 0)
                    continue
                for position in positions:
                    score = self._damage_race_score(battle, active, move, int(position))
                    if score > best[0]:
                        best = (score, move, int(position))

            if best[1] is None:
                return _multi_safe_random_order(self, battle)
            return self.create_order(best[1], move_target=int(best[2] or 0))
        except Exception:
            return DefaultBattleOrder()

    @classmethod
    def _damage_race_score(
        cls,
        battle: MultiBattle,
        active: Any,
        move: Any,
        position: Optional[int],
    ) -> float:
        mid = move_id(move)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        bp = move_base_power(move)
        if bp <= 0:
            return cls._damage_race_status_score(battle, active, move, position)

        visible_opponents = active_alive_mons(battle.opponent_active_pokemon[:2])
        if not visible_opponents:
            base = (bp / 100.0) * move_accuracy(move) * move_expected_hits(move)
            if has_type(active, move):
                base *= 1.25
            if is_spread_move(move):
                return 2.10 + 1.45 * base
            opp1_pos = int(getattr(battle, "OPPONENT_1_POSITION", 1))
            return base + (1.35 if position == opp1_pos else -0.25)

        if is_spread_move(move):
            score = 0.0
            hit_count = 0
            for slot, opp in enumerate(battle.opponent_active_pokemon[:2]):
                if opp is None or is_fainted(opp):
                    continue
                ratio = _advanced_damage_ratio(battle, move, active, opp, spread=True)
                hp = max(0.05, safe_hp_fraction(opp))
                effective = ratio / hp
                ko = _ko_prob_from_effective(effective)
                score += 3.25 * min(1.7, effective) + 4.10 * ko
                score += 0.80 * cls._target_pressure(battle, opp)
                hit_count += 1
            partner_risk = 0.0
            for ally in active_alive_mons(battle.active_pokemon[:2]):
                if ally is not active and damage_multiplier(ally, move) > 0:
                    partner_risk += damage_multiplier(ally, move)
            return score + 0.55 * hit_count - 1.25 * partner_risk + 0.30 * move_accuracy(move)

        target = LegalActionGenerator._target_from_position(battle, int(position or 0))
        if target is None or is_fainted(target):
            return -20.0
        if _target_is_ally(battle, target):
            return -8.0
        mult = damage_multiplier(target, move)
        if mult == 0:
            return -30.0
        target_slot = _opponent_slot_index(battle, target)
        ratio = _advanced_damage_ratio(battle, move, active, target)
        hp = max(0.05, safe_hp_fraction(target))
        effective = ratio / hp
        ko = _ko_prob_from_effective(effective)
        partner = battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
        partner_damage_same = cls._partner_best_damage_into_slot(battle, partner, target_slot)
        combined = ratio + partner_damage_same
        combined_ko = 1.0 if combined >= 0.92 * hp else 0.0

        other_slot = 1 - int(target_slot or 0) if target_slot in {0, 1} else None
        other = (
            battle.opponent_active_pokemon[other_slot]
            if other_slot is not None and other_slot < len(battle.opponent_active_pokemon)
            else None
        )
        other_partner_damage = cls._partner_best_damage_into_slot(battle, partner, other_slot)
        solo_ko = ratio >= 0.92 * hp
        if (
            solo_ko
            and other is not None
            and not is_fainted(other)
            and other_partner_damage >= 0.55 * max(0.05, safe_hp_fraction(other))
        ):
            split_credit = 0.95
        else:
            split_credit = 0.0

        pressure = cls._target_pressure(battle, target)
        low_hp = max(0.0, 1.0 - hp)
        score = (
            4.35 * min(1.8, effective)
            + 5.25 * ko
            + 2.10 * combined_ko
            + 0.95 * pressure
            + 0.50 * min(4.0, mult)
            + 0.38 * float(has_type(active, move))
            + 0.42 * move_accuracy(move)
            + 0.35 * move_priority(move)
            + 0.70 * low_hp
            + split_credit
        )
        if combined >= hp + 0.45 and not solo_ko:
            score -= 0.70 * min(1.5, combined - hp)
        if mid in RECOIL_MOVES and safe_hp_fraction(active) < 0.55 and ko < 0.65:
            score -= 1.25
        if move_accuracy(move) < 0.78 and ko < 0.70:
            score -= 0.75
        if turn <= 2 and mid in FAKE_OUT_MOVES:
            score += 1.25
        return score

    @classmethod
    def _damage_race_status_score(
        cls,
        battle: MultiBattle,
        active: Any,
        move: Any,
        position: Optional[int],
    ) -> float:
        mid = move_id(move)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        if mid in FAKE_OUT_MOVES and turn <= 2:
            target = LegalActionGenerator._target_from_position(battle, int(position or 0))
            return 2.60 + 1.00 * cls._target_pressure(battle, target)
        if mid in PROTECT_MOVES:
            incoming = CounterAbyssalMultiSlotOpponent._predicted_incoming_to_active(battle, active)
            hp = safe_hp_fraction(active)
            if incoming >= 0.95 and hp < 0.70:
                return 2.20 + 0.75 * incoming
            return -0.45 if turn <= 1 else -0.15
        if mid in SPEED_CONTROL_MOVES:
            ally_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.active_pokemon[:2])] or [100]
            )
            opp_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon[:2])]
                or [100]
            )
            return 1.45 if ally_speed + 20 < opp_speed else 0.10
        if mid in HAZARD_MOVES:
            return -1.20
        if mid in SETUP_MOVES:
            incoming = CounterAbyssalMultiSlotOpponent._predicted_incoming_to_active(battle, active)
            return (
                0.30 if turn <= 1 and incoming < 0.35 and safe_hp_fraction(active) > 0.85 else -0.90
            )
        return -0.35

    @staticmethod
    def _partner_best_damage_into_slot(
        battle: MultiBattle, partner: Any, slot: Optional[int]
    ) -> float:
        if partner is None or slot is None or slot not in {0, 1}:
            return 0.0
        target = (
            battle.opponent_active_pokemon[slot]
            if slot < len(battle.opponent_active_pokemon)
            else None
        )
        if target is None or is_fainted(target):
            return 0.0
        moves = safe_getattr(partner, "moves", {}) or {}
        move_values = moves.values() if isinstance(moves, dict) else moves
        best = 0.0
        for move in move_values:
            if move_base_power(move) <= 0 or damage_multiplier(target, move) == 0:
                continue
            best = max(
                best,
                _advanced_damage_ratio(battle, move, partner, target, spread=is_spread_move(move)),
            )
        return best

    @staticmethod
    def _target_pressure(battle: MultiBattle, target: Any) -> float:
        if target is None:
            return 0.0
        pressure = 0.0
        for move in OpponentThreatModel._predicted_moves_for_opp(target):
            if move_base_power(move) <= 0:
                continue
            spread = is_spread_move(move)
            for ally in active_alive_mons(battle.active_pokemon[:2]):
                pressure = max(
                    pressure, _advanced_damage_ratio(battle, move, target, ally, spread=spread)
                )
        return utility_damage_ratio(pressure, cap=1.35)


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
