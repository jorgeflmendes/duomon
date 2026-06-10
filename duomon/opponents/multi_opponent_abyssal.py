from __future__ import annotations

from .multi_opponent_context import *
from .multi_opponent_utils import *
from .multi_opponent_basic import *
from .multi_opponent_damage import *


class AbyssalStyleMultiSlotOpponent(MultiAwarePlayerMixin, Player):

    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                if not switches:
                    return DefaultBattleOrder()
                return max(switches, key=lambda order: self._switch_order_score(order, battle))

            active = battle.active_pokemon[0] if battle.active_pokemon else None
            moves = normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
            if active is None or is_fainted(active) or not moves:
                return _multi_safe_random_order(self, battle)

            best: Tuple[float, Optional[Any], Optional[int]] = (-1e9, None, None)
            for move in moves:
                positions = _legal_showdown_positions(battle, move, active)
                if move_base_power(move) <= 0:
                    score = self._status_move_score(battle, active, move)
                    if score > best[0]:
                        best = (score, move, positions[0] if positions else 0)
                    continue
                if is_spread_move(move) or move_target_type(move) in NO_EXPLICIT_TARGETS:
                    score = self._spread_move_score(battle, active, move)
                    if score > best[0]:
                        best = (score, move, 0)
                    continue
                for position in positions:
                    if position not in {battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION}:
                        continue
                    target = LegalActionGenerator._target_from_position(battle, int(position))
                    if target is None or is_fainted(target):
                        continue
                    score = self._targeted_move_score(battle, active, move, target)
                    if score > best[0]:
                        best = (score, move, int(position))

            if best[1] is None:
                return _multi_safe_random_order(self, battle)
            return self.create_order(best[1], move_target=int(best[2] or 0))
        except Exception:
            return DefaultBattleOrder()

    @staticmethod
    def _targeted_move_score(battle: MultiBattle, active: Any, move: Any, target: Any) -> float:
        mult = damage_multiplier(target, move)
        if mult == 0:
            return -50.0
        ratio = _advanced_damage_ratio(battle, move, active, target)
        ko = _advanced_ko_probability(battle, move, active, target)
        stab = 1.0 if has_type(active, move) else 0.0
        low_hp = max(0.0, 1.0 - safe_hp_fraction(target))
        return (
            5.40 * ratio
            + 4.10 * ko
            + 0.85 * min(4.0, mult)
            + 0.55 * stab
            + 0.42 * move_accuracy(move)
            + 0.28 * move_priority(move)
            + 0.75 * low_hp
            + 0.012 * min(150.0, move_base_power(move))
        )

    @staticmethod
    def _spread_move_score(battle: MultiBattle, active: Any, move: Any) -> float:
        score = 0.0
        hit_count = 0
        for opp in active_alive_mons(battle.opponent_active_pokemon[:2]):
            mult = damage_multiplier(opp, move)
            if mult == 0:
                score -= 4.0
                continue
            ratio = _advanced_damage_ratio(battle, move, active, opp, spread=True)
            ko = _advanced_ko_probability(battle, move, active, opp, spread=True)
            score += 4.20 * ratio + 3.30 * ko + 0.45 * min(4.0, mult)
            hit_count += 1
        partner_risk = 0.0
        for ally in active_alive_mons(battle.active_pokemon[:2]):
            if ally is not active and damage_multiplier(ally, move) > 0:
                partner_risk += damage_multiplier(ally, move)
        return score + 0.45 * hit_count - 0.85 * partner_risk + 0.20 * move_accuracy(move)

    @staticmethod
    def _status_move_score(battle: MultiBattle, active: Any, move: Any) -> float:
        mid = move_id(move)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        if mid in PROTECT_MOVES:
            if safe_hp_fraction(active) < 0.35:
                return 1.25
            return -0.55 if turn <= 1 else -0.20
        if mid in FAKE_OUT_MOVES and turn <= 2:
            return 1.75
        if mid in SPEED_CONTROL_MOVES:
            ally_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.active_pokemon[:2])] or [100]
            )
            opp_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon[:2])]
                or [100]
            )
            return 1.20 if ally_speed + 10 < opp_speed else 0.25
        if mid in HAZARD_MOVES:
            return -0.85
        if mid in SETUP_MOVES:
            return 0.20 if turn <= 1 and safe_hp_fraction(active) > 0.85 else -0.55
        return -0.35

    @staticmethod
    def _switch_order_score(order: Any, battle: MultiBattle) -> float:
        mon = safe_getattr(order, "order", None)
        if mon is None:
            return -1e9
        score = 1.50 * safe_hp_fraction(mon)
        for opp in active_alive_mons(battle.opponent_active_pokemon[:2]):
            score -= max(
                (
                    damage_multiplier(mon, move)
                    for move in OpponentThreatModel._predicted_moves_for_opp(opp)
                ),
                default=1.0,
            )
            score += max(
                (
                    damage_multiplier(opp, move)
                    for move in (safe_getattr(mon, "moves", {}) or {}).values()
                ),
                default=1.0,
            )
        return score


class CounterAbyssalMultiSlotOpponent(AbyssalStyleMultiSlotOpponent):

    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                if not switches:
                    return DefaultBattleOrder()
                return max(switches, key=lambda order: self._switch_order_score(order, battle))

            active = battle.active_pokemon[0] if battle.active_pokemon else None
            moves = normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
            if active is None or is_fainted(active) or not moves:
                return _multi_safe_random_order(self, battle)

            best: Tuple[float, Optional[Any], Optional[int]] = (-1e9, None, None)
            for move in moves:
                positions = _legal_showdown_positions(battle, move, active)
                if move_base_power(move) <= 0:
                    score = self._counter_status_score(battle, active, move)
                    if score > best[0]:
                        best = (score, move, positions[0] if positions else 0)
                    continue
                if is_spread_move(move) or move_target_type(move) in NO_EXPLICIT_TARGETS:
                    score = self._counter_spread_score(battle, active, move)
                    if score > best[0]:
                        best = (score, move, 0)
                    continue
                for position in positions:
                    if position not in {battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION}:
                        continue
                    target = LegalActionGenerator._target_from_position(battle, int(position))
                    if target is None or is_fainted(target):
                        continue
                    score = self._counter_targeted_score(battle, active, move, target)
                    if score > best[0]:
                        best = (score, move, int(position))

            if best[1] is None:
                return _multi_safe_random_order(self, battle)
            return self.create_order(best[1], move_target=int(best[2] or 0))
        except Exception:
            return DefaultBattleOrder()

    @classmethod
    def _counter_targeted_score(
        cls, battle: MultiBattle, active: Any, move: Any, target: Any
    ) -> float:
        mult = damage_multiplier(target, move)
        if mult == 0:
            return -60.0
        ratio = _advanced_damage_ratio(battle, move, active, target)
        ko = _advanced_ko_probability(battle, move, active, target)
        target_threat = cls._target_pressure_into_our_side(battle, target)
        self_risk = cls._predicted_incoming_to_active(battle, active)
        goes_first = cls._moves_before(active, move, target)
        low_hp = max(0.0, 1.0 - safe_hp_fraction(target))
        score = (
            5.00 * ratio
            + 4.80 * ko
            + 1.25 * target_threat
            + 0.70 * min(4.0, mult)
            + 0.45 * float(has_type(active, move))
            + 0.35 * move_accuracy(move)
            + 0.38 * move_priority(move)
            + 0.85 * low_hp
        )
        if ko >= 0.55:
            score += 1.50 + 2.10 * float(goes_first) + 1.10 * target_threat
        elif ratio >= max(0.05, safe_hp_fraction(target) * 0.72):
            score += 0.85 + 0.55 * target_threat
        if self_risk >= 0.75 and not (goes_first and ko >= 0.55):
            score -= 0.85
        if move_accuracy(move) < 0.80 and ko < 0.75:
            score -= 0.65
        return score

    @classmethod
    def _counter_spread_score(cls, battle: MultiBattle, active: Any, move: Any) -> float:
        score = 0.0
        hits = 0
        for opp in active_alive_mons(battle.opponent_active_pokemon[:2]):
            mult = damage_multiplier(opp, move)
            if mult == 0:
                score -= 3.5
                continue
            ratio = _advanced_damage_ratio(battle, move, active, opp, spread=True)
            ko = _advanced_ko_probability(battle, move, active, opp, spread=True)
            score += (
                4.40 * ratio + 4.30 * ko + 0.75 * cls._target_pressure_into_our_side(battle, opp)
            )
            hits += 1
        partner_risk = estimate_partner_damage_risk(
            battle,
            JointAction(
                SlotAction(0, "move", DefaultBattleOrder(), move_id(move), move=move),
                SlotAction(1, "pass", PassBattleOrder(), "pass"),
            ),
        )
        return score + 0.55 * hits + 0.25 * move_accuracy(move) - 1.20 * partner_risk

    @classmethod
    def _counter_status_score(cls, battle: MultiBattle, active: Any, move: Any) -> float:
        mid = move_id(move)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        self_risk = cls._predicted_incoming_to_active(battle, active)
        if mid in PROTECT_MOVES:
            if self_risk >= 0.85:
                return 4.20 + 1.40 * safe_hp_fraction(active)
            if self_risk >= 0.55 and turn >= 2:
                return 2.10
            return -0.75 if turn <= 1 else -0.35
        if mid in FAKE_OUT_MOVES and turn <= 2:
            best_threat = max(
                (
                    cls._target_pressure_into_our_side(battle, opp)
                    for opp in active_alive_mons(battle.opponent_active_pokemon[:2])
                ),
                default=0.0,
            )
            return 2.40 + 0.90 * best_threat
        if mid in SPEED_CONTROL_MOVES:
            ally_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.active_pokemon[:2])] or [100]
            )
            opp_speed = np.mean(
                [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon[:2])]
                or [100]
            )
            return 1.45 if ally_speed + 20 < opp_speed else 0.15
        if mid in HAZARD_MOVES:
            return -1.00
        if mid in SETUP_MOVES:
            return -0.40 if turn <= 1 and self_risk < 0.35 else -1.20
        return -0.45

    @staticmethod
    def _moves_before(active: Any, move: Any, target: Any) -> bool:
        my_priority = float(move_priority(move))
        opp_best_priority = max(
            (
                float(move_priority(opp_move))
                for opp_move in OpponentThreatModel._predicted_moves_for_opp(target)
                if move_base_power(opp_move) > 0
            ),
            default=0.0,
        )
        if my_priority != opp_best_priority:
            return my_priority > opp_best_priority
        return safe_speed(active) >= safe_speed(target)

    @staticmethod
    def _target_pressure_into_our_side(battle: MultiBattle, target: Any) -> float:
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

    @staticmethod
    def _predicted_incoming_to_active(battle: MultiBattle, active: Any) -> float:
        incoming = 0.0
        for opp in active_alive_mons(battle.opponent_active_pokemon[:2]):
            for move in OpponentThreatModel._predicted_moves_for_opp(opp):
                if move_base_power(move) <= 0:
                    continue
                incoming = max(
                    incoming,
                    _advanced_damage_ratio(battle, move, opp, active, spread=is_spread_move(move)),
                )
        hp = max(0.05, safe_hp_fraction(active))
        return utility_damage_ratio(incoming / hp, cap=1.35)


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
