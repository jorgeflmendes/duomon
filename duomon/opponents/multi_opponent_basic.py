from __future__ import annotations

from .multi_opponent_context import *
from .multi_opponent_utils import *


class RandomMultiSlotOpponent(MultiAwarePlayerMixin, Player):
    async def choose_move(self, battle: MultiBattle):
        return _multi_safe_random_order(self, battle)


class MaxPowerMultiSlotOpponent(MultiAwarePlayerMixin, Player):
    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                return random.choice(switches) if switches else DefaultBattleOrder()
            active = battle.active_pokemon[0]
            moves = normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
            if active is None or not moves:
                return _multi_safe_random_order(self, battle)
            best_score = -1e9
            best_moves: List[Any] = []
            for move in moves:
                bp = move_base_power(move)
                if bp <= 0:
                    continue
                score = bp + 5.0 * move_accuracy(move) + 2.0 * move_priority(move)
                if score > best_score + 1e-9:
                    best_score = score
                    best_moves = [move]
                elif abs(score - best_score) <= 1e-9:
                    best_moves.append(move)
            if not best_moves:
                return _multi_safe_random_order(self, battle)
            move = random.choice(best_moves)
            positions = _legal_showdown_positions(battle, move, active)
            opp_positions = [
                pos
                for pos in positions
                if pos in {battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION}
            ]
            return self.create_order(move, move_target=random.choice(opp_positions or positions))
        except Exception:
            return DefaultBattleOrder()


class TypeAwareMultiSlotOpponent(MultiAwarePlayerMixin, Player):
    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                return random.choice(switches) if switches else DefaultBattleOrder()
            active = battle.active_pokemon[0]
            if active is None:
                return _multi_safe_random_order(self, battle)

            results = []
            for opp_id in [0, 1]:
                try:
                    result = SimpleHeuristicsPlayer.choose_singles_move(
                        PseudoBattle(battle, 0, opp_id)
                    )
                    order, score = result
                    if (
                        order is not None
                        and hasattr(order, "order")
                        and isinstance(order.order, Move)
                    ):
                        preferred_target = [battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION][
                            opp_id
                        ]
                        possible_targets = battle.get_possible_showdown_targets(order.order, active)
                        order.move_target = (
                            preferred_target
                            if preferred_target in possible_targets
                            else possible_targets[0]
                        )
                    score *= SimpleHeuristicsPlayer.get_double_target_multiplier(battle, order)
                    results.append((order, score))
                except Exception:
                    continue
            if not results:
                return _multi_safe_random_order(self, battle)
            return max(results, key=lambda item: item[1])[0]
        except Exception:
            return DefaultBattleOrder()


class SimpleHeuristicsMultiSlotOpponent(TypeAwareMultiSlotOpponent):
    pass


class SimpleSmartSwitchMultiSlotAgent(TypeAwareMultiSlotOpponent):

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
            return await super().choose_move(battle)
        except Exception:
            return DefaultBattleOrder()


class SimplePlusMultiSlotAgent(TypeAwareMultiSlotOpponent):

    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                return random.choice(switches) if switches else DefaultBattleOrder()
            active = battle.active_pokemon[0] if battle.active_pokemon else None
            if active is None:
                return _multi_safe_random_order(self, battle)

            results = []
            for opp_id in [0, 1]:
                try:
                    order, score = SimpleHeuristicsPlayer.choose_singles_move(
                        PseudoBattle(battle, 0, opp_id)
                    )
                    if order is None:
                        continue
                    move = safe_getattr(order, "order", None)
                    target = (
                        battle.opponent_active_pokemon[opp_id]
                        if opp_id < len(battle.opponent_active_pokemon)
                        else None
                    )
                    if hasattr(order, "order") and isinstance(move, Move):
                        preferred_target = [battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION][
                            opp_id
                        ]
                        possible_targets = battle.get_possible_showdown_targets(move, active)
                        order.move_target = (
                            preferred_target
                            if preferred_target in possible_targets
                            else possible_targets[0]
                        )
                    score *= SimpleHeuristicsPlayer.get_double_target_multiplier(battle, order)
                    score += self._simple_plus_bonus(battle, active, move, target)
                    results.append((order, score))
                except Exception:
                    continue
            if not results:
                return _multi_safe_random_order(self, battle)
            return max(results, key=lambda item: item[1])[0]
        except Exception:
            return DefaultBattleOrder()

    @staticmethod
    def _simple_plus_bonus(battle: MultiBattle, active: Any, move: Any, target: Any) -> float:
        if move is None:
            return 0.0
        mid = move_id(move)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        bp = move_base_power(move)
        if bp <= 0:
            if mid in FAKE_OUT_MOVES and turn <= 2 and target is not None:
                return 2.00 + 0.80 * DamageRaceMultiSlotAgent._target_pressure(battle, target)
            if mid in PROTECT_MOVES:
                incoming = CounterAbyssalMultiSlotOpponent._predicted_incoming_to_active(
                    battle, active
                )
                return (
                    1.35 * incoming
                    if incoming >= 0.85 and safe_hp_fraction(active) < 0.65
                    else -0.30
                )
            if mid in SPEED_CONTROL_MOVES:
                ally_speed = np.mean(
                    [safe_speed(m) for m in active_alive_mons(battle.active_pokemon[:2])] or [100]
                )
                opp_speed = np.mean(
                    [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon[:2])]
                    or [100]
                )
                return 0.85 if ally_speed + 15 < opp_speed else -0.10
            if mid in HAZARD_MOVES:
                return -1.00
            return 0.0
        if is_spread_move(move):
            value = 0.0
            hits = 0
            for opp in active_alive_mons(battle.opponent_active_pokemon[:2]):
                ratio = _advanced_damage_ratio(battle, move, active, opp, spread=True)
                hp = max(0.05, safe_hp_fraction(opp))
                effective = ratio / hp
                value += 1.85 * min(1.7, effective) + 2.20 * _ko_prob_from_effective(effective)
                hits += 1
            partner_risk = 0.0
            for ally in active_alive_mons(battle.active_pokemon[:2]):
                if ally is not active and damage_multiplier(ally, move) > 0:
                    partner_risk += damage_multiplier(ally, move)
            return value + 0.45 * hits - 1.15 * partner_risk
        if target is None or is_fainted(target) or _target_is_ally(battle, target):
            return -4.0
        mult = damage_multiplier(target, move)
        if mult == 0:
            return -8.0
        ratio = _advanced_damage_ratio(battle, move, active, target)
        hp = max(0.05, safe_hp_fraction(target))
        effective = ratio / hp
        ko = _ko_prob_from_effective(effective)
        pressure = DamageRaceMultiSlotAgent._target_pressure(battle, target)
        value = (
            2.25 * min(1.8, effective)
            + 2.85 * ko
            + 0.75 * pressure
            + 0.25 * min(4.0, mult)
            + 0.22 * float(has_type(active, move))
            + 0.18 * move_priority(move)
        )
        if ratio > hp + 0.55:
            value -= 0.35 * min(1.5, ratio - hp)
        if mid in RECOIL_MOVES and safe_hp_fraction(active) < 0.50 and ko < 0.65:
            value -= 0.75
        return value


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
