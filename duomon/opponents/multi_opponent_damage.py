from __future__ import annotations

from .multi_opponent_context import *
from .multi_opponent_utils import *
from .multi_opponent_basic import *


class SimpleDamageMultiSlotAgent(MultiAwarePlayerMixin, Player):

    async def choose_move(self, battle: MultiBattle):
        try:
            if _multi_request_must_switch(battle):
                switches = _multi_switch_orders(self, battle)
                if not switches:
                    return DefaultBattleOrder()
                return max(
                    switches,
                    key=lambda order: self._simple_switch_score(
                        safe_getattr(order, "order", None), battle
                    ),
                )

            active = battle.active_pokemon[0] if battle.active_pokemon else None
            moves = normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
            if active is None or is_fainted(active) or not moves:
                return _multi_safe_random_order(self, battle)

            best: Tuple[float, Optional[Any], int, bool] = (-1e9, None, 0, False)
            for move in moves:
                positions = _legal_showdown_positions(battle, move, active)
                if is_spread_move(move) or move_target_type(move) in NO_EXPLICIT_TARGETS:
                    score = self._simple_damage_score(battle, active, move, None)
                    if score > best[0]:
                        best = (score, move, 0, self._should_tera_simple(battle, move, 0))
                    continue
                for position in positions:
                    score = self._simple_damage_score(battle, active, move, int(position))
                    if score > best[0]:
                        target_slot = (
                            0
                            if int(position) == int(getattr(battle, "OPPONENT_1_POSITION", 1))
                            else 1
                        )
                        best = (
                            score,
                            move,
                            int(position),
                            self._should_tera_simple(battle, move, target_slot),
                        )

            if best[1] is None:
                return _multi_safe_random_order(self, battle)
            return self.create_order(
                best[1], move_target=int(best[2] or 0), terastallize=bool(best[3])
            )
        except Exception:
            return DefaultBattleOrder()

    @classmethod
    def _simple_damage_score(
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
            if mid in FAKE_OUT_MOVES and turn <= 2:
                target = LegalActionGenerator._target_from_position(battle, int(position or 0))
                return 85.0 + 25.0 * DamageRaceMultiSlotAgent._target_pressure(battle, target)
            if mid in PROTECT_MOVES:
                incoming = CounterAbyssalMultiSlotOpponent._predicted_incoming_to_active(
                    battle, active
                )
                return (
                    65.0 * incoming
                    if incoming >= 0.95 and safe_hp_fraction(active) < 0.65
                    else -20.0
                )
            if mid in SPEED_CONTROL_MOVES:
                ally_speed = np.mean(
                    [safe_speed(m) for m in active_alive_mons(battle.active_pokemon[:2])] or [100]
                )
                opp_speed = np.mean(
                    [safe_speed(m) for m in active_alive_mons(battle.opponent_active_pokemon[:2])]
                    or [100]
                )
                return 45.0 if ally_speed + 20 < opp_speed else -8.0
            if mid in HAZARD_MOVES:
                return -30.0
            if mid in SETUP_MOVES and turn <= 1 and safe_hp_fraction(active) > 0.90:
                return 35.0
            return -12.0

        visible_opponents = active_alive_mons(battle.opponent_active_pokemon[:2])
        if not visible_opponents:
            base = bp * move_accuracy(move) * move_expected_hits(move)
            if has_type(active, move):
                base *= 1.5
            if is_spread_move(move):
                return 1.45 * base
            opp1_pos = int(getattr(battle, "OPPONENT_1_POSITION", 1))
            return base + (45.0 if int(position or 0) == opp1_pos else -5.0)

        if is_spread_move(move):
            score = 0.0
            for opp in active_alive_mons(battle.opponent_active_pokemon[:2]):
                score += cls._simple_damage_formula(active, opp, move, battle, spread=True)
            partner_risk = 0.0
            for ally in active_alive_mons(battle.active_pokemon[:2]):
                if ally is not active and damage_multiplier(ally, move) > 0:
                    partner_risk += 45.0 * damage_multiplier(ally, move)
            return 0.78 * score - partner_risk

        target = LegalActionGenerator._target_from_position(battle, int(position or 0))
        if target is None or is_fainted(target) or _target_is_ally(battle, target):
            return -80.0
        return cls._simple_damage_formula(active, target, move, battle, spread=False)

    @staticmethod
    def _simple_damage_formula(
        active: Any, target: Any, move: Any, battle: MultiBattle, spread: bool
    ) -> float:
        try:
            physical_ratio = SimpleHeuristicsPlayer._stat_estimation(active, "atk") / max(
                1.0,
                SimpleHeuristicsPlayer._stat_estimation(target, "def"),
            )
            special_ratio = SimpleHeuristicsPlayer._stat_estimation(active, "spa") / max(
                1.0,
                SimpleHeuristicsPlayer._stat_estimation(target, "spd"),
            )
        except Exception:
            physical_ratio = float(safe_stat(active, "atk", 100)) / max(
                1.0, float(safe_stat(target, "def", 100))
            )
            special_ratio = float(safe_stat(active, "spa", 100)) / max(
                1.0, float(safe_stat(target, "spd", 100))
            )
        category = move_category(move)
        stat_ratio = physical_ratio if category == "physical" else special_ratio
        mult = damage_multiplier(target, move)
        if mult == 0:
            return -120.0
        base = (
            move_base_power(move)
            * (1.5 if has_type(active, move) else 1.0)
            * stat_ratio
            * move_accuracy(move)
            * move_expected_hits(move)
            * mult
        )
        ratio = _advanced_damage_ratio(battle, move, active, target, spread=spread)
        hp = max(0.05, safe_hp_fraction(target))
        effective = ratio / hp
        ko = _ko_prob_from_effective(effective)
        pressure = DamageRaceMultiSlotAgent._target_pressure(battle, target)
        low_hp = max(0.0, 1.0 - hp)
        value = base + 95.0 * ko + 38.0 * min(1.8, effective) + 22.0 * pressure + 18.0 * low_hp
        if ratio > hp + 0.55:
            value -= 18.0 * min(2.0, ratio - hp)
        if move_id(move) in RECOIL_MOVES and safe_hp_fraction(active) < 0.50 and ko < 0.75:
            value -= 45.0
        return value

    @staticmethod
    def _simple_switch_score(mon: Any, battle: MultiBattle) -> float:
        if mon is None:
            return -1e9
        return IndependentTwoSlotAgent._switch_in_pressure_score(mon, battle)

    @staticmethod
    def _should_tera_simple(battle: MultiBattle, move: Any, target_slot: int) -> bool:
        try:
            return bool(
                SimpleHeuristicsPlayer._should_terastallize(
                    PseudoBattle(battle, 0, target_slot), move
                )
            )
        except Exception:
            return False


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
