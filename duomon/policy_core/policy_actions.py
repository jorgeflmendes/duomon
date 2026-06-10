from __future__ import annotations

from .policy_context import *


class LegalActionGenerator:
    def __init__(self, player: Player):
        self.player = player

    def generate_slot_actions(self, battle: DoubleBattle, slot: int) -> List[SlotAction]:
        forced = force_switch_list(battle)
        if forced[slot]:
            switches = self._available_switches(battle, slot)
            if switches:
                return self._keep_valid_orders(
                    battle, slot, [self._switch_action(slot, mon) for mon in switches]
                )
            return [SlotAction(slot, "default", DefaultBattleOrder(), "default-forced-switch")]

        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if active is None or is_fainted(active):
            return [SlotAction(slot, "pass", PassBattleOrder(), "pass")]

        actions: List[SlotAction] = []
        allow_pivots = bool(
            safe_getattr(safe_getattr(self.player, "config", None), "allow_pivot_moves", False)
        )
        for move in self._available_moves(battle, slot):
            if not allow_pivots and move_id(move) in PIVOT_MOVES:
                continue
            actions.extend(self._move_actions_for_targets(battle, slot, move))

        allow_switches = bool(
            safe_getattr(
                safe_getattr(self.player, "config", None), "allow_voluntary_switches", False
            )
        )
        if allow_switches and not trapped_list(battle)[slot]:
            for mon in self._available_switches(battle, slot):
                actions.append(self._switch_action(slot, mon))

        actions = self._keep_valid_orders(battle, slot, actions)
        if self._should_force_blind_offense_without_opponents(battle, slot):
            offensive_actions = self._blind_offensive_actions_without_opponents(battle, actions)
            fallback_actions = self._blind_attack_actions_for_unhydrated_opponents(
                battle, slot, allow_pivots
            )
            if fallback_actions:
                offensive_actions.extend(self._safe_generated_actions(battle, fallback_actions))
            if offensive_actions:
                deduped: List[SlotAction] = []
                seen_messages: set[str] = set()
                for action in offensive_actions:
                    message = self._order_message(action.order)
                    if message in seen_messages:
                        continue
                    seen_messages.add(message)
                    deduped.append(action)
                actions = deduped
        if self._needs_unhydrated_target_fallback(battle, actions):
            fallback_actions = self._blind_attack_actions_for_unhydrated_opponents(
                battle, slot, allow_pivots
            )
            if fallback_actions:
                actions = self._safe_generated_actions(battle, fallback_actions)
        return actions or [SlotAction(slot, "default", DefaultBattleOrder(), "default")]

    @staticmethod
    def _order_message(order: Any) -> str:
        return str(
            safe_getattr(order, "message", None)
            or safe_getattr(order, "order", None)
            or repr(order)
        )

    def _valid_slot_orders(self, battle: DoubleBattle, slot: int) -> List[SingleBattleOrder]:
        try:
            orders_by_slot = safe_getattr(battle, "valid_orders", []) or []
            if slot < len(orders_by_slot):
                return [
                    o
                    for o in orders_by_slot[slot]
                    if not isinstance(o, (PassBattleOrder, DefaultBattleOrder))
                ]
        except Exception:
            pass
        return []

    def _action_from_order(
        self, battle: DoubleBattle, slot: int, order: SingleBattleOrder
    ) -> SlotAction:
        raw_order = safe_getattr(order, "order", None)
        if isinstance(order, PassBattleOrder):
            return SlotAction(slot, "pass", order, "pass")
        if isinstance(raw_order, str):
            return SlotAction(slot, "default", order, "default")
        available_switches = self._available_switches(battle, slot)
        if raw_order in available_switches or (
            safe_getattr(raw_order, "species", None) and not isinstance(raw_order, Move)
        ):
            return SlotAction(
                slot, "switch", order, f"switch->{safe_species(raw_order)}", switch=raw_order
            )
        if isinstance(raw_order, Move) or move_id(raw_order):
            target_position = int(safe_getattr(order, "move_target", 0) or 0)
            target = self._target_from_position(battle, target_position)
            label = (
                move_name(raw_order)
                if target is None
                else f"{move_name(raw_order)}->{safe_species(target)}"
            )
            return SlotAction(
                slot,
                "move",
                order,
                label,
                move=raw_order,
                target=target,
                target_position=target_position,
            )
        return SlotAction(
            slot, "switch", order, f"switch->{safe_species(raw_order)}", switch=raw_order
        )

    def _keep_valid_orders(
        self, battle: DoubleBattle, slot: int, actions: List[SlotAction]
    ) -> List[SlotAction]:
        safe_actions = [
            action for action in actions if self._action_has_safe_target(battle, action)
        ]
        valid_orders = self._valid_slot_orders(battle, slot)
        if not valid_orders:
            return safe_actions or actions

        generated_by_message = {
            self._order_message(action.order): action for action in safe_actions
        }
        merged: List[SlotAction] = []
        forced = force_switch_list(battle)
        allow_switches = bool(
            safe_getattr(
                safe_getattr(self.player, "config", None), "allow_voluntary_switches", False
            )
        )
        for order in valid_orders:
            message = self._order_message(order)
            action = generated_by_message.get(message)
            if action is None:
                action = self._action_from_order(battle, slot, order)
            if (
                action.kind == "switch"
                and not (slot < len(forced) and forced[slot])
                and not allow_switches
            ):
                continue
            if self._action_has_safe_target(battle, action):
                merged.append(action)

        return (
            merged
            or safe_actions
            or actions
            or [SlotAction(slot, "default", DefaultBattleOrder(), "default-no-safe-target")]
        )

    def _safe_generated_actions(
        self, battle: DoubleBattle, actions: List[SlotAction]
    ) -> List[SlotAction]:
        return [
            action for action in actions if self._action_has_safe_target(battle, action)
        ] or actions

    def _available_moves(self, battle: DoubleBattle, slot: int) -> List[Any]:
        return normalize_slot_list(safe_getattr(battle, "available_moves", []), slot)

    def _blind_candidate_moves_for_unhydrated_opponents(
        self, battle: DoubleBattle, slot: int
    ) -> List[Any]:
        moves = list(self._available_moves(battle, slot))
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        if turn <= 1 and not active_alive_mons(battle.opponent_active_pokemon[:2]):
            active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
            known_moves = safe_getattr(active, "moves", {}) if active is not None else {}
            if isinstance(known_moves, dict):
                known_iter = list(known_moves.values())
            elif isinstance(known_moves, (list, tuple, set)):
                known_iter = list(known_moves)
            else:
                known_iter = []
            seen = {move_id(move) for move in moves}
            for move in known_iter:
                mid = move_id(move)
                if not mid or mid in seen or move_base_power(move) <= 0:
                    continue
                seen.add(mid)
                moves.append(move)
        return moves

    def _available_switches(self, battle: DoubleBattle, slot: int) -> List[Any]:
        return normalize_slot_list(safe_getattr(battle, "available_switches", []), slot)

    def _needs_unhydrated_target_fallback(
        self, battle: DoubleBattle, actions: List[SlotAction]
    ) -> bool:
        if active_alive_mons(battle.opponent_active_pokemon[:2]):
            return False
        if not actions:
            return True
        return all(action.kind in {"default", "pass"} for action in actions)

    def _should_force_blind_offense_without_opponents(
        self, battle: DoubleBattle, slot: int
    ) -> bool:
        if active_alive_mons(battle.opponent_active_pokemon[:2]):
            return False
        return any(
            move_base_power(move) > 0
            for move in self._blind_candidate_moves_for_unhydrated_opponents(battle, slot)
        )

    @staticmethod
    def _blind_offensive_actions_without_opponents(
        battle: DoubleBattle,
        actions: List[SlotAction],
    ) -> List[SlotAction]:
        opponent_positions = {
            int(getattr(battle, "OPPONENT_1_POSITION", 1)),
            int(getattr(battle, "OPPONENT_2_POSITION", 2)),
        }
        offensive: List[SlotAction] = []
        for action in actions:
            if action.kind != "move" or action.move is None or move_base_power(action.move) <= 0:
                continue
            if is_spread_move(action.move) or move_target_type(action.move) in NO_EXPLICIT_TARGETS:
                offensive.append(action)
                continue
            if int(action.target_position or 0) in opponent_positions:
                offensive.append(action)
        return offensive

    def _blind_attack_actions_for_unhydrated_opponents(
        self,
        battle: DoubleBattle,
        slot: int,
        allow_pivots: bool,
    ) -> List[SlotAction]:
        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if active is None:
            return []
        positions = [
            int(getattr(battle, "OPPONENT_1_POSITION", 1)),
            int(getattr(battle, "OPPONENT_2_POSITION", 2)),
        ]
        actions: List[SlotAction] = []
        for move in self._blind_candidate_moves_for_unhydrated_opponents(battle, slot):
            mid = move_id(move)
            if not allow_pivots and mid in PIVOT_MOVES:
                continue
            if (
                move_base_power(move) <= 0
                or mid in PROTECT_MOVES
                or is_spread_move(move)
                or move_target_type(move) in NO_EXPLICIT_TARGETS
            ):
                continue
            for position in positions:
                order = self.player.create_order(move, move_target=int(position))
                actions.append(
                    SlotAction(
                        slot,
                        "move",
                        order,
                        f"{mid}->opp{1 if position == positions[0] else 2}",
                        move=move,
                        target=None,
                        target_position=position,
                    )
                )
        return actions

    def _switch_action(self, slot: int, mon: Any) -> SlotAction:
        return SlotAction(
            slot,
            "switch",
            self.player.create_order(mon),
            f"switch->{safe_species(mon)}",
            switch=mon,
        )

    @staticmethod
    def _target_from_position(battle: DoubleBattle, position: int) -> Optional[Any]:
        if position == 0:
            return None
        if position == getattr(battle, "POKEMON_1_POSITION", -1):
            return battle.active_pokemon[0] if len(battle.active_pokemon) > 0 else None
        if position == getattr(battle, "POKEMON_2_POSITION", -2):
            return battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
        if position == getattr(battle, "OPPONENT_1_POSITION", 1):
            return (
                battle.opponent_active_pokemon[0]
                if len(battle.opponent_active_pokemon) > 0
                else None
            )
        if position == getattr(battle, "OPPONENT_2_POSITION", 2):
            return (
                battle.opponent_active_pokemon[1]
                if len(battle.opponent_active_pokemon) > 1
                else None
            )
        return None

    @staticmethod
    def _action_has_safe_target(battle: DoubleBattle, action: SlotAction) -> bool:
        if action.kind != "move" or action.move is None:
            return True
        mid = move_id(action.move)
        if action.target is None:
            position = int(action.target_position or 0)
            own_positions = {
                int(getattr(battle, "POKEMON_1_POSITION", -1)),
                int(getattr(battle, "POKEMON_2_POSITION", -2)),
            }
            opponent_positions = {
                int(getattr(battle, "OPPONENT_1_POSITION", 1)),
                int(getattr(battle, "OPPONENT_2_POSITION", 2)),
            }
            if position in own_positions:
                target = LegalActionGenerator._target_from_position(battle, position)
                return mid in BENEFICIAL_ALLY_TARGET_MOVES or _move_hits_ally_activation(
                    action.move, target
                )
            if position in opponent_positions:
                return LegalActionGenerator._blind_enemy_position_allowed(
                    battle, action.move, position
                )
            return True
        if action.target not in active_alive_mons(battle.active_pokemon):
            return True
        if move_base_power(action.move) > 0:
            return _move_hits_ally_activation(action.move, action.target)
        return mid in BENEFICIAL_ALLY_TARGET_MOVES

    @staticmethod
    def _blind_enemy_position_allowed(battle: DoubleBattle, move: Any, position: int) -> bool:
        if move is None or move_base_power(move) <= 0:
            return False
        if is_spread_move(move) or move_target_type(move) in NO_EXPLICIT_TARGETS:
            return True
        opponent_positions = [
            int(getattr(battle, "OPPONENT_1_POSITION", 1)),
            int(getattr(battle, "OPPONENT_2_POSITION", 2)),
        ]
        if position not in opponent_positions:
            return False
        slot = opponent_positions.index(position)
        opponents = list(battle.opponent_active_pokemon[:2])


        return all(mon is None for mon in opponents) or (
            slot < len(opponents) and opponents[slot] is None
        )

    def _move_actions_for_targets(
        self, battle: DoubleBattle, slot: int, move: Any
    ) -> List[SlotAction]:
        mid = move_id(move)
        target_type = move_target_type(move)
        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if active is None:
            return []

        try:
            target_positions = list(battle.get_possible_showdown_targets(move, active))
        except Exception:
            target_positions = (
                [0]
                if target_type in NO_EXPLICIT_TARGETS
                or mid in PROTECT_MOVES
                or is_spread_move(move)
                else []
            )

        opponent_positions = {
            int(getattr(battle, "OPPONENT_1_POSITION", 1)),
            int(getattr(battle, "OPPONENT_2_POSITION", 2)),
        }
        if (
            (
                target_positions == [0]
                or not any(int(pos) in opponent_positions for pos in target_positions)
            )
            and move_base_power(move) > 0
            and target_type not in NO_EXPLICIT_TARGETS
            and mid not in PROTECT_MOVES
            and not is_spread_move(move)
            and not active_alive_mons(battle.opponent_active_pokemon[:2])
        ):
            target_positions = [
                int(getattr(battle, "OPPONENT_1_POSITION", 1)),
                int(getattr(battle, "OPPONENT_2_POSITION", 2)),
            ]

        if not target_positions:
            return []

        actions: List[SlotAction] = []
        for position in target_positions:
            target = self._target_from_position(battle, int(position))




            if target is not None and target is active and move_base_power(move) > 0:
                continue
            if (
                target is not None
                and target in active_alive_mons(battle.active_pokemon)
                and move_base_power(move) > 0
                and not _move_hits_ally_activation(move, target)
            ):
                continue
            if (
                target is not None
                and target in active_alive_mons(battle.active_pokemon)
                and move_base_power(move) <= 0
                and mid not in BENEFICIAL_ALLY_TARGET_MOVES
            ):
                continue

            order = self.player.create_order(move, move_target=int(position))
            if target is not None:
                label = f"{move_name(move)}->{safe_species(target)}"
            elif int(position) in opponent_positions:
                label = f"{move_name(move)}->opp{int(position)}"
            else:
                label = move_name(move)
            actions.append(
                SlotAction(
                    slot,
                    "move",
                    order,
                    label,
                    move=move,
                    target=target,
                    target_position=int(position),
                )
            )

        return actions


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
