from __future__ import annotations

from .multi_agent_context import *


class MultiAgentSummaryMixin:
    def _candidate_summaries(
        self, battle: MultiBattle, scored: List[Tuple[float, SlotAction]]
    ) -> List[Dict[str, Any]]:
        return [self._candidate_summary(battle, score, action) for score, action in scored]

    def _scaled_self_damage_ratio(self, ratio: Any) -> float:
        scale = max(0.05, float(getattr(self.config, "self_damage_estimate_scale", 1.0) or 1.0))
        return utility_damage_ratio(float(ratio or 0.0) * scale, cap=1.35)

    def _scaled_opponent_damage_ratio(self, ratio: Any) -> float:
        scale = max(0.05, float(getattr(self.config, "opponent_damage_estimate_scale", 1.0) or 1.0))
        return utility_damage_ratio(float(ratio or 0.0) * scale, cap=1.35)

    @staticmethod
    def _ko_probability_from_scaled_ratio(ratio: float, target: Any) -> float:
        hp = max(0.05, safe_hp_fraction(target) if target is not None else 1.0)
        return float(_ko_prob_from_effective(float(ratio or 0.0) / hp))

    def _candidate_summary(
        self, battle: MultiBattle, score: float, action: SlotAction
    ) -> Dict[str, Any]:
        attacker = battle.active_pokemon[0] if battle.active_pokemon else None
        mid = move_id(action.move) if action.move is not None else ""
        bp = float(move_base_power(action.move)) if action.move is not None else 0.0
        target_slot = self._intent_target_slot(battle, action)
        summary: Dict[str, Any] = {
            "signature": self._order_signature(action.order),
            "label": action.label,
            "score": float(score),
            "slot": int(action.slot),
            "kind": action.kind,
            "move": action.move,
            "move_id": mid,
            "bp": bp,
            "accuracy": float(move_accuracy(action.move)) if action.move is not None else 1.0,
            "priority": float(move_priority(action.move)) if action.move is not None else 0.0,
            "attacker_speed": float(safe_speed(attacker)) if attacker is not None else 0.0,
            "target": self._intent_target_species(battle, action),
            "target_slot": target_slot,
            "target_hp": self._target_hp_for_slot(battle, target_slot),
            "spread": bool(action.move is not None and is_spread_move(action.move)),
            "support": bool(action.move is not None and bp <= 0),
            "protect": bool(mid in PROTECT_MOVES),
            "recoil": bool(mid in RECOIL_MOVES),
            "self_sacrifice": bool(mid in SELF_SACRIFICE_MOVES),
            "tera": bool(action_uses_tera(action)),
            "speed_control": bool(mid in SPEED_CONTROL_MOVES),
            "status_control": bool(mid in STATUS_CONTROL_MOVES),
            "sleep_control": bool(mid in SLEEP_CONTROL_MOVES),
            "redirection": bool(mid in REDIRECTION_MOVES),
            "setup": bool(mid in SETUP_MOVES),
            "pivot": bool(mid in PIVOT_MOVES),
            "terrain": bool(mid in TERRAIN_MOVES),
            "screen": bool(mid in SCREEN_MOVES),
            "ally_target": bool(
                action.target is not None
                and action.target in active_alive_mons(battle.active_pokemon)
            ),
            "ally_activation": bool(
                action.target is not None and _move_hits_ally_activation(action.move, action.target)
            ),
            "ally_damage": 0.0,
            "field_effect": self._field_effect_for_move(action.move),
            "damage": {},
            "damage_by_slot": {},
            "ko_by_slot": {},
            "damage_sum": 0.0,
            "ko_sum": 0.0,
        }
        if action.kind != "move" or action.move is None or bp <= 0:
            return summary

        if attacker is None:
            return summary
        if action.target is not None and action.target in active_alive_mons(battle.active_pokemon):
            summary["ally_damage"] = float(
                _advanced_damage_ratio(battle, action.move, attacker, action.target)
            )
            return summary
        if is_spread_move(action.move):
            visible_opponents = active_alive_mons(battle.opponent_active_pokemon[:2])
            if not visible_opponents:
                ratio = self._scaled_self_damage_ratio(
                    blind_positional_damage_ratio(action.move, attacker, spread=True)
                )
                for opp_slot in (0, 1):
                    summary["damage"][f"opp-slot-{opp_slot}"] = ratio
                    summary["damage_by_slot"][opp_slot] = ratio
                    summary["ko_by_slot"][opp_slot] = float(blind_ko_probability_from_ratio(ratio))
            for opp_slot, opp in enumerate(battle.opponent_active_pokemon[:2]):
                if opp is None or is_fainted(opp):
                    continue
                ratio = self._scaled_self_damage_ratio(
                    _advanced_damage_ratio(battle, action.move, attacker, opp, spread=True)
                )
                summary["damage"][safe_species(opp)] = float(ratio)
                summary["damage_by_slot"][opp_slot] = ratio
                summary["ko_by_slot"][opp_slot] = self._ko_probability_from_scaled_ratio(ratio, opp)
        elif action.target is not None:
            ratio = self._scaled_self_damage_ratio(
                _advanced_damage_ratio(battle, action.move, attacker, action.target)
            )
            summary["damage"][safe_species(action.target)] = float(ratio)
            target_slot = self._opponent_slot_index(battle, action.target)
            if target_slot is not None:
                summary["damage_by_slot"][target_slot] = ratio
                summary["ko_by_slot"][target_slot] = self._ko_probability_from_scaled_ratio(
                    ratio, action.target
                )
        elif target_slot is not None and action.target is None and not is_spread_move(action.move):
            ratio = self._scaled_self_damage_ratio(
                blind_positional_damage_ratio(action.move, attacker, spread=False)
            )
            summary["damage"][f"opp-slot-{target_slot}"] = ratio
            summary["damage_by_slot"][target_slot] = ratio
            summary["ko_by_slot"][target_slot] = float(blind_ko_probability_from_ratio(ratio))
        summary["damage_sum"] = float(sum(float(v) for v in summary["damage_by_slot"].values()))
        summary["ko_sum"] = float(sum(float(v) for v in summary["ko_by_slot"].values()))
        return summary

    @staticmethod
    def _intent_target_species(battle: MultiBattle, action: SlotAction) -> Optional[str]:
        if action.target is not None:
            return safe_species(action.target)
        blind_slot = MultiAgentSummaryMixin._blind_target_slot_from_position(battle, action)
        if blind_slot is not None:
            return f"opp-slot-{blind_slot}"
        if action.move is not None and is_spread_move(action.move):
            focus = None
            best = -1.0
            attacker = battle.active_pokemon[0] if battle.active_pokemon else None
            for opp in active_alive_mons(battle.opponent_active_pokemon):
                ratio = _advanced_damage_ratio(battle, action.move, attacker, opp, spread=True)
                if ratio > best:
                    best = ratio
                    focus = safe_species(opp)
            return focus
        return None

    @staticmethod
    def _opponent_slot_index(battle: MultiBattle, target: Any) -> Optional[int]:
        for idx, opp in enumerate(battle.opponent_active_pokemon[:2]):
            if opp is target:
                return idx
        for idx, opp in enumerate(battle.opponent_active_pokemon[:2]):
            if safe_species(opp) == safe_species(target):
                return idx
        return None

    @staticmethod
    def _blind_target_slot_from_position(battle: MultiBattle, action: SlotAction) -> Optional[int]:
        if action.target is not None or action.move is None or move_base_power(action.move) <= 0:
            return None
        try:
            position = int(action.target_position or 0)
        except Exception:
            return None
        if position == int(getattr(battle, "OPPONENT_1_POSITION", 1)):
            return 0
        if position == int(getattr(battle, "OPPONENT_2_POSITION", 2)):
            return 1
        return None

    @classmethod
    def _intent_target_slot(cls, battle: MultiBattle, action: SlotAction) -> Optional[int]:
        if action.target is not None:
            return cls._opponent_slot_index(battle, action.target)
        blind_slot = cls._blind_target_slot_from_position(battle, action)
        if blind_slot is not None:
            return blind_slot
        if action.move is not None and is_spread_move(action.move):
            focus_slot = None
            best = -1.0
            attacker = battle.active_pokemon[0] if battle.active_pokemon else None
            for idx, opp in enumerate(battle.opponent_active_pokemon[:2]):
                if opp is None or is_fainted(opp):
                    continue
                ratio = _advanced_damage_ratio(battle, action.move, attacker, opp, spread=True)
                if ratio > best:
                    best = ratio
                    focus_slot = idx
            return focus_slot
        return None

    @staticmethod
    def _cleanup_intent_blackboard() -> None:
        now = time.time()
        stale_keys = []
        for key, bucket in MULTI_INTENT_BLACKBOARD.items():
            stale_agents = [
                agent_name
                for agent_name, intent in bucket.items()
                if now - float(intent.get("updated_at", 0.0)) > 30.0
            ]
            for agent_name in stale_agents:
                bucket.pop(agent_name, None)
            if not bucket:
                stale_keys.append(key)
        for key in stale_keys:
            MULTI_INTENT_BLACKBOARD.pop(key, None)
        stale_mem = [
            key
            for key, mem in MULTI_SHORT_MEMORY.items()
            if now - float(mem.get("updated_at", 0.0)) > 1800.0
        ]
        for key in stale_mem:
            MULTI_SHORT_MEMORY.pop(key, None)

    def _safe_forced_switch_single_order(self, battle: MultiBattle, repeat_count: int = 1):
        try:
            switch_actions = [
                a
                for a in self.action_generator.generate_slot_actions(battle, 0)
                if a.kind == "switch" and a.switch is not None
            ]

            if not switch_actions:
                switches = self._raw_switch_candidates(battle)
                switch_actions = [
                    SlotAction(
                        0,
                        "switch",
                        self.create_order(mon),
                        f"switch->{safe_species(mon)}",
                        switch=mon,
                    )
                    for mon in switches
                ]

            if not switch_actions:
                return DefaultBattleOrder()

            active_species = {safe_species(m) for m in battle.active_pokemon if m is not None}
            filtered = [
                a for a in switch_actions if safe_species(a.switch) not in active_species
            ] or switch_actions
            filtered.sort(
                key=lambda a: self._switch_in_pressure_score(a.switch, battle), reverse=True
            )
            return filtered[(max(1, repeat_count) - 1) % len(filtered)].order
        except Exception:
            return DefaultBattleOrder()

    def _raw_switch_candidates(self, battle: MultiBattle) -> List[Any]:
        raw = self._raw_request(battle)
        side = raw.get("side", {}) if isinstance(raw.get("side", {}), dict) else {}
        result = []
        active_species = {safe_species(m) for m in battle.active_pokemon if m is not None}
        for pokemon in side.get("pokemon", []) or []:
            if pokemon.get("active"):
                continue
            ident = pokemon.get("ident")
            condition = str(pokemon.get("condition", ""))
            if not ident or condition.startswith("0 fnt") or condition == "0":
                continue
            mon = (safe_getattr(battle, "team", {}) or {}).get(ident)
            if mon is None:
                mon = safe_getattr(battle, "_team", {}).get(ident)
            if mon is not None and safe_species(mon) not in active_species and not is_fainted(mon):
                result.append(mon)
        return result

    def _anti_loop_single_order(self, battle: MultiBattle, repeat_count: int):
        try:
            if self._single_slot_must_switch(battle):
                return self._safe_forced_switch_single_order(battle, repeat_count=repeat_count)

            actions = self.action_generator.generate_slot_actions(battle, 0)
            move_actions = [a for a in actions if a.kind == "move" and a.move is not None]
            move_actions.sort(key=lambda a: self.slot_agent.score(battle, a), reverse=True)
            orders = self._dedupe_orders([a.order for a in move_actions])

            try:
                orders.extend(battle.valid_orders[0])
            except Exception:
                pass
            orders = self._dedupe_orders(
                [o for o in orders if not isinstance(o, (PassBattleOrder, DefaultBattleOrder))]
            )
            if not orders:
                return DefaultBattleOrder()
            return orders[(repeat_count - self.config.max_repeated_turn_calls - 1) % len(orders)]
        except Exception:
            return DefaultBattleOrder()

    def _estimate_current_state_value(self, battle: MultiBattle) -> float:
        actions = self.action_generator.generate_slot_actions(battle, 0)
        top = self.slot_agent.top_k(battle, actions, min(6, self.config.top_k_slot_actions))
        values = []
        for _, action in top:
            joint = JointAction(
                action,
                SlotAction(1, "pass", PassBattleOrder(), "partner-controlled-by-other-player"),
            )
            values.append(self.model.predict(self.encoder.encode(battle, joint)))
        return max(values) if values else -0.5

    def _hp_balance(self, battle: MultiBattle) -> float:
        my_active = active_alive_mons(battle.active_pokemon)
        opp_active = active_alive_mons(battle.opponent_active_pokemon)
        if my_active or opp_active:
            return (
                sum(safe_hp_fraction(m) for m in my_active)
                - sum(safe_hp_fraction(m) for m in opp_active)
            ) / 2.0
        return super()._hp_balance(battle)


__all__ = ["MultiAgentSummaryMixin"]
