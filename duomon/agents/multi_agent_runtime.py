from __future__ import annotations

from .multi_agent_context import *


class MultiAgentRuntimeMixin:
    def __init__(self, config: AgentConfig, agent_name: str = "multi-agent", **kwargs):
        self.agent_name = agent_name
        super().__init__(config=config, **kwargs)
        self.slot_agent = SlotAgent(0, config)
        self.ctde_joint_reranker = make_ctde_joint_reranker(config)
        self.transformer_action_prior = (
            get_transformer_action_prior(
                str(getattr(config, "transformer_action_prior_run_dir", "")),
                str(getattr(config, "transformer_action_prior_device", "cpu") or "cpu"),
            )
            if bool(getattr(config, "transformer_action_prior_enabled", False))
            and float(getattr(config, "transformer_action_prior_weight", 0.0) or 0.0) > 0.0
            else None
        )
        self._decision_gate_diagnostics: Dict[str, Dict[str, Any]] = {}
        self._decision_gate_raw_context: Dict[str, Dict[str, Any]] = {}

    async def choose_move(self, battle: MultiBattle):
        if battle.finished:
            self._handle_terminal_if_needed(battle)
            return DefaultBattleOrder()

        battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        rqid = self._single_slot_rqid(battle)
        call_key = f"{battle_key}:{self.agent_name}:{turn}:{rqid}"
        self.turn_call_counter[call_key] = self.turn_call_counter.get(call_key, 0) + 1
        repeat_count = self.turn_call_counter[call_key]
        cached_order = self.last_order_by_request.get(call_key)
        choice_was_rejected = safe_getattr(battle, "_last_choice_error_rqid", None) == rqid
        if repeat_count > 1 and cached_order is not None and not choice_was_rejected:
            return cached_order

        if self._single_slot_waiting(battle):
            if self._should_log_repeat(repeat_count):
                self._log_unique(battle, f"{self.agent_name} | wait request rqid={rqid} | no order")
            await asyncio.sleep(0.02)
            return DefaultBattleOrder()

        if self._single_slot_must_switch(battle):
            self._log_unique(battle, f"{self.agent_name} | forced-switch slot0 | single-slot")
            order = self._safe_forced_switch_single_order(battle, repeat_count=repeat_count)
            self.last_order_by_request[call_key] = order
            return order

        if repeat_count > self.config.max_repeated_turn_calls:
            if self._should_log_repeat(repeat_count):
                logger.info(
                    f"[decision] turn={turn} agent={self.agent_name} action=anti_loop repeated_requests={repeat_count} rqid={rqid}"
                )



            await asyncio.sleep(0.005 if repeat_count < 50 else 0.02)
            order = self._anti_loop_single_order(battle, repeat_count)
            self.last_order_by_request[call_key] = order
            return order

        start = time.time()
        try:
            self._online_update_from_current_state(battle)
            if turn >= self.config.max_battle_turns_soft:
                order = self._anti_loop_single_order(battle, repeat_count)
                self.last_order_by_request[call_key] = order
                return order

            chosen, features, value = await self._choose_single_slot_action(battle)
            self.model.remember(battle_key, features, value)
            self.replay_logger.log_turn(self.mode, battle, chosen, features, value)
            self._record_chosen_action(battle, chosen)

            if self.config.log_every_turn:
                elapsed = time.time() - start
                self._log_unique(
                    battle,
                    f"{self.agent_name} | multi-independent | {elapsed:.2f}s | value={value:.3f} | {chosen.left.label}",
                )

            self._remember_hp_balance(battle)
            self.last_order_by_request[call_key] = chosen.left.order
            return chosen.left.order
        except Exception as exc:
            logger.exception(
                f"[decision] agent={self.agent_name} action=fallback reason=multi_independent_exception error={exc}"
            )
            order = self._anti_loop_single_order(battle, repeat_count)
            self.last_order_by_request[call_key] = order
            return order

    @staticmethod
    def _raw_request(battle: MultiBattle) -> Dict[str, Any]:
        raw = safe_getattr(battle, "_last_raw_request", {}) or {}
        return raw if isinstance(raw, dict) else {}

    def _single_slot_rqid(self, battle: MultiBattle) -> Any:
        return self._raw_request(battle).get("rqid", "no-rqid")

    def _single_slot_waiting(self, battle: MultiBattle) -> bool:
        return bool(self._raw_request(battle).get("wait", False))

    def _single_slot_must_switch(self, battle: MultiBattle) -> bool:
        raw_force = self._raw_request(battle).get("forceSwitch", None)
        if isinstance(raw_force, list):
            return any(bool(x) for x in raw_force)
        if raw_force is not None:
            return bool(raw_force)
        return bool(force_switch_list(battle)[0])

    @staticmethod
    def _should_log_repeat(count: int) -> bool:
        return count <= 3 or count in {5, 10, 25, 50} or (count >= 100 and count % 100 == 0)

    def _communication_rng(self, battle: MultiBattle, salt: str = "") -> random.Random:
        seed = int(getattr(self.config, "seed", 42) or 42)
        key = (
            f"{seed}:{self.agent_name}:"
            f"{safe_getattr(battle, 'battle_tag', 'unknown')}:"
            f"{int(safe_getattr(battle, 'turn', 0) or 0)}:{salt}"
        )
        return random.Random(key)

    def _communication_mode(self) -> str:
        return comm_mode(self.config)

    def _communication_packet_diagnostics(
        self,
        mode: str,
        packets_before: List[Dict[str, Any]],
        packets_after: List[Dict[str, Any]],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        gates = [packet_gate_value(packet) for packet in packets_after]
        return {
            **intervention,
            "mode": mode,
            "gate_enabled": bool(getattr(self.config, "communication_use_gate", False)),
            "gate_threshold": float(
                getattr(self.config, "communication_gate_threshold", 0.0) or 0.0
            ),
            "gate_mean": float(sum(gates) / len(gates)) if gates else 0.0,
            "gate_penalty": 0.0,
            "raw_packets": len(packets_before),
            "effective_packets": len(packets_after),
        }

    @staticmethod
    def _order_signature(order: Any) -> str:
        return str(
            safe_getattr(order, "message", None)
            or safe_getattr(order, "order", None)
            or repr(order)
        )

    def _dedupe_orders(self, orders: List[SingleBattleOrder]) -> List[SingleBattleOrder]:
        seen = set()
        unique = []
        for order in orders:
            key = self._order_signature(order)
            if key in seen:
                continue
            seen.add(key)
            unique.append(order)
        return unique

    async def _choose_single_slot_action(
        self, battle: MultiBattle
    ) -> Tuple[JointAction, np.ndarray, float]:
        sync_wait = float(getattr(self.config, "early_state_sync_wait_seconds", 0.0) or 0.0)
        if (
            sync_wait > 0.0
            and int(safe_getattr(battle, "turn", 0) or 0) <= 1
            and not active_alive_mons(battle.opponent_active_pokemon[:2])
        ):
            await asyncio.sleep(min(0.50, sync_wait))
        actions = self.action_generator.generate_slot_actions(battle, 0)
        actions = self._progress_filtered_actions(battle, actions)
        partner_placeholder = SlotAction(
            slot=1,
            kind="pass",
            order=PassBattleOrder(),
            label="partner-controlled-by-other-player",
        )
        scored = [(self.slot_agent.score(battle, action), action) for action in actions]
        scored = [
            (score + self._blind_opening_target_bonus(battle, action), action)
            for score, action in scored
        ]
        scored = [
            (
                score
                + float(getattr(self.config, "hard_benchmark_protect_weight", 0.0) or 0.0)
                * self._hard_benchmark_protect_bonus(battle, action),
                action,
            )
            for score, action in scored
        ]
        scored = self._apply_decision_gates(battle, scored)
        scored.sort(key=lambda item: item[0], reverse=True)
        communication_mode = self._communication_mode()
        communication_active = not mode_disables_partner_messages(communication_mode)
        communication_type = str(
            getattr(self.config, "communication_type", "structured") or "structured"
        ).lower()
        intent_scored = scored
        if communication_active and communication_type == "transformer_prior":
            intent_scored = self._apply_transformer_action_prior(battle, scored, [])
            intent_scored.sort(key=lambda item: item[0], reverse=True)
        self._refresh_shared_memory_from_battle(battle, intent_scored)
        own_message = self._build_structured_comm_message(battle, intent_scored)
        if communication_mode == "zero_messages":
            own_message = zero_message(own_message)
        if communication_active:
            self._publish_single_slot_intent(battle, intent_scored, own_message)



        await asyncio.sleep(
            max(0.0, float(getattr(self.config, "partner_intent_wait_seconds", 0.45) or 0.0))
        )
        if communication_active:
            partner_packets = self._partner_packets_for_mode(battle, communication_mode)
        else:
            partner_packets = []
        raw_partner_packets = list(partner_packets)
        partner_packets, communication_diagnostics = self._intervene_partner_packets(
            battle, communication_mode, partner_packets
        )
        partner_candidates = self._partner_candidates_from_packets(partner_packets)
        partner_messages = [
            packet.get("message")
            for packet in partner_packets
            if isinstance(packet.get("message"), dict)
        ]
        self._merge_partner_messages_into_memory(battle, partner_messages)
        if communication_type == "transformer_prior":
            team_plan = None
        else:
            team_plan = self._arbitrate_structured_team_plan(battle, own_message, partner_messages)
        fused_partner_messages = partner_messages
        combo_prebonus_weight = float(
            getattr(self.config, "shared_joint_combo_prebonus_weight", 0.0) or 0.0
        )
        if partner_candidates and (
            not self.config.use_shared_joint_selector or combo_prebonus_weight > 0.0
        ):
            scored = [
                (
                    score
                    + combo_prebonus_weight
                    * self._joint_combo_bonus(battle, action, partner_candidates),
                    action,
                )
                for score, action in scored
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
        if fused_partner_messages and communication_type != "transformer_prior":
            scored = [
                (
                    score
                    + self._structured_communication_bonus(
                        battle, action, fused_partner_messages, own_message, team_plan
                    ),
                    action,
                )
                for score, action in scored
            ]
            scored.sort(key=lambda item: item[0], reverse=True)



        scored = [
            (score + self._role_based_focus_bonus(battle, action), action)
            for score, action in scored
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = self._apply_transformer_action_prior(battle, scored, fused_partner_messages)
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = self._apply_learned_value_bonus(battle, scored, partner_placeholder)
        scored.sort(key=lambda item: item[0], reverse=True)
        joint_selection_info: Optional[Dict[str, Any]] = None
        if bool(getattr(self.config, "use_joint_plan_barrier", False)):
            role = str(_multi_side_role(battle) or self.agent_name or "").lower()
            if "p3" in role or "p4" in role or "p3" in self.agent_name.lower():
                await asyncio.sleep(
                    max(
                        0.0,
                        float(getattr(self.config, "joint_plan_follow_wait_seconds", 0.08) or 0.0),
                    )
                )
        joint_selection = self._select_shared_joint_action(
            battle, scored, partner_candidates, partner_placeholder
        )
        if joint_selection is not None:
            selected_score, selected_action, joint_selection_info = joint_selection
        else:
            selected_score, selected_action = scored[0]
            accepted_contextual = False
            for score, action in scored:
                candidate = JointAction(action, partner_placeholder)
                candidate.search_score = float(score)
                if not self._contextual_action_is_bad(battle, candidate):
                    selected_score, selected_action = score, action
                    accepted_contextual = True
                    break
            if not accepted_contextual:
                damaging = self._fallback_progress_damaging_items(
                    battle, scored, partner_placeholder
                )
                if damaging:
                    selected_score, selected_action = damaging[0]
            else:
                damaging = self._fallback_progress_damaging_items(
                    battle, scored, partner_placeholder
                )
                if (
                    damaging
                    and selected_action.move is not None
                    and move_base_power(selected_action.move) <= 0
                ):
                    if damaging[0][0] >= selected_score - 2.50 or self._is_low_progress_move(
                        selected_action
                    ):
                        selected_score, selected_action = damaging[0]
        chosen = JointAction(selected_action, partner_placeholder)
        features = self.encoder.encode(battle, chosen)
        value = self.model.predict(features)
        chosen.value = value
        chosen.search_score = float(selected_score)
        final_message = self._build_structured_commitment_message(
            battle, selected_action, selected_score, own_message, fused_partner_messages, team_plan
        )
        if joint_selection_info:
            final_message.setdefault("content", {})["shared_joint_plan"] = json_safe(
                joint_selection_info
            )
        communication_diagnostics = self._communication_packet_diagnostics(
            communication_mode, raw_partner_packets, partner_packets, communication_diagnostics
        )
        final_message.setdefault("content", {})["communication_diagnostics"] = json_safe(
            communication_diagnostics
        )
        decision_diagnostics = self._final_decision_diagnostics(
            battle, selected_action, selected_score
        )
        if decision_diagnostics:
            decision_diagnostics["communication_diagnostics"] = json_safe(communication_diagnostics)
            final_message.setdefault("content", {})["decision_diagnostics"] = json_safe(
                decision_diagnostics
            )
            chosen.decision_diagnostics = json_safe(decision_diagnostics)
        if communication_active:
            self._publish_single_slot_commitment(battle, final_message)
            chosen.messages = [own_message] + partner_messages + [final_message]
        else:
            chosen.messages = []
        structured_gain = (
            self._structured_communication_bonus(
                battle, selected_action, fused_partner_messages, own_message, team_plan
            )
            if fused_partner_messages
            and communication_type != "transformer_prior"
            else 0.0
        )
        if joint_selection_info:
            structured_gain += float(joint_selection_info.get("pair_bonus", 0.0) or 0.0)
        chosen.communication_score = float(structured_gain)
        if not communication_active:
            chosen.protocol_used = "independent-local"
            chosen.protocol_reason = "communication-disabled"
        elif joint_selection_info:
            chosen.protocol_used = "shared-joint-selector"
            chosen.protocol_reason = str(joint_selection_info.get("reason", "shared-joint-plan"))
        else:
            chosen.protocol_used = (
                "structured-vgc-comm" if partner_messages else "structured-vgc-solo-inform"
            )
            chosen.protocol_reason = str(final_message.get("content", {}).get("decision", "commit"))
        return chosen, features, value

    def _fallback_progress_damaging_items(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
        partner_placeholder: SlotAction,
    ) -> List[Tuple[float, SlotAction]]:
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        all_damaging = [
            item
            for item in scored
            if item[1].kind == "move"
            and item[1].move is not None
            and move_base_power(item[1].move) > 0
        ]
        if self.config.benchmark_type != "vs_abyssal":
            return all_damaging
        damaging = [
            item
            for item in all_damaging
            if not (turn > 2 and move_id(item[1].move) in FIRST_TURN_ONLY_STYLE_MOVES)
        ]
        contextual = []
        for score, action in damaging:
            candidate = JointAction(action, partner_placeholder)
            candidate.search_score = float(score)
            if not self._contextual_action_is_bad(battle, candidate):
                contextual.append((score, action))
        return contextual or damaging


__all__ = ["MultiAgentRuntimeMixin"]
