from __future__ import annotations

from .multi_agent_context import *


class MultiAgentCommRuntimeMixin:
    def _structured_communication_bonus(
        self,
        battle: MultiBattle,
        action: SlotAction,
        partner_messages: List[Dict[str, Any]],
        own_message: Dict[str, Any],
        team_plan: Optional[Dict[str, Any]] = None,
    ) -> float:
        if action.kind != "move" or action.move is None:
            return 0.0
        summary = self._public_candidate_summary(battle, 0.0, action)
        target_slot = summary.get("target_slot")
        damage_by_slot = {
            int(k): utility_damage_ratio(v)
            for k, v in (summary.get("damage_by_slot", {}) or {}).items()
        }
        bonus = 0.0
        mid = move_id(action.move)
        if team_plan:
            plan_strategy = str(team_plan.get("strategy", ""))
            plan_slot = team_plan.get("target_slot")
            try:
                plan_slot = int(plan_slot) if plan_slot is not None else None
            except Exception:
                plan_slot = None
            same_plan_target = plan_slot is not None and (
                target_slot == plan_slot or bool(summary.get("spread"))
            )
            if plan_strategy == "protect_partner_removes_threat" and plan_slot is not None:
                if str(team_plan.get("source_agent", "")) == self.agent_name:
                    if mid in PROTECT_MOVES:
                        danger = self._predicted_self_danger(battle)
                        bonus += 3.90 + 2.20 * min(1.0, danger["risk"] + danger["ko"])
                    elif move_base_power(action.move) > 0:
                        bonus -= 1.35
                elif same_plan_target and float(damage_by_slot.get(plan_slot, 0.0)) > 0.0:
                    bonus += 2.25 * float(team_plan.get("confidence", 0.7))
            elif (
                plan_strategy in {"finish_ko", "double_target_ko", "threat_removal"}
                and plan_slot is not None
            ):
                partner_damage = self._partner_best_damage_from_messages(partner_messages).get(
                    plan_slot, 0.0
                )
                my_damage = float(damage_by_slot.get(plan_slot, 0.0))
                solo_claim_handled = False
                if (
                    bool(getattr(self.config, "communication_solo_claims_enabled", False))
                    and team_plan.get("solo_claim")
                    and not team_plan.get("partner_required")
                ):
                    claim_penalty = float(
                        getattr(self.config, "communication_solo_claim_penalty", 1.0) or 1.0
                    )
                    if str(team_plan.get("source_agent", "")) != self.agent_name:
                        if same_plan_target and my_damage >= 0.15:
                            bonus -= claim_penalty * float(team_plan.get("confidence", 0.7))
                        elif not same_plan_target and move_base_power(action.move) > 0:
                            bonus += 0.18 * float(team_plan.get("confidence", 0.7))
                        solo_claim_handled = True
                    elif same_plan_target:
                        bonus += 0.35 * float(team_plan.get("confidence", 0.7))
                        solo_claim_handled = True
                if solo_claim_handled:
                    pass
                elif same_plan_target:
                    if (
                        team_plan.get("avoid_partner_overkill")
                        and partner_damage >= 0.95
                        and my_damage >= 0.25
                    ):
                        bonus -= 1.25
                    else:
                        bonus += 1.15 * float(team_plan.get("confidence", 0.7))
                    target_hp = float(team_plan.get("target_hp", 1.0) or 1.0)
                    if my_damage + partner_damage >= max(0.92, target_hp):
                        bonus += 0.70
                elif team_plan.get("partner_required"):
                    bonus -= 1.20
            elif plan_strategy == "spread_pressure" and bool(summary.get("spread")):
                bonus += 0.55
            elif plan_strategy == "speed_control_damage" and mid in SPEED_CONTROL_MOVES:
                bonus += 0.75
            elif plan_strategy == "terrain_control":
                effect = str(team_plan.get("field_effect") or "")
                boosted_type = str(team_plan.get("boosted_type") or "")
                if mid in TERRAIN_MOVES and (not effect or summary.get("field_effect") == effect):
                    bonus += 0.85 * float(team_plan.get("confidence", 0.7))
                elif (
                    boosted_type
                    and _move_type_name(action.move) == boosted_type
                    and move_base_power(action.move) > 0
                ):
                    bonus += 0.45 * float(team_plan.get("confidence", 0.7))
                elif (
                    effect == "psychicterrain"
                    and mid in FAKE_OUT_MOVES
                    and _psychic_terrain_blocks_priority(battle, action.target)
                ):
                    bonus -= 0.55
            elif plan_strategy == "pivot_cycle":
                if mid in PIVOT_MOVES:
                    bonus += 0.70 * float(team_plan.get("confidence", 0.7))
                elif (
                    bool(summary.get("spread"))
                    or utility_damage_sum(damage_by_slot.values()) >= 0.70
                ):
                    bonus += 0.18 * float(team_plan.get("confidence", 0.7))
            elif plan_strategy == "self_activation_combo":
                if bool(summary.get("ally_activation")):
                    bonus += 1.10 * float(team_plan.get("confidence", 0.7))
                    if float(summary.get("ally_damage", 1.0)) <= 0.35:
                        bonus += 0.35
                elif move_base_power(action.move) > 0 and not bool(summary.get("ally_target")):
                    bonus += 0.20 * float(team_plan.get("confidence", 0.7))
            elif plan_strategy == "setup_redirection":
                role = str(team_plan.get("coordination_role", ""))
                if str(team_plan.get("source_agent", "")) != self.agent_name:
                    role = str(team_plan.get("partner_role", role))
                if role == "protector":
                    if mid in REDIRECTION_MOVES or mid in FAKE_OUT_MOVES:
                        bonus += 1.15 * float(team_plan.get("confidence", 0.7))
                    elif mid in PROTECT_MOVES:
                        bonus += 0.25
                elif role == "setup_sweeper":
                    if mid in SETUP_MOVES:
                        bonus += 1.05 * float(team_plan.get("confidence", 0.7))
                    elif move_base_power(action.move) > 0:
                        bonus += 0.22 * float(team_plan.get("confidence", 0.7))
            elif plan_strategy == "screens_positioning":
                if mid in SCREEN_MOVES:
                    bonus += 0.90 * float(team_plan.get("confidence", 0.7))
                elif mid in SETUP_MOVES or utility_damage_sum(damage_by_slot.values()) >= 0.70:
                    bonus += 0.25 * float(team_plan.get("confidence", 0.7))

        for message in partner_messages:
            for proposal in message.get("proposals", []) or []:
                if not isinstance(proposal, dict):
                    continue
                strategy = str(proposal.get("strategy", ""))
                confidence = float(proposal.get("confidence", 0.0))
                slot = proposal.get("target_slot")
                try:
                    slot = int(slot) if slot is not None else None
                except Exception:
                    slot = None
                partner_damage = utility_damage_ratio(proposal.get("self_damage", 0.0))
                my_damage = (
                    float(damage_by_slot.get(slot, 0.0))
                    if slot is not None
                    else utility_damage_sum((summary.get("damage_by_slot", {}) or {}).values())
                )
                combined = my_damage + partner_damage
                if (
                    strategy in {"finish_ko", "double_target_ko", "threat_removal"}
                    and slot is not None
                ):
                    same_target = target_slot == slot or bool(summary.get("spread"))
                    if (
                        bool(getattr(self.config, "communication_solo_claims_enabled", False))
                        and proposal.get("solo_claim")
                        and not proposal.get("partner_required")
                    ):
                        claim_penalty = float(
                            getattr(self.config, "communication_solo_claim_penalty", 1.0) or 1.0
                        )
                        if same_target and my_damage >= 0.15:
                            bonus -= claim_penalty * confidence
                        elif not same_target and move_base_power(action.move) > 0:
                            bonus += 0.15 * confidence
                        continue
                    if (
                        same_target
                        and proposal.get("avoid_partner_overkill")
                        and not proposal.get("partner_required")
                    ):
                        bonus -= 1.10 * confidence
                    elif same_target and combined >= 1.0:
                        bonus += (1.35 if proposal.get("partner_required") else 0.70) * confidence
                    elif same_target:
                        bonus += 0.35 * confidence
                    elif proposal.get("partner_required") and strategy in {
                        "finish_ko",
                        "double_target_ko",
                    }:
                        bonus -= 0.60 * confidence
                elif strategy == "protect_partner_removes_threat" and slot is not None:
                    if target_slot == slot and my_damage > 0.0:
                        bonus += 1.20 * confidence
                    elif move_base_power(action.move) <= 0:
                        bonus -= 0.45 * confidence
                elif strategy == "speed_control_damage":
                    if move_id(action.move) in SPEED_CONTROL_MOVES:
                        bonus += 0.55 * confidence
                elif strategy == "spread_pressure" and bool(summary.get("spread")):
                    bonus += 0.35 * confidence
                elif strategy == "terrain_control":
                    boosted_type = str(proposal.get("boosted_type") or "")
                    if mid in TERRAIN_MOVES:
                        bonus += 0.45 * confidence
                    elif (
                        boosted_type
                        and _move_type_name(action.move) == boosted_type
                        and move_base_power(action.move) > 0
                    ):
                        bonus += 0.35 * confidence
                    elif (
                        proposal.get("field_effect") == "psychicterrain"
                        and mid in FAKE_OUT_MOVES
                        and _psychic_terrain_blocks_priority(battle, action.target)
                    ):
                        bonus -= 0.35 * confidence
                elif strategy == "pivot_cycle":
                    if mid in PIVOT_MOVES:
                        bonus += 0.35 * confidence
                    elif bool(summary.get("spread")) or my_damage >= 0.65:
                        bonus += 0.15 * confidence
                elif strategy == "self_activation_combo":
                    if bool(summary.get("ally_activation")):
                        bonus += 0.70 * confidence
                    elif move_base_power(action.move) > 0 and not bool(summary.get("ally_target")):
                        bonus += 0.12 * confidence
                elif strategy == "setup_redirection":
                    role = str(proposal.get("partner_role") or "")
                    if role == "protector" and (mid in REDIRECTION_MOVES or mid in FAKE_OUT_MOVES):
                        bonus += 0.85 * confidence
                    elif role == "setup_sweeper" and mid in SETUP_MOVES:
                        bonus += 0.80 * confidence
                elif strategy == "screens_positioning":
                    if mid in SCREEN_MOVES:
                        bonus += 0.55 * confidence
                    elif mid in SETUP_MOVES or my_damage >= 0.65:
                        bonus += 0.18 * confidence

        if mid in RECOIL_MOVES:
            attacker = battle.active_pokemon[0] if battle.active_pokemon else None
            hp = safe_hp_fraction(attacker)
            partner_damage = self._partner_best_damage_from_messages(partner_messages)
            if target_slot is not None:
                target_hp = self._target_hp_for_slot(battle, target_slot) or 1.0
                my_damage = float(damage_by_slot.get(target_slot, 0.0))
                if partner_damage.get(int(target_slot), 0.0) >= max(0.85, target_hp):
                    bonus -= 1.10
                if hp < 0.55 and my_damage >= target_hp:
                    bonus -= 0.65 + 0.85 * max(0.0, 0.55 - hp)

        own_slots = {
            p.get("target_slot")
            for p in (own_message.get("proposals", []) or [])
            if isinstance(p, dict)
            and p.get("strategy") in {"finish_ko", "double_target_ko", "threat_removal"}
        }
        partner_slots = {
            p.get("target_slot")
            for m in partner_messages
            for p in (m.get("proposals", []) or [])
            if isinstance(p, dict)
            and p.get("strategy") in {"finish_ko", "double_target_ko", "threat_removal"}
        }
        if target_slot is not None and target_slot in own_slots and target_slot in partner_slots:
            bonus += 0.35
        return bonus

    def _build_structured_commitment_message(
        self,
        battle: MultiBattle,
        selected_action: SlotAction,
        selected_score: float,
        own_message: Dict[str, Any],
        partner_messages: List[Dict[str, Any]],
        team_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        selected = self._public_candidate_summary(battle, selected_score, selected_action)
        best_partner = self._best_partner_proposal_for_action(selected, partner_messages)
        if team_plan and not best_partner:
            selected_slot = selected.get("target_slot")
            plan_slot = team_plan.get("target_slot")
            plan_strategy = str(team_plan.get("strategy", ""))
            if (
                plan_strategy == "protect_partner_removes_threat"
                and str(team_plan.get("source_agent", "")) == self.agent_name
                and selected.get("protect")
            ):
                best_partner = team_plan
            elif plan_slot is not None and (selected_slot == plan_slot or selected.get("spread")):
                best_partner = team_plan
        speech = "accept" if best_partner else "commit"
        if partner_messages and not best_partner:
            speech = "reject"
        return {
            "schema": "vgc_structured_comm_v1",
            "speech_act": speech,
            "agent": self.agent_name,
            "role": str(_multi_side_role(battle) or "unknown"),
            "battle_tag": str(safe_getattr(battle, "battle_tag", "unknown")),
            "turn": int(safe_getattr(battle, "turn", 0) or 0),
            "content": {
                "decision": f"{speech}:{selected.get('label')}",
                "selected": selected,
                "accepted_partner_plan": best_partner,
                "team_plan": team_plan,
                "own_top_proposal": (own_message.get("proposals") or [None])[0],
                "reason": "selected_action_matches_structured_plan"
                if best_partner
                else "local_candidate_scores_over_partner_proposal",
            },
        }

    @staticmethod
    def _best_partner_proposal_for_action(
        selected: Dict[str, Any],
        partner_messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        target_slot = selected.get("target_slot")
        best = None
        best_score = -1.0
        for message in partner_messages:
            for proposal in message.get("proposals", []) or []:
                if not isinstance(proposal, dict):
                    continue
                slot = proposal.get("target_slot")
                same = slot is not None and target_slot == slot
                spread_ok = bool(selected.get("spread")) and slot in (
                    selected.get("damage_by_slot", {}) or {}
                )
                if not (
                    same
                    or spread_ok
                    or proposal.get("strategy")
                    in {
                        "spread_pressure",
                        "speed_control_damage",
                        "terrain_control",
                        "pivot_cycle",
                        "self_activation_combo",
                        "setup_redirection",
                        "screens_positioning",
                    }
                ):
                    continue
                score = float(proposal.get("confidence", 0.0))
                if score > best_score:
                    best_score = score
                    best = proposal
        return best

    def _publish_single_slot_intent(
        self, battle: MultiBattle, scored: List[Tuple[float, SlotAction]], message: Dict[str, Any]
    ) -> None:
        key = _multi_coordination_key(battle, _multi_side_role(battle))
        bucket = MULTI_INTENT_BLACKBOARD.setdefault(key, {})
        bucket[self.agent_name] = {
            "agent_name": self.agent_name,
            "candidates": self._candidate_summaries(
                battle, scored[: max(8, int(getattr(self.config, "shared_joint_top_k", 12) or 12))]
            ),
            "message": json_safe(message),
            "commitment": None,
            "updated_at": time.time(),
        }
        self._cleanup_intent_blackboard()

    def _publish_single_slot_commitment(self, battle: MultiBattle, message: Dict[str, Any]) -> None:
        key = _multi_coordination_key(battle, _multi_side_role(battle))
        bucket = MULTI_INTENT_BLACKBOARD.get(key, {})
        if self.agent_name in bucket:
            bucket[self.agent_name]["commitment"] = json_safe(message)
            bucket[self.agent_name]["updated_at"] = time.time()
        try:
            mem = self._shared_memory(battle)
            mem.setdefault("commitments", {})[self.agent_name] = json_safe(message)
        except Exception:
            pass

    def _partner_packets(self, battle: MultiBattle) -> List[Dict[str, Any]]:
        key = _multi_coordination_key(battle, _multi_side_role(battle))
        bucket = MULTI_INTENT_BLACKBOARD.get(key, {})
        packets = []
        for agent_name, intent in bucket.items():
            if agent_name == self.agent_name or str(agent_name).startswith("__"):
                continue
            packet = dict(intent)
            packet.setdefault("agent_name", agent_name)
            packets.append(packet)
        return packets

    def _coordination_key_for_turn(self, battle: MultiBattle, turn: int) -> str:
        role = _multi_side_role(battle)
        battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
        side = "odd" if _role_num(role) in {1, 3} else "even"
        return f"{battle_key}:{max(0, int(turn))}:{side}"

    def _partner_packets_for_key(self, key: str) -> List[Dict[str, Any]]:
        bucket = MULTI_INTENT_BLACKBOARD.get(key, {})
        packets = []
        for agent_name, intent in bucket.items():
            if agent_name == self.agent_name or str(agent_name).startswith("__"):
                continue
            packet = dict(intent)
            packet.setdefault("agent_name", agent_name)
            packets.append(packet)
        return packets

    def _partner_packets_for_mode(
        self, battle: MultiBattle, communication_mode: str
    ) -> List[Dict[str, Any]]:
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        delay_steps = int(getattr(self.config, "communication_delay_steps", 0) or 0)
        if communication_mode == "delayed_messages":
            delay_steps = max(1, delay_steps)
        if delay_steps > 0:
            return self._partner_packets_for_key(
                self._coordination_key_for_turn(battle, turn - delay_steps)
            )
        if communication_mode == "shuffled_messages":
            keys = [
                self._coordination_key_for_turn(battle, prev_turn)
                for prev_turn in range(max(0, turn - 6), turn + 1)
            ]
            candidates: List[List[Dict[str, Any]]] = [
                packets
                for packets in (self._partner_packets_for_key(key) for key in keys)
                if packets
            ]
            if candidates:
                return self._communication_rng(battle, "shuffle").choice(candidates)
        return self._partner_packets(battle)

    def _intervene_partner_packets(
        self,
        battle: MultiBattle,
        communication_mode: str,
        partner_packets: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        packets, diagnostics = apply_packet_intervention(
            partner_packets,
            mode=communication_mode,
            rng=self._communication_rng(battle, "intervention"),
            noise_std=float(getattr(self.config, "communication_noise_std", 0.0) or 0.0),
            dropout_prob=float(getattr(self.config, "communication_dropout_prob", 0.0) or 0.0),
            zero_agent=str(getattr(self.config, "communication_zero_agent", "") or ""),
        )
        if bool(getattr(self.config, "communication_use_gate", False)):
            before_gate = len(packets)
            if communication_mode in {"hard_gated_messages", "hard_gate"}:
                packets = gate_packets(
                    packets,
                    float(getattr(self.config, "communication_gate_threshold", 0.0) or 0.0),
                )
            diagnostics["gated_packets"] = before_gate - len(packets)
        else:
            diagnostics["gated_packets"] = 0
        return packets, diagnostics

    @staticmethod
    def _partner_candidates_from_packets(packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for intent in packets:
            candidates = intent.get("candidates", [])
            if candidates:
                source_agent = str(
                    intent.get("agent_name")
                    or (intent.get("message") or {}).get("agent")
                    or "partner"
                )
                return [dict(candidate, source_agent=source_agent) for candidate in candidates]
        return []


__all__ = ["MultiAgentCommRuntimeMixin"]
