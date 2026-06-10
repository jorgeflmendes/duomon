from __future__ import annotations

from .multi_agent_context import *


class MultiAgentJointSelectionMixin:
    def _select_shared_joint_action(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
        partner_candidates: List[Dict[str, Any]],
        partner_placeholder: SlotAction,
    ) -> Optional[Tuple[float, SlotAction, Dict[str, Any]]]:
        if not self.config.use_shared_joint_selector or not partner_candidates:
            return None

        top_k = max(1, int(self.config.shared_joint_top_k or 1))
        local_top = scored[:top_k]
        partner_top = [
            self._normalise_partner_candidate_summary(candidate)
            for candidate in partner_candidates[:top_k]
        ]
        partner_top = [candidate for candidate in partner_top if candidate.get("kind") == "move"]
        if not local_top or not partner_top:
            return None

        ranked: List[Tuple[float, float, SlotAction, Dict[str, Any]]] = []
        threat = self.slot_agent.tactical.threat_model.analyze(battle)
        state_snapshot = self._battle_state_snapshot(battle)
        for local_rank, (local_score, action) in enumerate(local_top):
            candidate_action = JointAction(action, partner_placeholder)
            candidate_action.search_score = float(local_score)
            if self._contextual_action_is_bad(battle, candidate_action):
                continue

            my_summary = self._candidate_summary(battle, local_score, action)
            if my_summary.get("kind") != "move":
                continue

            for partner_rank, partner_summary in enumerate(partner_top):
                pair_score, details = self._shared_joint_pair_score(
                    battle,
                    threat,
                    my_summary,
                    partner_summary,
                    float(local_score),
                    local_rank,
                    partner_rank,
                )
                if not math.isfinite(pair_score):
                    continue
                details["local_rank"] = int(local_rank)
                details["partner_rank"] = int(partner_rank)
                details["state"] = state_snapshot
                pair_score, details = self._apply_ctde_joint_reranker(pair_score, details)
                ranked.append((pair_score, float(local_score), action, details))

        if not ranked:
            return None

        ranked.sort(
            key=lambda item: (
                item[0],
                -int(item[3].get("local_rank", 99)),
                -int(item[3].get("partner_rank", 99)),
                str(item[3].get("local_signature", "")),
                str(item[3].get("partner_signature", "")),
            ),
            reverse=True,
        )
        selected = self._select_barrier_joint_plan(battle, ranked)
        pair_score, _local_score, action, details = selected
        self._log_ctde_joint_example(battle, ranked, details)
        return float(pair_score), action, details

    def _battle_state_snapshot(self, battle: MultiBattle) -> Dict[str, float]:
        actives = list(battle.active_pokemon[:2]) if battle.active_pokemon else []
        opps = list(battle.opponent_active_pokemon[:2]) if battle.opponent_active_pokemon else []

        def hp(mon: Any) -> float:
            if mon is None or is_fainted(mon):
                return 0.0
            return float(safe_hp_fraction(mon))

        def spd(mon: Any) -> float:
            if mon is None or is_fainted(mon):
                return 0.0
            return float(safe_speed(mon))

        def status_eq(mon: Any, code: str) -> float:
            if mon is None or is_fainted(mon):
                return 0.0
            s = str(safe_compact_name(safe_getattr(mon, "status", None)) or "").lower()
            return float(s == code)

        self0 = actives[0] if len(actives) > 0 else None
        self1 = actives[1] if len(actives) > 1 else None
        opp0 = opps[0] if len(opps) > 0 else None
        opp1 = opps[1] if len(opps) > 1 else None

        self0_hp = hp(self0)
        self1_hp = hp(self1)
        opp0_hp = hp(opp0)
        opp1_hp = hp(opp1)

        self0_spd = spd(self0)
        self1_spd = spd(self1)
        opp_spds = [spd(o) for o in (opp0, opp1) if o is not None and not is_fainted(o)]
        opp_spd_avg = (sum(opp_spds) / len(opp_spds)) if opp_spds else 0.0

        weather = _lower_names(safe_getattr(battle, "weather", None)) or []
        fields = _battle_field_names(battle) or []
        own_side = _battle_side_condition_names(battle, own_side=True) or []
        opp_side = _battle_side_condition_names(battle, own_side=False) or []

        screens = {"lightscreen", "reflect", "auroraveil"}
        hazards = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}

        return {
            "self0_hp": self0_hp,
            "self1_hp": self1_hp,
            "opp0_hp": opp0_hp,
            "opp1_hp": opp1_hp,
            "self0_speed_adv": ((self0_spd - opp_spd_avg) / 100.0) if opp_spd_avg > 0 else 0.0,
            "self1_speed_adv": ((self1_spd - opp_spd_avg) / 100.0) if opp_spd_avg > 0 else 0.0,
            "weather_rain": float("raindance" in weather),
            "weather_sun": float("sunnyday" in weather),
            "weather_sand": float("sandstorm" in weather),
            "weather_snow": float("snow" in weather or "hail" in weather),
            "terrain_electric": float("electricterrain" in fields),
            "terrain_grassy": float("grassyterrain" in fields),
            "terrain_psychic": float("psychicterrain" in fields),
            "terrain_misty": float("mistyterrain" in fields),
            "trick_room": float("trickroom" in fields),
            "tailwind_self": float("tailwind" in own_side),
            "tailwind_opp": float("tailwind" in opp_side),
            "screens_self": float(any(s in own_side for s in screens)),
            "screens_opp": float(any(s in opp_side for s in screens)),
            "hazards_self": float(sum(1 for s in own_side if s in hazards)),
            "hazards_opp": float(sum(1 for s in opp_side if s in hazards)),
            "alive_count_self": float(
                sum(1 for m in actives if m is not None and not is_fainted(m))
            ),
            "alive_count_opp": float(sum(1 for m in opps if m is not None and not is_fainted(m))),
            "min_self_hp": min(self0_hp, self1_hp) if (self0_hp > 0.0 or self1_hp > 0.0) else 0.0,
            "team_hp_diff": (self0_hp + self1_hp) - (opp0_hp + opp1_hp),
            "self0_burned": status_eq(self0, "brn"),
            "self0_paralyzed": status_eq(self0, "par"),
            "self1_burned": status_eq(self1, "brn"),
            "self1_paralyzed": status_eq(self1, "par"),
            "opp0_burned": status_eq(opp0, "brn"),
            "opp0_poisoned": min(1.0, status_eq(opp0, "psn") + status_eq(opp0, "tox")),
            "opp1_burned": status_eq(opp1, "brn"),
            "opp1_poisoned": min(1.0, status_eq(opp1, "psn") + status_eq(opp1, "tox")),
            "turn_norm": min(1.0, float(safe_getattr(battle, "turn", 0) or 0) / 30.0),
        }

    def _apply_ctde_joint_reranker(
        self,
        pair_score: float,
        details: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        enriched = dict(details)
        enriched["base_pair_score"] = float(pair_score)




        enriched["benchmark_type"] = self.config.benchmark_type
        feature_details = enriched
        if not bool(getattr(self.config, "communication_critic_uses_messages", True)):
            feature_details = dict(enriched)
            feature_details.update(
                {
                    "partner_score": 0.0,
                    "partner_rank": 0.0,
                    "partner_target_slot": None,
                    "partner_move_id": "",
                    "combined_damage_by_slot": {},
                    "split_target_value": 0.0,
                    "split_ko_count": 0,
                    "split_pressure_count": 0,
                }
            )
        features = ctde_runtime_features_from_details(feature_details)
        enriched["features"] = [float(value) for value in features]
        enriched["ctde_score"] = 0.0
        enriched["ctde_bonus"] = 0.0
        adjusted = float(pair_score)

        if self.ctde_joint_reranker is not None:
            weight = float(getattr(self.config, "ctde_joint_reranker_weight", 0.0) or 0.0)
            if weight > 0.0:
                clip = float(getattr(self.config, "ctde_joint_reranker_clip", 0.75) or 0.75)
                raw = float(self.ctde_joint_reranker.predict(features))

                if getattr(self.ctde_joint_reranker, "objective", "") == "value_regression":
                    try:
                        p_win = 1.0 / (1.0 + math.exp(-raw))
                    except OverflowError:
                        p_win = 1.0 if raw > 0.0 else 0.0
                    signal = (p_win - 0.5) * 2.0
                else:
                    signal = raw
                bonus = weight * max(-clip, min(clip, signal))
                adjusted += bonus
                enriched["ctde_score"] = raw
                enriched["ctde_bonus"] = float(bonus)

        enriched["pair_score"] = adjusted
        return adjusted, enriched

    def _log_ctde_joint_example(
        self,
        battle: MultiBattle,
        ranked: List[Tuple[float, float, SlotAction, Dict[str, Any]]],
        selected_details: Dict[str, Any],
    ) -> None:
        if not getattr(self.config, "log_ctde_joint_examples", True):
            return
        if not self.config.ctde_joint_dataset_path:
            return
        if self.config.benchmark_type == "vs_random":
            return
        try:
            candidates = []
            for pair_score, _local_score, _action, details in ranked[:64]:
                pair_signature = self._pair_signature(details)
                features = details.get("features")
                if not isinstance(features, list):
                    features = ctde_runtime_features_from_details(details)
                candidates.append(
                    {
                        "pair_signature": pair_signature,
                        "pair_score": float(pair_score),
                        "base_pair_score": float(
                            details.get("base_pair_score", pair_score) or pair_score
                        ),
                        "local_signature": str(details.get("local_signature", "")),
                        "partner_signature": str(details.get("partner_signature", "")),
                        "local_label": str(details.get("local_label", "")),
                        "partner_label": str(details.get("partner_label", "")),
                        "local_move_id": str(details.get("local_move_id", "")),
                        "partner_move_id": str(details.get("partner_move_id", "")),
                        "local_target_slot": details.get("local_target_slot"),
                        "partner_target_slot": details.get("partner_target_slot"),
                        "reason": str(details.get("reason", "")),
                        "features": [float(value) for value in features],
                        "details": json_safe(
                            {
                                "pair_score": pair_score,
                                "base_pair_score": details.get("base_pair_score", pair_score),
                                "local_score": details.get("local_score", 0.0),
                                "partner_score": details.get("partner_score", 0.0),
                                "local_rank": details.get("local_rank", 0),
                                "partner_rank": details.get("partner_rank", 0),
                                "local_move_id": details.get("local_move_id", ""),
                                "partner_move_id": details.get("partner_move_id", ""),
                                "local_target_slot": details.get("local_target_slot"),
                                "partner_target_slot": details.get("partner_target_slot"),
                                "local_damage_target": details.get("local_damage_target", 0.0),
                                "partner_damage_target": details.get("partner_damage_target", 0.0),
                                "local_damage_by_slot": details.get("local_damage_by_slot", {}),
                                "partner_damage_by_slot": details.get("partner_damage_by_slot", {}),
                                "local_damage_total": details.get("local_damage_total", 0.0),
                                "partner_damage_total": details.get("partner_damage_total", 0.0),
                                "target_slot": details.get("target_slot"),
                                "target_damage": details.get("target_damage"),
                                "ko_slots": details.get("ko_slots", []),
                                "combined_damage_by_slot": details.get(
                                    "combined_damage_by_slot", {}
                                ),
                                "trade_value": details.get("trade_value", 0.0),
                                "survival_value": details.get("survival_value", 0.0),
                                "hard_benchmark_value": details.get("hard_benchmark_value", 0.0),
                                "split_target_value": details.get("split_target_value", 0.0),
                                "split_ko_count": details.get("split_ko_count", 0),
                                "split_pressure_count": details.get("split_pressure_count", 0),
                                "ctde_score": details.get("ctde_score", 0.0),
                                "ctde_bonus": details.get("ctde_bonus", 0.0),
                            }
                        ),
                    }
                )
            selected_pair_signature = self._pair_signature(selected_details)
            state_snapshot = selected_details.get("state") or (
                ranked[0][3].get("state") if ranked else None
            )
            record = {
                "battle_tag": str(safe_getattr(battle, "battle_tag", "unknown")),
                "turn": int(safe_getattr(battle, "turn", 0) or 0),
                "agent_name": self.agent_name,
                "benchmark_type": self.config.benchmark_type,
                "chosen_pair_signature": selected_pair_signature,
                "chosen_local_signature": str(selected_details.get("local_signature", "")),
                "chosen_partner_signature": str(selected_details.get("partner_signature", "")),
                "chosen_reason": str(selected_details.get("reason", "")),
                "state_snapshot": json_safe(state_snapshot) if state_snapshot else None,
                "candidates": candidates,
            }
            ensure_parent_dir(self.config.ctde_joint_dataset_path)
            with open(self.config.ctde_joint_dataset_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            return

    @staticmethod
    def _pair_signature(details: Dict[str, Any]) -> str:
        return f"{details.get('local_signature', '')}||{details.get('partner_signature', '')}"

    def _select_barrier_joint_plan(
        self,
        battle: MultiBattle,
        ranked: List[Tuple[float, float, SlotAction, Dict[str, Any]]],
    ) -> Tuple[float, float, SlotAction, Dict[str, Any]]:
        if not ranked or not bool(getattr(self.config, "use_joint_plan_barrier", True)):
            return ranked[0]

        key = _multi_coordination_key(battle, _multi_side_role(battle))
        bucket = MULTI_INTENT_BLACKBOARD.setdefault(key, {})
        plan_key = "__joint_plan__"
        existing = bucket.get(plan_key)
        if isinstance(existing, dict):
            planned = self._ranked_action_matching_joint_plan(existing, ranked)
            if planned is not None:
                return planned

        pair_score, local_score, action, details = ranked[0]
        local_agent = str(details.get("local_agent") or self.agent_name)
        partner_agent = str(details.get("partner_agent") or "partner")
        plan = {
            "kind": "joint_plan",
            "created_by": self.agent_name,
            "battle_tag": str(safe_getattr(battle, "battle_tag", "unknown")),
            "turn": int(safe_getattr(battle, "turn", 0) or 0),
            "updated_at": time.time(),
            "actions": {
                local_agent: str(details.get("local_signature", "")),
                partner_agent: str(details.get("partner_signature", "")),
            },
            "labels": {
                local_agent: str(details.get("local_label", "")),
                partner_agent: str(details.get("partner_label", "")),
            },
            "reason": str(details.get("reason", "shared-joint-plan")),
            "pair_score": float(pair_score),
            "pair_bonus": float(details.get("pair_bonus", 0.0) or 0.0),
            "target_slot": details.get("target_slot"),
            "ko_slots": list(details.get("ko_slots", []) or []),
        }
        bucket[plan_key] = json_safe(plan)
        selected_details = dict(details)
        selected_details["shared_plan_barrier"] = "leader"
        selected_details["joint_plan"] = plan
        return float(pair_score), float(local_score), action, selected_details

    def _ranked_action_matching_joint_plan(
        self,
        plan: Dict[str, Any],
        ranked: List[Tuple[float, float, SlotAction, Dict[str, Any]]],
    ) -> Optional[Tuple[float, float, SlotAction, Dict[str, Any]]]:
        actions = plan.get("actions", {}) if isinstance(plan, dict) else {}
        if not isinstance(actions, dict):
            return None
        desired_signature = str(actions.get(self.agent_name) or "")
        if not desired_signature:
            return None
        for pair_score, local_score, action, details in ranked:
            if str(details.get("local_signature", "")) != desired_signature:
                continue
            selected_details = dict(details)
            selected_details["shared_plan_barrier"] = "follower"
            selected_details["joint_plan"] = plan
            selected_details["reason"] = str(
                plan.get("reason") or selected_details.get("reason") or "shared-joint-plan"
            )
            return float(pair_score), float(local_score), action, selected_details
        return None

    @staticmethod
    def _live_or_blind_opp_slots(
        battle: MultiBattle,
        own_damage: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[int, Any]]:
        visible = [
            (slot, opp)
            for slot, opp in enumerate(battle.opponent_active_pokemon[:2])
            if opp is not None and not is_fainted(opp)
        ]
        if visible:
            return visible
        slots = sorted({int(slot) for slot in (own_damage or {}).keys() if int(slot) in {0, 1}})
        if not slots:
            slots = [0, 1]
        return [(slot, None) for slot in slots]

    @staticmethod
    def _normalise_partner_candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
        def slot_map(raw: Any) -> Dict[int, float]:
            values: Dict[int, float] = {}
            if not isinstance(raw, dict):
                return values
            for key, value in raw.items():
                try:
                    values[int(key)] = float(value)
                except Exception:
                    continue
            return values

        def string_float_map(raw: Any) -> Dict[str, float]:
            values: Dict[str, float] = {}
            if not isinstance(raw, dict):
                return values
            for key, value in raw.items():
                try:
                    values[str(key)] = float(value)
                except Exception:
                    continue
            return values

        move = candidate.get("move")
        mid = str(candidate.get("move_id") or (move_id(move) if move is not None else ""))
        bp = float(candidate.get("bp", move_base_power(move) if move is not None else 0.0) or 0.0)
        damage_by_slot = slot_map(candidate.get("damage_by_slot", {}))
        ko_by_slot = slot_map(candidate.get("ko_by_slot", {}))
        target_slot = candidate.get("target_slot", None)
        try:
            target_slot = int(target_slot) if target_slot is not None else None
        except Exception:
            target_slot = None
        return {
            "signature": str(candidate.get("signature") or candidate.get("label") or mid),
            "label": str(candidate.get("label") or mid or "partner-action"),
            "source_agent": str(candidate.get("source_agent") or "partner"),
            "score": float(candidate.get("score", 0.0) or 0.0),
            "slot": 1,
            "kind": str(candidate.get("kind", "move")),
            "move": move,
            "move_id": mid,
            "bp": bp,
            "accuracy": float(
                candidate.get("accuracy", move_accuracy(move) if move is not None else 1.0) or 1.0
            ),
            "priority": float(candidate.get("priority", 0.0) or 0.0),
            "attacker_speed": float(candidate.get("attacker_speed", 0.0) or 0.0),
            "target": candidate.get("target"),
            "target_slot": target_slot,
            "target_hp": candidate.get("target_hp"),
            "spread": bool(candidate.get("spread", False)),
            "support": bool(candidate.get("support", bp <= 0.0)),
            "protect": bool(candidate.get("protect", mid in PROTECT_MOVES)),
            "recoil": bool(candidate.get("recoil", mid in RECOIL_MOVES)),
            "self_sacrifice": bool(candidate.get("self_sacrifice", mid in SELF_SACRIFICE_MOVES)),
            "tera": bool(candidate.get("tera", False)),
            "speed_control": bool(candidate.get("speed_control", mid in SPEED_CONTROL_MOVES)),
            "damage": string_float_map(candidate.get("damage", {})),
            "damage_by_slot": damage_by_slot,
            "ko_by_slot": ko_by_slot,
            "damage_sum": float(candidate.get("damage_sum", sum(damage_by_slot.values())) or 0.0),
            "ko_sum": float(candidate.get("ko_sum", sum(ko_by_slot.values())) or 0.0),
        }


__all__ = ["MultiAgentJointSelectionMixin"]
