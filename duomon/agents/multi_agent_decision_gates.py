from __future__ import annotations

from .multi_agent_context import *


class MultiAgentDecisionGateMixin:
    @staticmethod
    def _normalised_type_name(raw: Any) -> str:
        return str(getattr(raw, "name", raw) or "").lower().replace(" ", "").replace("-", "")

    @classmethod
    def _usable_type_name(cls, raw: Any) -> str:
        value = cls._normalised_type_name(raw)
        if value in {"", "none", "unknown", "false", "true", "0", "1"}:
            return ""
        return value

    def _self_tera_type_name(self, battle: MultiBattle, active: Any) -> str:
        for attr in ("tera_type", "terastallized_type", "_tera_type"):
            value = self._usable_type_name(safe_getattr(active, attr, ""))
            if value:
                return value
        raw = self._raw_request(battle)
        active_entries = raw.get("active", []) if isinstance(raw, dict) else []
        if isinstance(active_entries, list) and active_entries:
            first = active_entries[0] if isinstance(active_entries[0], dict) else {}
            for key in ("teraType", "tera_type", "teratype", "canTerastallize", "can_terastallize"):
                value = self._usable_type_name(first.get(key, ""))
                if value:
                    return value
        side = raw.get("side", {}) if isinstance(raw, dict) else {}
        pokemon_entries = side.get("pokemon", []) if isinstance(side, dict) else []
        if isinstance(pokemon_entries, list):
            active_ident = self._usable_type_name(safe_getattr(active, "identifier", ""))
            for entry in pokemon_entries:
                if not isinstance(entry, dict):
                    continue
                if (
                    not bool(entry.get("active", False))
                    and active_ident
                    and self._usable_type_name(entry.get("ident", "")) != active_ident
                ):
                    continue
                for key in (
                    "teraType",
                    "tera_type",
                    "teratype",
                    "canTerastallize",
                    "can_terastallize",
                ):
                    value = self._usable_type_name(entry.get(key, ""))
                    if value:
                        return value
        return ""

    @staticmethod
    def _type_chart_multiplier(move_type: str, target_types: List[str]) -> float:
        move_type = str(move_type or "").upper()
        if not move_type:
            return 1.0
        try:
            type_chart = GenData.from_format("gen9randombattle").type_chart
            multiplier = 1.0
            for target_type in target_types or []:
                row = type_chart.get(str(target_type).upper(), {})
                multiplier *= float(row.get(move_type, 1.0))
            return multiplier
        except Exception:
            return 1.0

    def _tera_defensive_delta(
        self,
        battle: MultiBattle,
        active: Any,
        tera_type: str,
        responses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        current_types = _mon_type_names(active)
        if not current_types or not tera_type:
            return {"current_pressure": 0.0, "tera_pressure": 0.0, "defensive_delta": 0.0}
        current_pressure = 0.0
        tera_pressure = 0.0
        for response in (
            responses if responses is not None else self._predict_opponent_responses(battle)
        ):
            if 0 not in [
                int(slot) for slot in response.get("target_slots", []) if isinstance(slot, int)
            ]:
                continue
            move = response.get("move")
            if move is None or move_base_power(move) <= 0:
                continue
            move_type = _move_type_name(move)
            weight = float(response.get("weight", 1.0) or 1.0)
            pressure = max(
                0.12,
                utility_damage_ratio(
                    (response.get("damage_by_slot", {}) or {}).get(0, 0.0), cap=1.25
                ),
            )
            current_pressure += (
                weight * pressure * self._type_chart_multiplier(move_type, current_types)
            )
            tera_pressure += weight * pressure * self._type_chart_multiplier(move_type, [tera_type])
        return {
            "current_pressure": float(current_pressure),
            "tera_pressure": float(tera_pressure),
            "defensive_delta": float(current_pressure - tera_pressure),
        }

    def _tera_offensive_delta(
        self,
        battle: MultiBattle,
        action: SlotAction,
        summary: Dict[str, Any],
        tera_type: str,
    ) -> Dict[str, Any]:
        active = battle.active_pokemon[0] if battle.active_pokemon else None
        move = action.move
        if active is None or move is None or move_base_power(move) <= 0 or not tera_type:
            return {"offensive_delta": 0.0, "secures_ko": False, "stab_gain": 0.0}
        move_type = _move_type_name(move)
        current_types = _mon_type_names(active)
        if move_type != tera_type:
            return {"offensive_delta": 0.0, "secures_ko": False, "stab_gain": 0.0}
        stab_gain = 0.33 if move_type in current_types else 0.50
        damage_sum = utility_damage_sum(
            (summary.get("damage_by_slot", {}) or {}).values(), cap=1.35
        )
        secures_ko = False
        for raw_slot, raw_ratio in (summary.get("damage_by_slot", {}) or {}).items():
            try:
                slot = int(raw_slot)
            except Exception:
                continue
            opp = (
                battle.opponent_active_pokemon[slot]
                if slot < len(battle.opponent_active_pokemon)
                else None
            )
            hp = max(0.05, safe_hp_fraction(opp) if opp is not None else 1.0)
            ratio = float(raw_ratio or 0.0)
            if ratio < 0.92 * hp and ratio * (1.0 + stab_gain) >= 0.92 * hp:
                secures_ko = True
                break
        return {
            "offensive_delta": float(damage_sum * stab_gain),
            "secures_ko": bool(secures_ko),
            "stab_gain": float(stab_gain),
        }

    def _tera_gate_evaluation(
        self,
        battle: MultiBattle,
        action: SlotAction,
        summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(action_uses_tera(action)):
            return {"allowed": True, "reason": "not_tera", "adjustment": 0.0}
        if not bool(getattr(self.config, "tera_gate_enabled", True)):
            return {"allowed": True, "reason": "tera_gate_disabled", "adjustment": 0.0}

        active = battle.active_pokemon[0] if battle.active_pokemon else None
        tera_type = self._self_tera_type_name(battle, active)
        offensive = self._tera_offensive_delta(battle, action, summary, tera_type)
        defensive = self._tera_defensive_delta(
            battle,
            active,
            tera_type,
            [context.get("primary_response")]
            if isinstance(context.get("primary_response"), dict)
            else None,
        )
        turn = int(context.get("turn", 0) or 0)
        risk = float(context.get("risk", 0.0) or 0.0)
        offensive_delta = float(offensive.get("offensive_delta", 0.0))
        defensive_delta = float(defensive.get("defensive_delta", 0.0))
        secures_ko = bool(offensive.get("secures_ko", False))
        min_gain = float(getattr(self.config, "tera_min_offensive_gain", 0.18) or 0.18)
        allowed_reason = ""
        if secures_ko:
            allowed_reason = "tera_secures_same_turn_ko"
        elif defensive_delta >= 0.22 and risk >= 0.35:
            allowed_reason = "tera_improves_defensive_matchup"
        elif (
            turn > 1
            and offensive_delta >= min_gain
            and float(summary.get("ko_sum", 0.0) or 0.0) >= 0.55
            and risk < 0.75
        ):
            allowed_reason = "tera_adds_decisive_offensive_pressure"

        bad_defense = defensive_delta <= -0.18 and risk >= 0.30
        if turn <= 1 and not allowed_reason:
            return {
                "allowed": False,
                "reason": "reject_turn1_without_clear_ko_or_defensive_gain",
                "tera_type": tera_type,
                "offensive_delta": offensive_delta,
                "defensive_delta": defensive_delta,
                "adjustment": -float(getattr(self.config, "tera_turn1_penalty", 3.75) or 3.75),
            }
        if bad_defense and not secures_ko:
            return {
                "allowed": False,
                "reason": "reject_tera_worsens_incoming_type_matchup",
                "tera_type": tera_type,
                "offensive_delta": offensive_delta,
                "defensive_delta": defensive_delta,
                "adjustment": -float(getattr(self.config, "tera_reject_penalty", 4.50) or 4.50),
            }
        if not allowed_reason and offensive_delta < min_gain and defensive_delta < 0.10:
            return {
                "allowed": False,
                "reason": "reject_tera_low_expected_value",
                "tera_type": tera_type,
                "offensive_delta": offensive_delta,
                "defensive_delta": defensive_delta,
                "adjustment": -0.65
                * float(getattr(self.config, "tera_reject_penalty", 4.50) or 4.50),
            }
        bonus = min(0.65, 0.40 * max(0.0, offensive_delta) + 0.35 * max(0.0, defensive_delta))
        return {
            "allowed": True,
            "reason": allowed_reason or "tera_expected_value_positive",
            "tera_type": tera_type,
            "offensive_delta": offensive_delta,
            "defensive_delta": defensive_delta,
            "adjustment": float(bonus),
        }

    def _candidate_preempts_primary_response(
        self,
        battle: MultiBattle,
        summary: Dict[str, Any],
        response: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(response, dict):
            return False
        opp_slot = response.get("opp_slot")
        if not isinstance(opp_slot, int):
            return False
        mid = str(summary.get("move_id", ""))
        raw_target_slot = summary.get("target_slot")
        try:
            target_slot = int(raw_target_slot) if raw_target_slot is not None else -1
        except Exception:
            target_slot = -1
        if mid in FAKE_OUT_MOVES and target_slot == opp_slot:
            return int(safe_getattr(battle, "turn", 0) or 0) <= 2
        damage = self._preemptive_damage_into_opp_slot(summary, opp_slot, response)
        opp = (
            battle.opponent_active_pokemon[opp_slot]
            if opp_slot < len(battle.opponent_active_pokemon)
            else None
        )
        threshold = self._current_hp_ko_threshold(opp, fallback=0.95, margin=0.92)
        return bool(
            damage >= threshold
            or float((summary.get("ko_by_slot", {}) or {}).get(opp_slot, 0.0)) >= 0.80
        )

    def _defensive_gate_evaluation(
        self,
        battle: MultiBattle,
        action: SlotAction,
        summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(getattr(self.config, "risk_gate_enabled", True)):
            return {"reason": "risk_gate_disabled", "adjustment": 0.0}
        risk = float(context.get("risk", 0.0) or 0.0)
        hp = float(context.get("hp", 1.0) or 1.0)
        protect = bool(summary.get("protect", False))
        primary_slot = context.get("primary_opp_slot")
        partner_damage = (context.get("partner_damage_by_slot", {}) or {}).get(primary_slot, 0.0)
        primary = context.get("primary_response")
        preempts = self._candidate_preempts_primary_response(
            battle, summary, primary if isinstance(primary, dict) else None
        )
        partner_can_remove = False
        if isinstance(primary_slot, int):
            opp = (
                battle.opponent_active_pokemon[primary_slot]
                if primary_slot < len(battle.opponent_active_pokemon)
                else None
            )
            partner_can_remove = float(partner_damage or 0.0) >= self._current_hp_ko_threshold(
                opp, fallback=0.95, margin=0.92
            )

        if protect:
            if risk >= 0.48 or hp <= 0.55:
                bonus = float(getattr(self.config, "risk_gate_protect_bonus", 2.30) or 2.30) * min(
                    1.25, risk + max(0.0, 0.65 - hp)
                )
                if partner_can_remove:
                    bonus += 1.15
                return {
                    "reason": "protect_high_self_risk"
                    if not partner_can_remove
                    else "protect_while_partner_can_remove_threat",
                    "risk": risk,
                    "partner_can_remove_threat": partner_can_remove,
                    "adjustment": float(bonus),
                }
            return {"reason": "protect_low_immediate_risk", "risk": risk, "adjustment": -0.25}

        if risk >= 0.66 and not preempts:
            protect_available = bool(
                context.get("protect_candidate_available", context.get("protect_available", False))
            )
            penalty = float(getattr(self.config, "risk_gate_attack_penalty", 1.75) or 1.75) * min(
                1.35, risk
            )
            if not protect_available:
                scale = float(
                    getattr(self.config, "risk_gate_no_protect_penalty_scale", 0.40) or 0.40
                )
                penalty *= max(0.0, min(1.0, scale))
            elif partner_can_remove:
                penalty += 0.95
            if bool(summary.get("support", False)) or float(summary.get("bp", 0.0) or 0.0) <= 0.0:
                penalty += 0.75
            return {
                "reason": (
                    "soft_penalty_risky_action_without_protect"
                    if not protect_available
                    else "penalize_non_defensive_action_under_ko_risk"
                ),
                "risk": risk,
                "preempts_primary_threat": preempts,
                "protect_available": protect_available,
                "partner_can_remove_threat": partner_can_remove,
                "adjustment": -float(penalty),
            }
        if hp <= 0.45 and risk >= 0.38 and not preempts and not bool(summary.get("protect", False)):
            low_hp_weight = float(getattr(self.config, "low_hp_attack_risk_penalty", 1.15) or 1.15)
            attack_quality = float(summary.get("damage_sum", 0.0) or 0.0) + 1.25 * float(
                summary.get("ko_sum", 0.0) or 0.0
            )
            if attack_quality < 0.85:
                penalty = low_hp_weight * (0.55 + risk + max(0.0, 0.50 - hp))
                if bool(summary.get("support", False)):
                    penalty *= 1.15
                return {
                    "reason": "penalize_low_hp_action_without_threat_removal",
                    "risk": risk,
                    "hp": hp,
                    "attack_quality": attack_quality,
                    "preempts_primary_threat": preempts,
                    "adjustment": -float(penalty),
                }
        return {
            "reason": "risk_covered_by_attack" if preempts else "risk_below_gate",
            "risk": risk,
            "preempts_primary_threat": preempts,
            "adjustment": 0.0,
        }

    def _accuracy_gate_evaluation(
        self,
        battle: MultiBattle,
        action: SlotAction,
        summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        move = action.move
        if action.kind != "move" or move is None or move_base_power(move) <= 0:
            return {"reason": "not_damaging_move", "adjustment": 0.0}
        accuracy = move_accuracy(move)
        if accuracy >= 0.95:
            return {"reason": "reliable_move", "accuracy": accuracy, "adjustment": 0.0}
        risk = float(context.get("risk", 0.0) or 0.0)
        ko_sum = float(summary.get("ko_sum", 0.0) or 0.0)
        primary = context.get("primary_response")
        preempts = self._candidate_preempts_primary_response(
            battle, summary, primary if isinstance(primary, dict) else None
        )
        penalty = (0.95 - accuracy) * float(
            getattr(self.config, "accuracy_risk_weight", 2.40) or 2.40
        )
        penalty *= 1.0 + 1.15 * min(1.0, risk)
        if bool(summary.get("spread", False)):
            penalty += 0.35 * (1.0 - accuracy)
        if accuracy < 0.80:
            penalty += 0.35
        if preempts or ko_sum >= 0.80:
            penalty *= 0.45
        return {
            "reason": "penalize_low_accuracy_under_risk",
            "accuracy": float(accuracy),
            "risk": risk,
            "ko_sum": ko_sum,
            "preempts_primary_threat": preempts,
            "adjustment": -float(penalty),
        }

    def _early_survival_gate_evaluation(
        self,
        battle: MultiBattle,
        action: SlotAction,
        summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        turn = int(context.get("turn", 0) or 0)
        risk = float(context.get("risk", 0.0) or 0.0)
        if turn > 2 or risk < 0.42 or not bool(context.get("speed_disadvantage", False)):
            return {"reason": "survival_mode_inactive", "adjustment": 0.0}
        weight = float(getattr(self.config, "early_survival_mode_weight", 1.15) or 1.15)
        primary = context.get("primary_response")
        preempts = self._candidate_preempts_primary_response(
            battle, summary, primary if isinstance(primary, dict) else None
        )
        if bool(summary.get("protect", False)):
            return {"reason": "early_survival_prefers_protect", "adjustment": 0.85 * weight}
        if preempts or float(summary.get("priority", 0.0) or 0.0) > 0.0:
            return {
                "reason": "early_survival_threat_removed_or_priority",
                "adjustment": 0.30 * weight,
            }
        if bool(summary.get("sleep_control", False)):
            return {"reason": "early_survival_allows_sleep_control", "adjustment": 0.35 * weight}
        if bool(summary.get("status_control", False)) or bool(summary.get("speed_control", False)):
            return {"reason": "early_survival_allows_tempo_control", "adjustment": 0.15 * weight}
        if bool(summary.get("support", False)) or float(summary.get("bp", 0.0) or 0.0) <= 0.0:
            return {
                "reason": "early_survival_penalizes_passive_action",
                "adjustment": -0.90 * weight,
            }
        accuracy = float(summary.get("accuracy", 1.0) or 1.0)
        if accuracy < 0.90 and float(summary.get("ko_sum", 0.0) or 0.0) < 0.75:
            return {
                "reason": "early_survival_penalizes_unreliable_damage_race",
                "adjustment": -0.70 * weight,
            }
        return {"reason": "early_survival_neutral_attack", "adjustment": 0.0}

    def _arbitrate_structured_team_plan(
        self,
        battle: MultiBattle,
        own_message: Dict[str, Any],
        partner_messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        for message in [own_message] + list(partner_messages or []):
            if not isinstance(message, dict):
                continue
            source = str(message.get("agent", "unknown"))
            for proposal in message.get("proposals", []) or []:
                if isinstance(proposal, dict):
                    item = dict(proposal)
                    item["source_agent"] = source
                    proposals.append(item)
        if not proposals:
            return None

        partner_damage = self._partner_best_damage_from_messages(partner_messages)
        best_score = -1e9
        best_plan: Optional[Dict[str, Any]] = None
        for proposal in proposals:
            strategy = str(proposal.get("strategy", ""))
            confidence = float(proposal.get("confidence", 0.0))
            slot = proposal.get("target_slot")
            try:
                slot = int(slot) if slot is not None else None
            except Exception:
                slot = None
            hp = float(
                proposal.get("target_hp", 1.0) if proposal.get("target_hp") is not None else 1.0
            )
            self_damage = utility_damage_ratio(proposal.get("self_damage", 0.0))
            combined = self_damage + (
                utility_damage_ratio(partner_damage.get(slot, 0.0)) if slot is not None else 0.0
            )
            priority = {
                "protect_partner_removes_threat": 1.20,
                "finish_ko": 1.05,
                "double_target_ko": 0.95,
                "threat_removal": 0.90,
                "spread_pressure": 0.45,
                "speed_control_damage": 0.40,
                "setup_redirection": 0.72,
                "terrain_control": 0.58,
                "self_activation_combo": 0.56,
                "screens_positioning": 0.50,
                "pivot_cycle": 0.38,
            }.get(strategy, 0.0)
            score = confidence + priority
            if slot is not None:
                score += 0.35 * max(0.0, 1.0 - hp)
                if combined >= max(0.92, hp):
                    score += 0.35
            if proposal.get("avoid_partner_overkill") and not proposal.get("partner_required"):
                score -= 0.15
            if (
                bool(getattr(self.config, "communication_solo_claims_enabled", False))
                and proposal.get("solo_claim")
                and not proposal.get("partner_required")
            ):
                score -= 0.20
            if (
                strategy == "protect_partner_removes_threat"
                and proposal.get("source_agent") == self.agent_name
            ):
                danger = self._predicted_self_danger(battle)
                score += 0.45 * min(1.0, danger["risk"] + danger["ko"])
            if strategy == "setup_redirection" and proposal.get("partner_required"):
                score += 0.20
            if strategy == "self_activation_combo":
                score += 0.18 * float(proposal.get("ally_damage", 1.0) <= 0.35)
            if strategy == "terrain_control" and proposal.get("field_effect") == "psychicterrain":
                score += 0.12 * float(
                    (proposal.get("terrain_synergy", {}) or {}).get(
                        "psychic_blocks_priority", False
                    )
                )
            plan = {
                "id": str(proposal.get("id", f"{strategy}:{slot}")),
                "strategy": strategy,
                "target_slot": slot,
                "target_species": proposal.get("target_species"),
                "target_hp": proposal.get("target_hp"),
                "confidence": max(0.0, min(1.0, confidence)),
                "source_agent": proposal.get("source_agent"),
                "recommended_action": proposal.get("recommended_action"),
                "partner_required": bool(proposal.get("partner_required")),
                "avoid_partner_overkill": bool(proposal.get("avoid_partner_overkill")),
                "solo_claim": bool(proposal.get("solo_claim")),
                "self_damage": float(proposal.get("self_damage", 0.0) or 0.0),
                "score": float(score),
            }
            for key_name in (
                "coordination_role",
                "partner_role",
                "field_effect",
                "boosted_type",
                "terrain_synergy",
                "positioning_goal",
                "self_risk",
                "activation_target",
                "activation_item",
                "activation_ability",
                "ally_damage",
            ):
                if key_name in proposal:
                    plan[key_name] = proposal.get(key_name)
            key = (
                score,
                str(plan["strategy"]),
                str(plan["target_slot"]),
                str(plan["source_agent"]),
                str(plan["id"]),
            )
            if best_plan is None or key > (
                best_score,
                str(best_plan["strategy"]),
                str(best_plan["target_slot"]),
                str(best_plan["source_agent"]),
                str(best_plan["id"]),
            ):
                best_score = score
                best_plan = plan
        if best_plan is not None:
            mem = self._shared_memory(battle)
            mem["team_plan"] = json_safe(best_plan)
            mem["focus_target"] = best_plan.get("target_species")
            mem["focus_target_slot"] = best_plan.get("target_slot")
            mem["focus_ttl"] = 2 if best_plan.get("target_slot") is not None else 1
        return best_plan


__all__ = ["MultiAgentDecisionGateMixin"]
