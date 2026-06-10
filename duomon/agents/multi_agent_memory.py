from __future__ import annotations

from .multi_agent_context import *


class MultiAgentMemoryMixin:
    def _shared_memory(self, battle: MultiBattle) -> Dict[str, Any]:
        return _get_multi_short_memory(battle, _multi_side_role(battle))

    def _refresh_shared_memory_from_battle(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
    ) -> None:
        mem = self._shared_memory(battle)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        for idx, mon in enumerate(battle.active_pokemon[:2]):
            if mon is None:
                continue
            species = safe_species(mon)
            fact = mem.setdefault("ally_facts", {}).setdefault(species, {"species": species})
            fact.update(self._pokemon_public_fact(mon))
            fact["slot"] = idx
            fact["source"] = "local_request" if idx == 0 else "local_ally_view"
            fact["last_seen_turn"] = turn
        for idx, mon in enumerate(battle.opponent_active_pokemon[:2]):
            if mon is None:
                continue
            species = safe_species(mon)
            fact = mem.setdefault("enemy_facts", {}).setdefault(species, {"species": species})
            public = self._pokemon_public_fact(mon)
            revealed = fact.get("revealed_moves", {})
            fact.update(public)
            if revealed:
                fact["revealed_moves"] = revealed
            fact["slot"] = idx
            fact["last_seen_turn"] = turn
        mem["predictions"] = {
            str(item.get("opp_slot", idx)): self._public_response_summary(item)
            for idx, item in enumerate(self._predict_opponent_responses(battle))
        }
        mem["top_local_candidates"] = [
            self._public_candidate_summary(battle, score, action) for score, action in scored[:5]
        ]
        mem["updated_at"] = time.time()

    def _merge_partner_messages_into_memory(
        self,
        battle: MultiBattle,
        partner_messages: List[Dict[str, Any]],
    ) -> None:
        if not partner_messages:
            return
        mem = self._shared_memory(battle)
        for message in partner_messages:
            if not isinstance(message, dict):
                continue
            agent = str(message.get("agent", "partner"))
            mem.setdefault("partner_reports", {})[agent] = {
                "turn": int(message.get("turn", 0) or 0),
                "facts": json_safe(message.get("facts", {})),
                "capabilities": json_safe(message.get("capabilities", {})),
                "proposals": json_safe(message.get("proposals", [])),
            }
            facts = message.get("facts", {}) or {}
            for fact in [facts.get("self_active"), facts.get("partner_active")]:
                if isinstance(fact, dict) and fact.get("known") and fact.get("species"):
                    species = str(fact.get("species"))
                    stored = mem.setdefault("ally_facts", {}).setdefault(
                        species, {"species": species}
                    )
                    stored.update(json_safe(fact))
                    stored["source"] = f"partner_report:{agent}"
            for fact in facts.get("opponent_active", []) or []:
                if isinstance(fact, dict) and fact.get("known") and fact.get("species"):
                    species = str(fact.get("species"))
                    stored = mem.setdefault("enemy_facts", {}).setdefault(
                        species, {"species": species}
                    )
                    revealed = stored.get("revealed_moves", {})
                    stored.update(json_safe(fact))
                    if revealed:
                        stored["revealed_moves"] = revealed
                    stored["source"] = f"partner_report:{agent}"

    @staticmethod
    def _public_response_summary(response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "opp_slot": response.get("opp_slot"),
            "opp_species": response.get("opp_species"),
            "move_id": response.get("move_id"),
            "priority": float(response.get("priority", 0.0)),
            "attacker_speed": float(response.get("attacker_speed", 0.0)),
            "target_slots": [
                int(s) for s in response.get("target_slots", []) if isinstance(s, int)
            ],
            "damage_by_slot": {
                int(k): float(v) for k, v in (response.get("damage_by_slot", {}) or {}).items()
            },
            "ko_by_slot": {
                int(k): float(v) for k, v in (response.get("ko_by_slot", {}) or {}).items()
            },
            "weight": float(response.get("weight", 1.0)),
        }

    def _communication_memory_snapshot(self, battle: MultiBattle) -> Dict[str, Any]:
        mem = self._shared_memory(battle)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        enemy_facts = {}
        for species, fact in (mem.get("enemy_facts", {}) or {}).items():
            if int(fact.get("last_seen_turn", turn) or turn) < turn - 8:
                continue
            enemy_facts[str(species)] = json_safe(fact)
        return {
            "focus_target": mem.get("focus_target"),
            "focus_target_slot": mem.get("focus_target_slot"),
            "focus_ttl": int(mem.get("focus_ttl", 0) or 0),
            "enemy_facts": enemy_facts,
            "predictions": json_safe(mem.get("predictions", {})),
            "recent_enemy_events": json_safe((mem.get("enemy_events", []) or [])[-6:]),
            "partner_reports": json_safe(mem.get("partner_reports", {})),
        }

    def _predicted_self_danger(self, battle: MultiBattle) -> Dict[str, float]:
        damage = 0.0
        ko = 0.0
        targeted = 0.0
        responses = self._predict_opponent_responses(battle)
        for response in responses:
            target_slots = [int(s) for s in response.get("target_slots", []) if isinstance(s, int)]
            if 0 not in target_slots:
                continue
            weight = float(response.get("weight", 1.0))
            targeted += weight
            damage += weight * utility_damage_ratio(
                (response.get("damage_by_slot", {}) or {}).get(0, 0.0), cap=1.25
            )
            ko += weight * float((response.get("ko_by_slot", {}) or {}).get(0, 0.0))
        unknown_floor, _unknown_slot = self._generic_unknown_threat_floor(battle, 0)
        unknown_weight = float(getattr(self.config, "unknown_threat_risk_floor_weight", 1.0) or 1.0)
        if unknown_weight > 0.0:
            damage = max(float(damage), float(unknown_floor) * unknown_weight)
            if targeted <= 0.0 and unknown_floor >= 0.25:
                targeted = 0.45
        return {"targeted": targeted, "damage": damage, "ko": ko, "risk": max(damage, ko)}

    def _decision_call_key(self, battle: MultiBattle) -> str:
        return f"{safe_getattr(battle, 'battle_tag', 'unknown')}:{int(safe_getattr(battle, 'turn', 0) or 0)}"

    @staticmethod
    def _generic_unknown_threat_floor(
        battle: MultiBattle, slot: int
    ) -> Tuple[float, Optional[int]]:
        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        if active is None or is_fainted(active):
            return 0.0, None
        hp = max(0.08, safe_hp_fraction(active))
        bulk = (
            float(base_stat(active, "hp", 90))
            + 0.70 * float(base_stat(active, "def", 90))
            + 0.70 * float(base_stat(active, "spd", 90))
        ) / 2.40
        frailty = max(0.0, (112.0 - bulk) / 120.0)
        own_speed = float(safe_speed(active))
        best_floor = 0.0
        best_slot: Optional[int] = None
        total = 0.0
        visible = [
            (idx, opp)
            for idx, opp in enumerate(battle.opponent_active_pokemon[:2])
            if opp is not None and not is_fainted(opp)
        ]
        if not visible and int(safe_getattr(battle, "turn", 0) or 0) <= 1:
            blind = min(0.48, 0.24 + 0.22 * frailty + 0.20 * max(0.0, 0.55 - hp))
            return float(blind), None
        for opp_slot, opp in visible:
            offensive = max(float(base_stat(opp, "atk", 95)), float(base_stat(opp, "spa", 95)))
            speed_gap = float(safe_speed(opp)) - own_speed
            speed_pressure = 0.16 if speed_gap > 8.0 else 0.04 if speed_gap > -8.0 else 0.0
            type_pressure = 0.0
            for opp_type in _mon_type_names(opp):
                try:
                    type_chart = GenData.from_format("gen9randombattle").type_chart
                    move_type = str(opp_type).upper()
                    mult = 1.0
                    for active_type in _mon_type_names(active):
                        mult *= float(
                            type_chart.get(str(active_type).upper(), {}).get(move_type, 1.0)
                        )
                    type_pressure = max(type_pressure, 0.12 * max(0.0, mult - 1.0))
                except Exception:
                    continue
            floor = (
                0.13
                + 0.0020 * min(180.0, offensive)
                + speed_pressure
                + type_pressure
                + 0.22 * frailty
            )
            floor *= 0.82 + 0.55 * max(0.0, 0.70 - hp)
            floor = max(0.0, min(0.62, floor))
            total += floor
            if floor > best_floor:
                best_floor = floor
                best_slot = int(opp_slot)
        return float(min(0.78, max(best_floor, 0.62 * total))), best_slot

    def _decision_risk_context(self, battle: MultiBattle) -> Dict[str, Any]:
        threat = self.slot_agent.tactical.threat_model.analyze(battle)
        active = battle.active_pokemon[0] if battle.active_pokemon else None
        partner_damage = self._partner_best_damage_potential_by_slot(battle)
        responses = self._predict_opponent_responses(battle)
        targeted = 0.0
        damage = 0.0
        ko = 0.0
        for response in responses:
            target_slots = [
                int(slot) for slot in response.get("target_slots", []) if isinstance(slot, int)
            ]
            if 0 not in target_slots:
                continue
            weight = float(response.get("weight", 1.0) or 1.0)
            targeted += weight
            damage += weight * utility_damage_ratio(
                (response.get("damage_by_slot", {}) or {}).get(0, 0.0), cap=1.25
            )
            ko += weight * float((response.get("ko_by_slot", {}) or {}).get(0, 0.0))
        primary_response = max(
            (
                response
                for response in responses
                if 0
                in [int(slot) for slot in response.get("target_slots", []) if isinstance(slot, int)]
            ),
            key=lambda response: (
                utility_damage_ratio(
                    (response.get("damage_by_slot", {}) or {}).get(0, 0.0), cap=1.25
                )
                + float((response.get("ko_by_slot", {}) or {}).get(0, 0.0))
                + 0.25 * float(response.get("weight", 1.0) or 1.0)
            ),
            default=None,
        )
        primary_opp_slot = (
            primary_response.get("opp_slot") if isinstance(primary_response, dict) else None
        )
        if not isinstance(primary_opp_slot, int):
            danger_ref = threat.threatening_opp_ref.get(0)
            primary_opp_slot = (
                self._opponent_slot_index(battle, danger_ref) if danger_ref is not None else None
            )
        active_hp = safe_hp_fraction(active)
        slot_risk = max(
            float(max(damage, ko)),
            float(ko),
            float(threat.slot_threat.get(0, 0.0)),
            float(threat.slot_ko_risk.get(0, 0.0)),
            float(threat.slot_combined_ko.get(0, 0.0)),
            0.65 * float(threat.slot_2hko_risk.get(0, 0.0)),
        )
        unknown_floor, unknown_primary_slot = self._generic_unknown_threat_floor(battle, 0)
        unknown_weight = float(getattr(self.config, "unknown_threat_risk_floor_weight", 1.0) or 1.0)
        if unknown_weight > 0.0:
            slot_risk = max(float(slot_risk), float(unknown_floor) * unknown_weight)
            if targeted <= 0.0 and unknown_floor >= 0.25:
                targeted = max(targeted, 0.45)
            if not isinstance(primary_opp_slot, int) and isinstance(unknown_primary_slot, int):
                primary_opp_slot = unknown_primary_slot
        own_speed = safe_speed(active)
        opp_speeds = [
            safe_speed(opp) for opp in active_alive_mons(battle.opponent_active_pokemon[:2])
        ]
        speed_disadvantage = bool(opp_speeds and own_speed + 8.0 < max(opp_speeds))
        if not opp_speeds and unknown_floor >= 0.38 and active is not None:
            speed_disadvantage = base_stat(active, "spe", 85) < 90
        protect_available = any(
            move_id(move) in PROTECT_MOVES
            for move in normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
        )
        return {
            "risk": float(slot_risk),
            "unknown_threat_floor": float(unknown_floor),
            "targeted": float(targeted),
            "damage": float(damage),
            "ko": float(ko),
            "hp": float(active_hp),
            "primary_opp_slot": primary_opp_slot,
            "primary_response": primary_response,
            "primary_response_move": str(primary_response.get("move_id", ""))
            if isinstance(primary_response, dict)
            else "",
            "partner_damage_by_slot": {
                int(slot): float(value) for slot, value in partner_damage.items()
            },
            "speed_disadvantage": speed_disadvantage,
            "protect_available": protect_available,
            "turn": int(safe_getattr(battle, "turn", 0) or 0),
        }

    def _apply_decision_gates(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
    ) -> List[Tuple[float, SlotAction]]:
        if not scored or not bool(getattr(self.config, "decision_gate_enabled", True)):
            return scored

        context = self._decision_risk_context(battle)
        candidate_rows: List[Tuple[float, SlotAction, Dict[str, Any]]] = []
        protect_candidate_available = False
        for score, action in scored:
            summary = self._candidate_summary(battle, float(score), action)
            if bool(summary.get("protect", False)):
                protect_candidate_available = True
            candidate_rows.append((score, action, summary))
        context["protect_candidate_available"] = protect_candidate_available

        adjusted: List[Tuple[float, SlotAction]] = []
        diagnostics: List[Dict[str, Any]] = []
        for score, action, summary in candidate_rows:
            tera = self._tera_gate_evaluation(battle, action, summary, context)
            defense = self._defensive_gate_evaluation(battle, action, summary, context)
            accuracy = self._accuracy_gate_evaluation(battle, action, summary, context)
            survival = self._early_survival_gate_evaluation(battle, action, summary, context)
            total_adjustment = (
                float(tera.get("adjustment", 0.0))
                + float(defense.get("adjustment", 0.0))
                + float(accuracy.get("adjustment", 0.0))
                + float(survival.get("adjustment", 0.0))
            )
            new_score = float(score) + total_adjustment
            adjusted.append((new_score, action))
            diagnostics.append(
                {
                    "signature": self._order_signature(action.order),
                    "label": action.label,
                    "move_id": str(summary.get("move_id", "")),
                    "protect": bool(summary.get("protect", False)),
                    "tera": bool(summary.get("tera", False)),
                    "accuracy": float(summary.get("accuracy", 1.0) or 1.0),
                    "base_score": float(score),
                    "adjustment": float(total_adjustment),
                    "adjusted_score": float(new_score),
                    "tera_gate": tera,
                    "defensive_gate": defense,
                    "accuracy_gate": accuracy,
                    "survival_gate": survival,
                }
            )

        adjusted.sort(key=lambda item: item[0], reverse=True)
        diagnostics.sort(key=lambda item: float(item.get("adjusted_score", 0.0)), reverse=True)
        top_k = max(1, int(getattr(self.config, "decision_diagnostics_top_k", 8) or 8))
        key = self._decision_call_key(battle)
        self._decision_gate_raw_context[key] = context
        self._decision_gate_diagnostics[key] = {
            "risk_context": json_safe(context),
            "top_candidates": json_safe(diagnostics[:top_k]),
            "rejected_tera_candidates": [
                json_safe(item)
                for item in diagnostics
                if bool(item.get("tera"))
                and not bool((item.get("tera_gate") or {}).get("allowed", True))
            ][:top_k],
        }
        if len(self._decision_gate_raw_context) > 8192:
            for old_key in list(self._decision_gate_raw_context.keys())[:1024]:
                self._decision_gate_raw_context.pop(old_key, None)
                self._decision_gate_diagnostics.pop(old_key, None)
        return adjusted

    def _final_decision_diagnostics(
        self,
        battle: MultiBattle,
        selected_action: SlotAction,
        selected_score: float,
    ) -> Dict[str, Any]:
        if not bool(getattr(self.config, "decision_gate_enabled", True)):
            return {}
        key = self._decision_call_key(battle)
        context = self._decision_gate_raw_context.get(key)
        if context is None:
            context = self._decision_risk_context(battle)
        summary = self._candidate_summary(battle, float(selected_score), selected_action)
        diagnostics = {
            "selected_signature": self._order_signature(selected_action.order),
            "selected_label": selected_action.label,
            "selected_move_id": str(summary.get("move_id", "")),
            "selected_score": float(selected_score),
            "risk_context": json_safe(context),
            "selected_gates": {
                "tera": self._tera_gate_evaluation(battle, selected_action, summary, context),
                "defensive": self._defensive_gate_evaluation(
                    battle, selected_action, summary, context
                ),
                "accuracy": self._accuracy_gate_evaluation(
                    battle, selected_action, summary, context
                ),
                "survival": self._early_survival_gate_evaluation(
                    battle, selected_action, summary, context
                ),
            },
        }
        cached = self._decision_gate_diagnostics.get(key)
        if cached:
            diagnostics["candidate_gate_summary"] = cached
        return diagnostics


__all__ = ["MultiAgentMemoryMixin"]
