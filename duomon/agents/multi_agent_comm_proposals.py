from __future__ import annotations

from .multi_agent_context import *


class MultiAgentCommProposalMixin:
    def _strategy_proposals_from_summaries(
        self,
        battle: MultiBattle,
        summaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        threat = self.slot_agent.tactical.threat_model.analyze(battle)
        partner_potential = self._partner_best_damage_potential_by_slot(battle)
        for slot, opp in enumerate(battle.opponent_active_pokemon[:2]):
            if opp is None or is_fainted(opp):
                continue
            best = self._best_summary_for_slot(summaries, slot)
            if not best:
                continue
            hp = safe_hp_fraction(opp)
            self_damage = float((best.get("damage_by_slot", {}) or {}).get(slot, 0.0))
            combined = self_damage + partner_potential.get(slot, 0.0)
            if hp <= 0.35 and self_damage > 0.0:
                proposals.append(
                    self._proposal(
                        battle,
                        "finish_ko",
                        slot,
                        opp,
                        best,
                        confidence=0.76
                        + 0.16 * min(1.0, self_damage)
                        + 0.08 * float(combined >= 1.0),
                        need_partner=combined >= 1.0 and self_damage < 0.98,
                        rationale="remove_low_hp_target_to_deny_action",
                    )
                )
            elif combined >= 1.0 and self_damage >= 0.25:
                proposals.append(
                    self._proposal(
                        battle,
                        "double_target_ko",
                        slot,
                        opp,
                        best,
                        confidence=0.62 + 0.20 * min(1.0, combined) + 0.10 * max(0.0, 1.0 - hp),
                        need_partner=self_damage < 0.98,
                        rationale="combine_damage_for_same_turn_ko",
                    )
                )
            if safe_species(opp) == threat.global_target and self_damage > 0.0:
                proposals.append(
                    self._proposal(
                        battle,
                        "threat_removal",
                        slot,
                        opp,
                        best,
                        confidence=0.58
                        + 0.20 * min(1.0, threat.global_pressure)
                        + 0.12 * min(1.0, self_damage),
                        need_partner=combined >= 1.0 and self_damage < 0.98,
                        rationale="remove_highest_predicted_threat",
                    )
                )

        protect = next((s for s in summaries if s.get("protect")), None)
        danger = threat.threatening_opp_ref.get(0)
        danger_slot = self._opponent_slot_index(battle, danger) if danger is not None else None
        predicted_danger = self._predicted_self_danger(battle)
        if danger_slot is None and predicted_danger["targeted"] > 0.0:
            best_response = max(
                self._predict_opponent_responses(battle),
                key=lambda r: float((r.get("damage_by_slot", {}) or {}).get(0, 0.0))
                + float((r.get("ko_by_slot", {}) or {}).get(0, 0.0)),
                default=None,
            )
            danger_slot = best_response.get("opp_slot") if isinstance(best_response, dict) else None
            danger = (
                battle.opponent_active_pokemon[danger_slot]
                if isinstance(danger_slot, int)
                and danger_slot < len(battle.opponent_active_pokemon)
                else danger
            )
        if (
            protect
            and danger_slot is not None
            and max(
                threat.slot_threat.get(0, 0.0), predicted_danger["risk"], predicted_danger["ko"]
            )
            >= 0.48
        ):
            proposals.append(
                self._proposal(
                    battle,
                    "protect_partner_removes_threat",
                    danger_slot,
                    danger,
                    protect,
                    confidence=0.66
                    + 0.22
                    * min(
                        1.0,
                        max(
                            threat.slot_threat.get(0, 0.0),
                            predicted_danger["risk"] + predicted_danger["ko"],
                        ),
                    ),
                    need_partner=True,
                    rationale="protect_threatened_slot_while_partner_attacks",
                )
            )

        speed = next(
            (
                s
                for s in summaries
                if s.get("speed_control") and float(s.get("damage_sum", 0.0)) > 0.15
            ),
            None,
        )
        if speed and self._speed_control_communication_needed(battle):
            proposals.append(
                self._proposal(
                    battle,
                    "speed_control_damage",
                    speed.get("target_slot"),
                    None,
                    speed,
                    confidence=0.56 + 0.12 * min(1.0, float(speed.get("damage_sum", 0.0))),
                    need_partner=False,
                    rationale="slow_opponents_while_dealing_damage",
                )
            )

        spread = next(
            (s for s in summaries if s.get("spread") and float(s.get("damage_sum", 0.0)) >= 0.70),
            None,
        )
        if spread:
            proposals.append(
                self._proposal(
                    battle,
                    "spread_pressure",
                    spread.get("target_slot"),
                    None,
                    spread,
                    confidence=0.54 + 0.14 * min(1.5, float(spread.get("damage_sum", 0.0))),
                    need_partner=False,
                    rationale="pressure_both_opponents_without_overcommitting",
                )
            )

        terrain_profile = self._terrain_synergy_profile(battle)
        terrain = next((s for s in summaries if s.get("terrain")), None)
        if terrain:
            effect = str(terrain.get("field_effect") or "")
            boosted_type = TERRAIN_BOOST_TYPES.get(effect)
            has_boost_user = bool(
                boosted_type
                and boosted_type in (terrain_profile.get("boosted_types_available") or [])
            )
            psychic_priority_block = bool(
                effect == "psychicterrain" and terrain_profile.get("psychic_blocks_priority")
            )
            terrain_confidence = (
                0.46 + 0.16 * float(has_boost_user) + 0.18 * float(psychic_priority_block)
            )
            if effect == "remove_terrain":
                terrain_confidence = 0.48 + 0.12 * float(bool(_battle_field_names(battle)))
            if terrain_confidence >= 0.48:
                proposals.append(
                    self._proposal(
                        battle,
                        "terrain_control",
                        terrain.get("target_slot"),
                        None,
                        terrain,
                        confidence=terrain_confidence,
                        need_partner=bool(has_boost_user or psychic_priority_block),
                        rationale="coordinate_field_effect_with_team_damage_or_priority_control",
                        extra={
                            "field_effect": effect,
                            "coordination_role": "field_setter",
                            "boosted_type": boosted_type,
                            "terrain_synergy": terrain_profile,
                        },
                    )
                )

        pivot = next((s for s in summaries if s.get("pivot")), None)
        if pivot:
            self_danger = self._predicted_self_danger(battle)
            pivot_confidence = (
                0.46
                + 0.18 * min(1.0, self_danger["risk"])
                + 0.08
                * float(
                    safe_hp_fraction(battle.active_pokemon[0] if battle.active_pokemon else None)
                    <= 0.55
                )
            )
            proposals.append(
                self._proposal(
                    battle,
                    "pivot_cycle",
                    pivot.get("target_slot"),
                    None,
                    pivot,
                    confidence=pivot_confidence,
                    need_partner=False,
                    rationale="reposition_while_preserving_tempo_and_cycling_support",
                    extra={
                        "coordination_role": "pivot",
                        "positioning_goal": "cycle_debuffs_or_restore_matchup",
                        "self_risk": self_danger,
                    },
                )
            )

        activation = next((s for s in summaries if s.get("ally_activation")), None)
        if activation:
            partner = battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
            ally_damage = float(activation.get("ally_damage", 0.0))
            if partner is not None and ally_damage <= 0.45:
                proposals.append(
                    self._proposal(
                        battle,
                        "self_activation_combo",
                        None,
                        partner,
                        activation,
                        confidence=0.54
                        + 0.18 * float(_mon_item_name(partner) in ALLY_ACTIVATION_ITEMS)
                        + 0.10 * float(_mon_ability_name(partner) in ALLY_ACTIVATION_ABILITIES),
                        need_partner=True,
                        rationale="activate_partner_item_or_ability_with_controlled_ally_hit",
                        extra={
                            "coordination_role": "activator",
                            "activation_target": safe_species(partner),
                            "activation_item": _mon_item_name(partner),
                            "activation_ability": _mon_ability_name(partner),
                            "ally_damage": ally_damage,
                        },
                    )
                )

        redirection = next((s for s in summaries if s.get("redirection")), None)
        setup = next((s for s in summaries if s.get("setup")), None)
        partner = battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
        partner_has_setup = self._mon_has_any_move(partner, SETUP_MOVES)
        partner_has_redirect = self._mon_has_any_move(partner, REDIRECTION_MOVES | FAKE_OUT_MOVES)
        if redirection and partner_has_setup:
            proposals.append(
                self._proposal(
                    battle,
                    "setup_redirection",
                    None,
                    partner,
                    redirection,
                    confidence=0.58
                    + 0.16
                    * safe_hp_fraction(battle.active_pokemon[0] if battle.active_pokemon else None),
                    need_partner=True,
                    rationale="redirect_or_disrupt_attacks_while_partner_sets_up",
                    extra={"coordination_role": "protector", "partner_role": "setup_sweeper"},
                )
            )
        if setup and partner_has_redirect:
            proposals.append(
                self._proposal(
                    battle,
                    "setup_redirection",
                    None,
                    battle.active_pokemon[0] if battle.active_pokemon else None,
                    setup,
                    confidence=0.56
                    + 0.12
                    * safe_hp_fraction(battle.active_pokemon[0] if battle.active_pokemon else None),
                    need_partner=True,
                    rationale="request_partner_redirection_or_fake_out_for_safe_setup",
                    extra={"coordination_role": "setup_sweeper", "partner_role": "protector"},
                )
            )

        screen = next((s for s in summaries if s.get("screen")), None)
        if screen:
            ally_hp = [
                safe_hp_fraction(mon) for mon in active_alive_mons(battle.active_pokemon[:2])
            ]
            proposals.append(
                self._proposal(
                    battle,
                    "screens_positioning",
                    None,
                    None,
                    screen,
                    confidence=0.50 + 0.14 * float(bool(ally_hp and min(ally_hp) >= 0.45)),
                    need_partner=False,
                    rationale="set_team_damage_reduction_before_setup_or_trade_sequence",
                    extra={
                        "coordination_role": "screen_setter",
                        "field_effect": screen.get("field_effect"),
                    },
                )
            )

        proposals.sort(key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
        return proposals[:6]

    def _proposal(
        self,
        battle: MultiBattle,
        strategy: str,
        target_slot: Any,
        target_mon: Any,
        summary: Dict[str, Any],
        confidence: float,
        need_partner: bool,
        rationale: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        slot = int(target_slot) if target_slot is not None else None
        damage_by_slot = {
            int(k): utility_damage_ratio(v)
            for k, v in (summary.get("damage_by_slot", {}) or {}).items()
        }
        ko_by_slot = {int(k): float(v) for k, v in (summary.get("ko_by_slot", {}) or {}).items()}
        target_hp = self._target_hp_for_slot(battle, slot) if slot is not None else None
        self_damage = (
            float(damage_by_slot.get(slot, 0.0))
            if slot is not None
            else utility_damage_sum((summary.get("damage_by_slot", {}) or {}).values())
        )
        solo_claim = bool(slot is not None and self_damage >= 0.98 and not need_partner)
        packet = {
            "id": f"{self.agent_name}:{int(safe_getattr(battle, 'turn', 0) or 0)}:{strategy}:{slot}",
            "speech_act": "propose",
            "strategy": strategy,
            "target_slot": slot,
            "target_species": safe_species(target_mon)
            if target_mon is not None
            else summary.get("target"),
            "target_hp": target_hp,
            "recommended_action": summary.get("signature"),
            "recommended_label": summary.get("label"),
            "recommended_move": summary.get("move_id"),
            "self_damage_by_slot": damage_by_slot,
            "self_ko_by_slot": ko_by_slot,
            "self_damage": self_damage,
            "partner_required": bool(need_partner),
            "solo_ko": bool(slot is not None and self_damage >= 0.98),
            "solo_claim": solo_claim,
            "communication_role": "claim"
            if solo_claim
            else ("request" if need_partner else "inform"),
            "avoid_partner_overkill": bool(
                slot is not None
                and self_damage >= 1.15
                and (target_hp is None or target_hp <= 0.50)
            ),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "rationale": rationale,
            "vector": self._proposal_vector(
                strategy, slot, self_damage, target_hp, confidence, need_partner
            ),
        }
        if extra:
            packet.update(json_safe(extra))
        return packet

    @staticmethod
    def _proposal_vector(
        strategy: str,
        slot: Optional[int],
        damage: float,
        hp: Optional[float],
        confidence: float,
        need_partner: bool,
    ) -> List[float]:
        return [
            1.0 if strategy == "finish_ko" else 0.0,
            1.0 if strategy == "double_target_ko" else 0.0,
            1.0 if strategy == "protect_partner_removes_threat" else 0.0,
            1.0 if strategy == "speed_control_damage" else 0.0,
            1.0 if strategy == "spread_pressure" else 0.0,
            1.0 if strategy == "terrain_control" else 0.0,
            1.0 if strategy == "pivot_cycle" else 0.0,
            1.0 if strategy == "self_activation_combo" else 0.0,
            1.0 if strategy == "setup_redirection" else 0.0,
            1.0 if strategy == "screens_positioning" else 0.0,
            float(slot if slot is not None else -1) / 2.0,
            min(2.0, float(damage)) / 2.0,
            float(hp if hp is not None else 1.0),
            max(0.0, min(1.0, float(confidence))),
            1.0 if need_partner else 0.0,
        ]

    def _communication_state_vector(
        self,
        battle: MultiBattle,
        summaries: List[Dict[str, Any]],
        proposals: List[Dict[str, Any]],
    ) -> List[float]:
        opp_hps = [
            safe_hp_fraction(opp) for opp in active_alive_mons(battle.opponent_active_pokemon[:2])
        ]
        best_damage = max((float(s.get("damage_sum", 0.0)) for s in summaries), default=0.0)
        return [
            min(1.0, float(int(safe_getattr(battle, "turn", 0) or 0)) / 20.0),
            min(1.0, min(opp_hps) if opp_hps else 1.0),
            min(2.0, best_damage) / 2.0,
            float(any(p.get("strategy") == "finish_ko" for p in proposals)),
            float(any(p.get("strategy") == "double_target_ko" for p in proposals)),
            float(any(p.get("strategy") == "protect_partner_removes_threat" for p in proposals)),
            float(any(p.get("strategy") == "terrain_control" for p in proposals)),
            float(any(p.get("strategy") == "pivot_cycle" for p in proposals)),
            float(any(p.get("strategy") == "self_activation_combo" for p in proposals)),
            float(any(p.get("strategy") == "setup_redirection" for p in proposals)),
            float(any(p.get("strategy") == "screens_positioning" for p in proposals)),
            max((float(p.get("confidence", 0.0)) for p in proposals), default=0.0),
        ]

    @staticmethod
    def _best_summary_for_slot(
        summaries: List[Dict[str, Any]], slot: int
    ) -> Optional[Dict[str, Any]]:
        candidates = [s for s in summaries if slot in (s.get("damage_by_slot", {}) or {})]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda s: (
                float((s.get("damage_by_slot", {}) or {}).get(slot, 0.0)),
                float(s.get("score", 0.0)),
            ),
        )

    def _partner_best_damage_potential_by_slot(self, battle: MultiBattle) -> Dict[int, float]:
        partner = battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
        if partner is None:
            return {}
        result: Dict[int, float] = {}
        moves = safe_getattr(partner, "moves", {}) or {}
        for move in moves.values() if isinstance(moves, dict) else moves:
            if move_base_power(move) <= 0:
                continue
            if is_spread_move(move):
                for slot, opp in enumerate(battle.opponent_active_pokemon[:2]):
                    if opp is not None and not is_fainted(opp):
                        result[slot] = max(
                            result.get(slot, 0.0),
                            _advanced_damage_ratio(battle, move, partner, opp, spread=True),
                        )
            else:
                for slot, opp in enumerate(battle.opponent_active_pokemon[:2]):
                    if opp is not None and not is_fainted(opp) and damage_multiplier(opp, move) > 0:
                        result[slot] = max(
                            result.get(slot, 0.0),
                            _advanced_damage_ratio(battle, move, partner, opp),
                        )
        return result

    @staticmethod
    def _speed_control_communication_needed(battle: MultiBattle) -> bool:
        mine = [safe_speed(mon) for mon in active_alive_mons(battle.active_pokemon[:2])]
        opp = [safe_speed(mon) for mon in active_alive_mons(battle.opponent_active_pokemon[:2])]
        return bool(mine and opp and (sum(mine) / len(mine) + 10.0 < sum(opp) / len(opp)))


__all__ = ["MultiAgentCommProposalMixin"]
