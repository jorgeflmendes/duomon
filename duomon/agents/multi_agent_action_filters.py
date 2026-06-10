from __future__ import annotations

from .multi_agent_context import *


class MultiAgentActionFilterMixin:
    def _progress_filtered_actions(
        self,
        battle: MultiBattle,
        actions: List[SlotAction],
    ) -> List[SlotAction]:
        damaging = [
            action
            for action in actions
            if action.kind == "move"
            and action.move is not None
            and move_base_power(action.move) > 0
        ]
        non_immune_damage = [
            action for action in damaging if not known_target_is_immune(action, battle)
        ]
        if non_immune_damage:
            actions = [action for action in actions if not known_target_is_immune(action, battle)]
            damaging = non_immune_damage
        elif damaging and active_alive_mons(battle.opponent_active_pokemon):



            non_immune_actions = [
                action for action in actions if not known_target_is_immune(action, battle)
            ]
            switches = [
                action
                for action in non_immune_actions
                if action.kind == "switch" and action.switch is not None
            ]
            if switches:
                return switches
            progress = [
                action
                for action in non_immune_actions
                if not self._is_low_progress_move(action)
                and move_id(action.move) not in HAZARD_MOVES
            ]
            if progress:
                return progress
            damaging = []
            actions = non_immune_actions or actions
        if not damaging:
            blind_known = self._initial_blind_offense_from_known_moves(battle)
            if blind_known:
                return blind_known
            switches = [
                action
                for action in actions
                if action.kind == "switch" and action.switch is not None
            ]
            if switches and active_alive_mons(battle.opponent_active_pokemon):
                return switches
            progress = [
                action
                for action in actions
                if not self._is_low_progress_move(action)
                and move_id(action.move) not in HAZARD_MOVES
            ]
            return progress or actions
        if not active_alive_mons(battle.opponent_active_pokemon):
            return damaging
        active = battle.active_pokemon[0] if battle.active_pokemon else None
        best_damage = 0.0
        for action in damaging:
            if action.move is None:
                continue
            if is_spread_move(action.move):
                best_damage = max(
                    best_damage,
                    sum(
                        _advanced_damage_ratio(battle, action.move, active, opp, spread=True)
                        for opp in active_alive_mons(battle.opponent_active_pokemon)
                    ),
                )
            elif action.target is not None and _target_is_opponent(battle, action.target):
                best_damage = max(
                    best_damage, _advanced_damage_ratio(battle, action.move, active, action.target)
                )
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        mem = self._memory(battle)
        threat = self.slot_agent.tactical.threat_model.analyze(battle)
        filtered: List[SlotAction] = []
        for action in actions:
            mid = move_id(action.move) if action.move is not None else ""
            if action.kind == "move" and action.move is not None:
                if (
                    mid in RELIABLE_TEMPO_SPEED_MOVES
                    and mem["used_moves_by_slot"][0].get(mid, 0) >= 2
                    and best_damage >= 0.25
                ):
                    continue
                if mid in HAZARD_MOVES:
                    continue
                if mid in RECOVERY_MOVES and not (
                    safe_hp_fraction(active) < 0.35 and best_damage < 0.25
                ):
                    continue
                if mid in PROTECT_MOVES and mem["last_protect_turn_by_slot"].get(0) == turn - 1:
                    continue
                if mid == "trickroom":
                    ally_speeds = [
                        float(safe_speed(mon)) for mon in active_alive_mons(battle.active_pokemon)
                    ]
                    opp_speeds = [
                        float(safe_speed(mon))
                        for mon in active_alive_mons(battle.opponent_active_pokemon)
                    ]
                    slower_board = bool(
                        ally_speeds
                        and opp_speeds
                        and sum(ally_speeds) / len(ally_speeds) + 12
                        < sum(opp_speeds) / len(opp_speeds)
                    )
                    if not (turn <= 2 and slower_board and best_damage < 0.45):
                        continue
            if not self._is_low_progress_move(action):
                filtered.append(action)
                continue
            if self._should_preserve_macro_support_action(battle, action, threat, best_damage):
                filtered.append(action)
                continue
            if mid in PROTECT_MOVES:
                own_threat = max(threat.slot_threat.get(0, 0.0), threat.slot_ko_risk.get(0, 0.0))
                combined_ko = threat.slot_combined_ko.get(0, 0.0)
                two_hko = threat.slot_2hko_risk.get(0, 0.0)
                ko_risk = max(threat.slot_ko_risk.get(0, 0.0), combined_ko, 0.65 * two_hko)
                hp_frac = safe_hp_fraction(active) if active is not None else 1.0
                last_protect = mem["last_protect_turn_by_slot"].get(0, -10)
                used_recently = (turn - int(last_protect or -10)) <= 1
                partner_can_punish = 0.0
                danger = threat.threatening_opp_ref.get(0)
                if danger is not None:
                    partner_can_punish = TacticalKnowledgeEvaluator._best_slot_damage_into_target(
                        battle, 1, danger
                    )
                allow = not used_recently and (
                    (own_threat >= 0.60 and partner_can_punish >= 0.40)
                    or (ko_risk >= 0.40 and hp_frac < 0.70)
                    or (combined_ko >= 0.30 and hp_frac < 0.75)
                    or (two_hko >= 0.85 and hp_frac < 0.60 and partner_can_punish >= 0.25)
                    or (hp_frac < 0.40 and own_threat >= 0.35)
                )
                if (
                    not allow
                    and not used_recently
                    and bool(getattr(self.config, "protect_filter_predicted_danger_enabled", True))
                ):
                    try:
                        danger = self._predicted_self_danger(battle)
                    except Exception:
                        danger = {}
                    predicted_risk = min(
                        1.0,
                        float(danger.get("risk", 0.0) or 0.0)
                        + float(danger.get("ko", 0.0) or 0.0),
                    )
                    predicted_targeted = float(danger.get("targeted", 0.0) or 0.0)
                    allow = bool(
                        predicted_targeted > 0.0
                        and (
                            (predicted_risk >= 0.72 and hp_frac < 0.88)
                            or (predicted_risk >= 0.58 and partner_can_punish >= 0.25)
                        )
                    )
                if allow:
                    filtered.append(action)
                continue
            if mid in FAKE_OUT_MOVES and turn <= 2:
                filtered.append(action)
                continue
            if mid in RELIABLE_TEMPO_SPEED_MOVES:
                filtered.append(action)
                continue
        return filtered or damaging or actions

    def _initial_blind_offense_from_known_moves(self, battle: MultiBattle) -> List[SlotAction]:
        if int(safe_getattr(battle, "turn", 0) or 0) > 1:
            return []
        if active_alive_mons(battle.opponent_active_pokemon[:2]):
            return []
        active = battle.active_pokemon[0] if battle.active_pokemon else None
        if active is None or is_fainted(active):
            return []
        known_moves = safe_getattr(active, "moves", {}) or {}
        if isinstance(known_moves, dict):
            move_values = list(known_moves.values())
        elif isinstance(known_moves, (list, tuple, set)):
            move_values = list(known_moves)
        else:
            return []
        positions = [
            int(getattr(battle, "OPPONENT_1_POSITION", 1)),
            int(getattr(battle, "OPPONENT_2_POSITION", 2)),
        ]
        actions: List[SlotAction] = []
        seen: set[str] = set()
        for move in move_values:
            mid = move_id(move)
            if (
                not mid
                or mid in seen
                or move_base_power(move) <= 0
                or is_spread_move(move)
                or move_target_type(move) in NO_EXPLICIT_TARGETS
            ):
                continue
            seen.add(mid)
            for position in positions:
                order = self.create_order(move, move_target=int(position))
                actions.append(
                    SlotAction(
                        0,
                        "move",
                        order,
                        f"{mid}->opp{1 if position == positions[0] else 2}",
                        move=move,
                        target=None,
                        target_position=position,
                    )
                )
        if not actions:
            return []
        return self.action_generator._safe_generated_actions(battle, actions)

    def _blind_opening_target_bonus(self, battle: MultiBattle, action: SlotAction) -> float:
        if int(safe_getattr(battle, "turn", 0) or 0) > 1:
            return 0.0
        if action.kind != "move" or action.move is None:
            return 0.0
        opponents_visible = bool(active_alive_mons(battle.opponent_active_pokemon[:2]))
        mid = move_id(action.move)
        bonus = 0.0
        if mid in FAKE_OUT_MOVES:
            bonus += 3.25
        if is_spread_move(action.move) and not opponents_visible:
            bonus += 0.65
        bp = move_base_power(action.move)
        if not opponents_visible and bp > 0 and bp < 70 and not is_spread_move(action.move):
            bonus -= 0.35
        if mid in PROTECT_MOVES:
            try:
                active = (
                    battle.active_pokemon[self.slot]
                    if self.slot < len(battle.active_pokemon)
                    else None
                )
                base_hp = safe_getattr(active, "base_stats", {}) or {}
                hp_stat = base_hp.get("hp", 100) if isinstance(base_hp, dict) else 100
                if hp_stat is not None and int(hp_stat) <= 80:
                    bonus += 0.75
            except Exception:
                pass
        if bp > 0 and not is_spread_move(action.move):
            try:
                position = int(action.target_position or 0)
            except Exception:
                return bonus
            role = str(_multi_side_role(battle) or self.agent_name or "").lower()
            policy = str(getattr(self.config, "blind_opening_policy", "split") or "split").lower()
            opp1_pos = int(getattr(battle, "OPPONENT_1_POSITION", 1))
            opp2_pos = int(getattr(battle, "OPPONENT_2_POSITION", 2))
            if policy in {"focus_opp2", "opp2", "right"}:
                preferred = opp2_pos
            elif policy in {"focus_opp1", "opp1", "left", "focus"}:
                preferred = opp1_pos
            else:
                preferred = opp2_pos if "p3" in role else opp1_pos
            if position == preferred:
                bonus += 0.85
            elif position in {
                opp1_pos,
                opp2_pos,
            }:
                bonus -= 0.20
        return bonus

    def _hard_benchmark_protect_bonus(self, battle: MultiBattle, action: SlotAction) -> float:
        if self.config.benchmark_type not in {"vs_simpleheuristics", "vs_abyssal"}:
            return 0.0
        if action.kind != "move" or action.move is None:
            return 0.0
        mid = move_id(action.move)
        if mid not in PROTECT_MOVES:
            return 0.0
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        mem = self._memory(battle)
        if mem["last_protect_turn_by_slot"].get(0) == turn - 1:
            return -2.50
        danger = self._predicted_self_danger(battle)
        targeted = float(danger.get("targeted", 0.0) or 0.0)
        risk = min(1.0, float(danger.get("risk", 0.0) or 0.0) + float(danger.get("ko", 0.0) or 0.0))
        active = battle.active_pokemon[0] if battle.active_pokemon else None
        hp = safe_hp_fraction(active)
        if targeted <= 0.0:
            return -0.35 if turn <= 1 else -0.10
        partner_punish = 0.0
        best_response = max(
            self._predict_opponent_responses(battle),
            key=lambda item: float((item.get("damage_by_slot", {}) or {}).get(0, 0.0))
            + float((item.get("ko_by_slot", {}) or {}).get(0, 0.0)),
            default=None,
        )
        if isinstance(best_response, dict):
            opp_slot = best_response.get("opp_slot", None)
            if isinstance(opp_slot, int):
                opp = (
                    battle.opponent_active_pokemon[opp_slot]
                    if opp_slot < len(battle.opponent_active_pokemon)
                    else None
                )
                if opp is not None:
                    partner_punish = TacticalKnowledgeEvaluator._best_slot_damage_into_target(
                        battle, 1, opp
                    )
        if risk >= 0.75 and hp < 0.80 and partner_punish >= 0.35:
            return 2.75 + 1.35 * risk + 0.45 * partner_punish
        if risk >= 0.95 and hp < 0.55:
            return 2.10 + 0.80 * risk
        if (
            self.config.benchmark_type == "vs_simpleheuristics"
            and risk >= 0.55
            and hp < 0.45
            and partner_punish >= 0.25
        ):
            return 1.35 + 0.65 * risk
        return -0.20

    def _role_based_focus_bonus(self, battle: MultiBattle, action: SlotAction) -> float:
        weight = float(getattr(self.config, "role_focus_weight", 1.0) or 1.0)
        if weight <= 0.0:
            return 0.0
        if action.kind != "move" or action.move is None:
            return 0.0
        if move_base_power(action.move) <= 0:
            return 0.0
        if is_spread_move(action.move):
            return 0.10 * weight
        try:
            position = int(action.target_position or 0)
        except Exception:
            return 0.0
        opps = active_alive_mons(battle.opponent_active_pokemon[:2])
        if len(opps) < 2:
            return 0.0
        opp1_pos = int(getattr(battle, "OPPONENT_1_POSITION", 1))
        opp2_pos = int(getattr(battle, "OPPONENT_2_POSITION", 2))

        try:
            opp1 = (
                battle.opponent_active_pokemon[0]
                if len(battle.opponent_active_pokemon) > 0
                else None
            )
            opp2 = (
                battle.opponent_active_pokemon[1]
                if len(battle.opponent_active_pokemon) > 1
                else None
            )
        except Exception:
            opp1 = opp2 = None
        if opp1 is not None and not is_fainted(opp1) and opp2 is not None and not is_fainted(opp2):
            hp1 = safe_hp_fraction(opp1)
            hp2 = safe_hp_fraction(opp2)

            if min(hp1, hp2) < 0.55 and abs(hp1 - hp2) > 0.20:
                target_pos = opp1_pos if hp1 < hp2 else opp2_pos
                if position == target_pos:
                    return 0.45 * weight
                return -0.20 * weight
        threat = self.slot_agent.tactical.threat_model.analyze(battle)
        target = action.target
        if target is not None and threat.global_target != "none":
            if safe_species(target) == threat.global_target:
                return 0.35 * weight
            return -0.08 * weight
        return 0.0

    def _apply_learned_value_bonus(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
        partner_placeholder: SlotAction,
    ) -> List[Tuple[float, SlotAction]]:
        weight = float(getattr(self.config, "learned_value_weight", 0.0) or 0.0)
        if weight <= 0.0:
            return scored
        if self.config.benchmark_type in {"vs_simpleheuristics", "vs_abyssal"}:
            weight = min(
                weight,
                float(getattr(self.config, "hard_benchmark_learned_value_cap", 0.15) or 0.15),
            )
        clip = float(getattr(self.config, "learned_value_clip", 0.45) or 0.45)
        adjusted: List[Tuple[float, SlotAction]] = []
        for score, action in scored:
            joint = JointAction(action, partner_placeholder)
            try:
                value = self.model.predict(self.encoder.encode(battle, joint))
            except Exception:
                value = 0.0
            value = max(-clip, min(clip, float(value)))
            adjusted.append((float(score) + weight * value, action))
        return adjusted

    def _apply_transformer_action_prior(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
        partner_messages: List[Dict[str, Any]],
    ) -> List[Tuple[float, SlotAction]]:
        prior = getattr(self, "transformer_action_prior", None)
        if prior is None or not getattr(prior, "available", False):
            return scored
        weight = float(getattr(self.config, "transformer_action_prior_weight", 0.0) or 0.0)
        if weight <= 0.0:
            return scored
        clip = float(getattr(self.config, "transformer_action_prior_clip", 2.0) or 2.0)
        try:
            summaries = [
                self._public_candidate_summary(battle, score, action) for score, action in scored
            ]
            prompt = build_action_prior_prompt(
                battle,
                agent_name=self.agent_name,
                benchmark_type=str(getattr(self.config, "benchmark_type", "default")),
                partner_messages=partner_messages,
            )
            lines = [
                candidate_action_line(battle, summary, partner_messages) for summary in summaries
            ]
            raw_scores = prior.score_lines(prompt, lines)
            bonuses = normalized_prior_scores(raw_scores, clip)
            return [
                (float(score) + weight * float(bonus), action)
                for (score, action), bonus in zip(scored, bonuses)
            ]
        except Exception:
            return scored

    @staticmethod
    def _is_low_progress_move(action: SlotAction) -> bool:
        if action.kind != "move" or action.move is None:
            return False
        mid = move_id(action.move)
        if mid in FAKE_OUT_MOVES:
            return False
        if mid in RELIABLE_TEMPO_SPEED_MOVES:
            return False
        if float(move_base_power(action.move)) > 0:
            return False
        return (
            mid in PROTECT_MOVES
            or mid in SETUP_MOVES
            or mid in HELPING_HAND_MOVES
            or mid in REDIRECTION_MOVES
            or mid in LOW_PROGRESS_SUPPORT_MOVES
            or mid in ONE_TIME_FIELD_MOVES
            or mid in TERRAIN_MOVES
            or mid in SCREEN_MOVES
            or mid in REPEAT_BAD_STATUS_MOVES
        )

    def _should_preserve_macro_support_action(
        self,
        battle: MultiBattle,
        action: SlotAction,
        threat: ThreatEstimate,
        best_damage: float,
    ) -> bool:
        if action.kind != "move" or action.move is None or move_base_power(action.move) > 0:
            return False
        mid = move_id(action.move)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        active = battle.active_pokemon[0] if battle.active_pokemon else None
        partner = battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
        if mid in TERRAIN_SET_MOVES:
            if mid in _battle_field_names(battle):
                return False
            boosted_type = TERRAIN_BOOST_TYPES.get(mid, "")
            grounded_boosted = any(
                boosted_type and boosted_type in _mon_type_names(mon) and _is_grounded_approx(mon)
                for mon in active_alive_mons(battle.active_pokemon)
            )
            return bool(turn <= 4 or grounded_boosted or best_damage < 0.45)
        if mid in TERRAIN_REMOVE_MOVES:
            return bool(_battle_field_names(battle))
        if mid in SCREEN_MOVES:
            if mid in _battle_side_condition_names(battle, own_side=True):
                return False
            incoming = max(
                max(threat.slot_threat.values(), default=0.0),
                max(threat.slot_ko_risk.values(), default=0.0),
            )
            return bool(turn <= 3 or incoming >= 0.45)
        if mid in REDIRECTION_MOVES:
            partner_risk = max(threat.slot_threat.get(1, 0.0), threat.slot_ko_risk.get(1, 0.0))
            active_hp_ok = active is not None and safe_hp_fraction(active) >= 0.45
            return bool(active_hp_ok and (partner_risk >= 0.40 or turn <= 2))
        if mid in STATUS_CONTROL_MOVES:
            if action.target is None or not _target_is_opponent(battle, action.target):
                return False
            if not self.slot_agent.tactical._status_control_target_is_valid(
                battle, action.target, mid
            ):
                return False
            target_label = safe_species(action.target)
            target_threat = 0.0
            if target_label == threat.global_target:
                target_threat = max(target_threat, float(threat.global_pressure))
            for idx, label in threat.threatening_opp.items():
                if label == target_label:
                    target_threat = max(target_threat, float(threat.slot_threat.get(idx, 0.0)))
            return bool(turn <= 4 or target_threat >= 0.35 or best_damage < 0.55)
        if mid in SETUP_MOVES:
            own_risk = max(threat.slot_threat.get(0, 0.0), threat.slot_ko_risk.get(0, 0.0))
            return bool(
                active is not None
                and safe_hp_fraction(active) >= 0.70
                and own_risk <= 0.42
                and turn <= 4
            )
        if mid in HELPING_HAND_MOVES:
            partner_best = 0.0
            if partner is not None:
                partner_moves_raw = safe_getattr(partner, "moves", {}) or {}
                partner_moves = (
                    list(partner_moves_raw.values())
                    if isinstance(partner_moves_raw, dict)
                    else list(partner_moves_raw)
                )


                if not partner_moves:
                    partner_moves = OpponentThreatModel._predicted_moves_for_opp(partner)
                for move in partner_moves:
                    if move_base_power(move) <= 0:
                        continue
                    if is_spread_move(move):
                        partner_best = max(
                            partner_best,
                            sum(
                                _advanced_damage_ratio(battle, move, partner, opp, spread=True)
                                for opp in active_alive_mons(battle.opponent_active_pokemon)
                                if damage_multiplier(opp, move) > 0
                            ),
                        )
                    else:
                        partner_best = max(
                            partner_best,
                            max(
                                (
                                    _advanced_damage_ratio(battle, move, partner, opp)
                                    for opp in active_alive_mons(battle.opponent_active_pokemon)
                                    if damage_multiplier(opp, move) > 0
                                ),
                                default=0.0,
                            ),
                        )
            return bool(partner_best >= 0.65)
        return False


__all__ = ["MultiAgentActionFilterMixin"]
