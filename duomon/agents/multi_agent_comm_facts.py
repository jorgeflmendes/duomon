from __future__ import annotations

from .multi_agent_context import *


class MultiAgentCommFactsMixin:
    @staticmethod
    def _partner_best_damage_from_messages(
        partner_messages: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        best: Dict[int, float] = {}
        for message in partner_messages or []:
            if not isinstance(message, dict):
                continue
            caps = message.get("capabilities", {}) or {}
            for raw_slot, value in (caps.get("best_damage_by_opp_slot", {}) or {}).items():
                try:
                    slot = int(raw_slot)
                    best[slot] = max(best.get(slot, 0.0), utility_damage_ratio(value))
                except Exception:
                    continue
        return best

    def _build_structured_comm_message(
        self,
        battle: MultiBattle,
        scored: List[Tuple[float, SlotAction]],
    ) -> Dict[str, Any]:
        summaries = [
            self._public_candidate_summary(battle, score, action) for score, action in scored[:8]
        ]
        proposals = self._strategy_proposals_from_summaries(battle, summaries)
        top_strategy = proposals[0]["strategy"] if proposals else "local_best_action"
        message = {
            "schema": "vgc_structured_comm_v1",
            "speech_act": "inform_propose",
            "agent": self.agent_name,
            "role": str(_multi_side_role(battle) or "unknown"),
            "battle_tag": str(safe_getattr(battle, "battle_tag", "unknown")),
            "turn": int(safe_getattr(battle, "turn", 0) or 0),
            "facts": self._structured_board_facts(battle),
            "memory": self._communication_memory_snapshot(battle),
            "capabilities": self._structured_capabilities(battle, summaries),
            "proposals": proposals,
            "top_strategy": top_strategy,
            "top_candidates": summaries[:5],
            "vector": self._communication_state_vector(battle, summaries, proposals),
            "rationale": [
                "legal_action_mask_first",
                "deny_extra_opponent_action_by_finishing_low_hp_targets",
                "coordinate_double_targets_only_when_partner_damage_is_needed",
                "use_protect_as_positioning_when_partner_can_remove_threat",
                "gate_terastallization_until_expected_value_is_positive",
                "penalize_unreliable_moves_when_the_slot_is_at_ko_risk",
                "treat_field_control_pivoting_self_activation_and_setup_support_as_explicit_team_intents",
            ],
        }
        if str(getattr(self.config, "communication_type", "") or "").lower() in {
            "transformer_prior",
        }:
            message["transformer_intent"] = self._transformer_intent_from_summaries(summaries)
            message["speech_act"] = "latent_intent"
            message["schema"] = "duomon_transformer_comm_v1"
        return message

    @staticmethod
    def _transformer_intent_from_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = sorted(
            [summary for summary in summaries if isinstance(summary, dict)],
            key=lambda summary: float(summary.get("score", 0.0) or 0.0),
            reverse=True,
        )
        top = scored[:3]
        target_scores: Dict[int, float] = {}
        for summary in scored:
            slot = summary.get("target_slot")
            try:
                slot_int = int(slot)
            except Exception:
                continue
            if slot_int not in {0, 1}:
                continue
            target_scores[slot_int] = max(
                target_scores.get(slot_int, -999.0),
                float(summary.get("score", 0.0) or 0.0),
            )
        best_target = None
        if target_scores:
            best_target = max(target_scores.items(), key=lambda item: item[1])[0]
        score_values = [float(summary.get("score", 0.0) or 0.0) for summary in top]
        margin = (
            (score_values[0] - score_values[1])
            if len(score_values) > 1
            else (score_values[0] if score_values else 0.0)
        )
        gate = max(0.0, min(1.0, margin / 4.0))
        return {
            "source": "transformer_action_prior",
            "target_slot": best_target,
            "gate": gate,
            "score_margin": float(margin),
            "target_scores": {str(k): float(v) for k, v in target_scores.items()},
            "top_actions": [
                {
                    "label": str(summary.get("label", "")),
                    "move_id": str(summary.get("move_id", "")),
                    "target_slot": summary.get("target_slot"),
                    "target_hp": summary.get("target_hp"),
                    "score": float(summary.get("score", 0.0) or 0.0),
                    "damage_by_slot": summary.get("damage_by_slot", {}),
                    "ko_by_slot": summary.get("ko_by_slot", {}),
                    "protect": bool(summary.get("protect", False)),
                    "spread": bool(summary.get("spread", False)),
                }
                for summary in top
            ],
        }

    def _structured_board_facts(self, battle: MultiBattle) -> Dict[str, Any]:
        return {
            "self_active": self._pokemon_public_fact(
                battle.active_pokemon[0] if battle.active_pokemon else None
            ),
            "partner_active": self._pokemon_public_fact(
                battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None
            ),
            "opponent_active": [
                self._pokemon_public_fact(mon)
                for mon in (
                    battle.opponent_active_pokemon[:2] if battle.opponent_active_pokemon else []
                )
            ],
            "weather": _lower_names(safe_getattr(battle, "weather", None)),
            "fields": _battle_field_names(battle),
            "own_side_conditions": _battle_side_condition_names(battle, own_side=True),
            "opponent_side_conditions": _battle_side_condition_names(battle, own_side=False),
        }

    def _pokemon_public_fact(self, mon: Any) -> Dict[str, Any]:
        if mon is None:
            return {"species": "none", "known": False}
        moves = safe_getattr(mon, "moves", {}) or {}
        move_values = list(moves.values()) if isinstance(moves, dict) else list(moves)
        return {
            "species": safe_species(mon),
            "known": True,
            "hp": round(float(safe_hp_fraction(mon)), 4),
            "speed": int(safe_speed(mon)),
            "types": _mon_type_names(mon),
            "item": safe_compact_name(safe_getattr(mon, "item", None)),
            "ability": safe_compact_name(safe_getattr(mon, "ability", None)),
            "moves": [
                {
                    "id": move_id(move),
                    "bp": float(move_base_power(move)),
                    "type": _move_type_name(move),
                    "category": move_category(move),
                    "accuracy": move_accuracy(move),
                    "priority": move_priority(move),
                    "target": move_target_type(move),
                }
                for move in move_values[:4]
            ],
        }

    def _structured_capabilities(
        self,
        battle: MultiBattle,
        summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        moves = normalize_slot_list(safe_getattr(battle, "available_moves", []), 0)
        return {
            "has_protect": any(move_id(move) in PROTECT_MOVES for move in moves),
            "fake_out": [move_id(move) for move in moves if move_id(move) in FAKE_OUT_MOVES],
            "speed_control": [
                move_id(move) for move in moves if move_id(move) in SPEED_CONTROL_MOVES
            ],
            "redirection": [move_id(move) for move in moves if move_id(move) in REDIRECTION_MOVES],
            "setup": [move_id(move) for move in moves if move_id(move) in SETUP_MOVES],
            "pivot": [move_id(move) for move in moves if move_id(move) in PIVOT_MOVES],
            "terrain_control": [move_id(move) for move in moves if move_id(move) in TERRAIN_MOVES],
            "screens": [move_id(move) for move in moves if move_id(move) in SCREEN_MOVES],
            "self_activation": [s for s in summaries if bool(s.get("ally_activation"))],
            "spread_damage": [
                s
                for s in summaries
                if bool(s.get("spread")) and float(s.get("damage_sum", 0.0)) > 0.0
            ],
            "terrain_synergy": self._terrain_synergy_profile(battle),
            "partner_setup_capable": self._mon_has_any_move(
                battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None,
                SETUP_MOVES,
            ),
            "partner_redirection_capable": self._mon_has_any_move(
                battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None,
                REDIRECTION_MOVES | FAKE_OUT_MOVES,
            ),
            "partner_activation_receiver": self._activation_receiver_profile(
                battle.active_pokemon[1] if len(battle.active_pokemon) > 1 else None,
            ),
            "best_damage_by_opp_slot": self._best_summary_damage_by_slot(summaries),
            "predicted_self_danger": self._predicted_self_danger(battle),
            "solo_ko_slots": [
                int(slot)
                for slot, value in self._best_summary_damage_by_slot(summaries).items()
                if float(value) >= 0.98
            ],
        }

    @staticmethod
    def _mon_has_any_move(mon: Any, move_ids: set[str]) -> bool:
        moves = safe_getattr(mon, "moves", {}) or {}
        values = moves.values() if isinstance(moves, dict) else moves
        return any(move_id(move) in move_ids for move in values)

    @staticmethod
    def _activation_receiver_profile(mon: Any) -> Dict[str, Any]:
        if mon is None:
            return {"enabled": False}
        item = _mon_item_name(mon)
        ability = _mon_ability_name(mon)
        return {
            "enabled": bool(item in ALLY_ACTIVATION_ITEMS or ability in ALLY_ACTIVATION_ABILITIES),
            "species": safe_species(mon),
            "item": item,
            "ability": ability,
        }

    def _terrain_synergy_profile(self, battle: MultiBattle) -> Dict[str, Any]:
        allies = active_alive_mons(battle.active_pokemon[:2])
        enemy_priority = any(
            any(
                move_priority(move) > 0
                for move in (
                    (safe_getattr(opp, "moves", {}) or {}).values()
                    if isinstance(safe_getattr(opp, "moves", {}) or {}, dict)
                    else safe_getattr(opp, "moves", {}) or []
                )
            )
            for opp in active_alive_mons(battle.opponent_active_pokemon[:2])
        )
        ally_moves = []
        grounded_ally_moves = []
        for mon in allies:
            moves = safe_getattr(mon, "moves", {}) or {}
            for move in moves.values() if isinstance(moves, dict) else moves:
                ally_moves.append(move)
                if _is_grounded_approx(mon):
                    grounded_ally_moves.append(move)
        move_types = {_move_type_name(move) for move in grounded_ally_moves}
        return {
            "psychic_blocks_priority": bool(enemy_priority),
            "boosted_types_available": sorted(
                t for t in move_types if t in {"electric", "grass", "psychic"}
            ),
            "has_priority_ally": any(move_priority(move) > 0 for move in ally_moves),
        }

    @staticmethod
    def _best_summary_damage_by_slot(summaries: List[Dict[str, Any]]) -> Dict[int, float]:
        best: Dict[int, float] = {}
        for summary in summaries:
            for raw_slot, value in (summary.get("damage_by_slot", {}) or {}).items():
                try:
                    slot = int(raw_slot)
                    best[slot] = max(best.get(slot, 0.0), utility_damage_ratio(value))
                except Exception:
                    continue
        return best

    def _public_candidate_summary(
        self, battle: MultiBattle, score: float, action: SlotAction
    ) -> Dict[str, Any]:
        raw = self._candidate_summary(battle, score, action)
        damage_by_slot = {
            int(k): float(v) for k, v in (raw.get("damage_by_slot", {}) or {}).items()
        }
        ko_by_slot = {int(k): float(v) for k, v in (raw.get("ko_by_slot", {}) or {}).items()}
        return {
            "signature": self._order_signature(action.order),
            "label": action.label,
            "score": float(score),
            "kind": action.kind,
            "move_id": str(raw.get("move_id", "")),
            "bp": float(raw.get("bp", 0.0)),
            "accuracy": float(raw.get("accuracy", 1.0)),
            "priority": float(raw.get("priority", 0.0)),
            "attacker_speed": float(raw.get("attacker_speed", 0.0)),
            "target": raw.get("target"),
            "target_slot": raw.get("target_slot"),
            "target_hp": self._target_hp_for_slot(battle, raw.get("target_slot")),
            "spread": bool(action.move is not None and is_spread_move(action.move)),
            "support": bool(action.move is not None and move_base_power(action.move) <= 0),
            "protect": bool(action.move is not None and move_id(action.move) in PROTECT_MOVES),
            "recoil": bool(action.move is not None and move_id(action.move) in RECOIL_MOVES),
            "self_sacrifice": bool(
                action.move is not None and move_id(action.move) in SELF_SACRIFICE_MOVES
            ),
            "tera": bool(action_uses_tera(action)),
            "speed_control": bool(
                action.move is not None and move_id(action.move) in SPEED_CONTROL_MOVES
            ),
            "redirection": bool(
                action.move is not None and move_id(action.move) in REDIRECTION_MOVES
            ),
            "setup": bool(action.move is not None and move_id(action.move) in SETUP_MOVES),
            "pivot": bool(action.move is not None and move_id(action.move) in PIVOT_MOVES),
            "terrain": bool(action.move is not None and move_id(action.move) in TERRAIN_MOVES),
            "screen": bool(action.move is not None and move_id(action.move) in SCREEN_MOVES),
            "ally_target": bool(raw.get("ally_target", False)),
            "ally_activation": bool(raw.get("ally_activation", False)),
            "ally_damage": float(raw.get("ally_damage", 0.0)),
            "field_effect": raw.get("field_effect"),
            "damage": {str(k): float(v) for k, v in (raw.get("damage", {}) or {}).items()},
            "damage_by_slot": damage_by_slot,
            "ko_by_slot": ko_by_slot,
            "damage_sum": float(sum(damage_by_slot.values())),
            "ko_sum": float(sum(ko_by_slot.values())),
        }

    @staticmethod
    def _target_hp_for_slot(battle: MultiBattle, slot: Any) -> Optional[float]:
        try:
            idx = int(slot)
            mon = (
                battle.opponent_active_pokemon[idx]
                if idx < len(battle.opponent_active_pokemon)
                else None
            )
            return safe_hp_fraction(mon) if mon is not None else None
        except Exception:
            return None

    @staticmethod
    def _field_effect_for_move(move: Any) -> Optional[str]:
        mid = move_id(move)
        if mid in TERRAIN_SET_MOVES:
            return mid
        if mid in TERRAIN_REMOVE_MOVES:
            return "remove_terrain"
        if mid in SCREEN_MOVES:
            return mid
        return None


__all__ = ["MultiAgentCommFactsMixin"]
