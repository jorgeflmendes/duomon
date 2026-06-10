from __future__ import annotations

from .policy_context import *
from .policy_actions import *
from .policy_features import *
from .policy_slot_agent import *
from .policy_logging import *


class IndependentTwoSlotAgent(Player):
    def __init__(self, config: AgentConfig, **kwargs):
        self.config = config
        self.mode = AgentMode.INDEPENDENT
        self.encoder = FeatureEncoder()
        self.model = LinearPolicyValueModel(config, self.encoder)
        self.replay_logger = ReplayLogger(
            config.replay_path,
            flush_every=config.replay_flush_every,
            benchmark_type=config.benchmark_type,
        )
        self.action_generator: Optional[LegalActionGenerator] = None
        self.left_agent = SlotAgent(0, config)
        self.right_agent = SlotAgent(1, config)
        self.last_hp_balance: Dict[str, float] = {}
        self.turn_call_counter: Dict[str, int] = {}
        self.last_logged_turn_action: Dict[str, str] = {}
        self.last_order_by_request: Dict[str, SingleBattleOrder] = {}
        self.battle_memory: Dict[str, Dict[str, Any]] = {}
        self.finished_battles_seen: set[str] = set()
        super().__init__(**kwargs)
        self.action_generator = LegalActionGenerator(self)

    async def choose_move(self, battle: DoubleBattle):
        if battle.finished:
            self._handle_terminal_if_needed(battle)
            return DoubleBattleOrder(DefaultBattleOrder(), DefaultBattleOrder())

        battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        call_key = f"{battle_key}:{turn}"
        self.turn_call_counter[call_key] = self.turn_call_counter.get(call_key, 0) + 1

        forced = force_switch_list(battle)
        if any(forced):
            self._log_unique(battle, f"forced-switch {forced} | safe deterministic switch")
            return self._safe_forced_switch_order(battle, forced)

        if self.turn_call_counter[call_key] > self.config.max_repeated_turn_calls:
            logger.info(
                f"[decision] turn={turn} action=anti_loop repeated_requests={self.turn_call_counter[call_key]} fallback=safe_attack"
            )
            return self._safe_non_switch_order(battle)

        start = time.time()
        try:
            self._online_update_from_current_state(battle)
            if turn >= self.config.max_battle_turns_soft:
                self._log_unique(battle, f"turn={turn} reached_soft_limit fallback=safe_attack")
                return self._safe_non_switch_order(battle)

            chosen, features, value = self._choose_independent_action(battle)
            self.model.remember(battle_key, features, value)
            self.replay_logger.log_turn(self.mode, battle, chosen, features, value)
            self._record_chosen_action(battle, chosen)

            if self.config.log_every_turn:
                elapsed = time.time() - start
                self._log_unique(
                    battle,
                    f"independent | {elapsed:.2f}s | value={value:.3f} | search={chosen.search_score:.3f} | "
                    f"compat=0.000 | pen=0.000 | {chosen.label}",
                )
            self._remember_hp_balance(battle)
            return DoubleBattleOrder(chosen.left.order, chosen.right.order)
        except Exception as exc:
            logger.exception(f"[decision] action=fallback reason=independent_exception error={exc}")
            return self._safe_non_switch_order(battle)

    def _choose_independent_action(
        self, battle: DoubleBattle
    ) -> Tuple[JointAction, np.ndarray, float]:
        left_top = self.left_agent.top_k(
            battle,
            self.action_generator.generate_slot_actions(battle, 0),
            self.config.top_k_slot_actions,
        )
        right_top = self.right_agent.top_k(
            battle,
            self.action_generator.generate_slot_actions(battle, 1),
            self.config.top_k_slot_actions,
        )

        left_action = left_top[0][1]
        right_action = right_top[0][1]
        if (
            left_action.kind == "switch"
            and right_action.kind == "switch"
            and left_action.switch is right_action.switch
            and len(right_top) > 1
        ):
            right_action = right_top[1][1]
        chosen = JointAction(left_action, right_action)

        features = self.encoder.encode(battle, chosen)
        value = self.model.predict(features)
        chosen.value = value
        if not chosen.search_score:
            chosen.search_score = value
        return chosen, features, value

    def _memory(self, battle: DoubleBattle) -> Dict[str, Any]:
        battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
        if battle_key not in self.battle_memory:
            self.battle_memory[battle_key] = {
                "used_moves_by_slot": {0: {}, 1: {}},
                "used_action_labels_by_slot": {0: {}, 1: {}},
                "used_status_on_target": {},
                "last_actions": [],
                "field_moves_used": set(),
                "last_protect_turn_by_slot": {},
            }
        return self.battle_memory[battle_key]

    def _record_chosen_action(self, battle: DoubleBattle, action: JointAction) -> None:
        mem = self._memory(battle)
        turn = int(safe_getattr(battle, "turn", 0) or 0)
        for sa in [action.left, action.right]:
            if sa.kind != "move" or sa.move is None:
                continue
            mid = move_id(sa.move)
            mem["used_moves_by_slot"][sa.slot][mid] = (
                mem["used_moves_by_slot"][sa.slot].get(mid, 0) + 1
            )
            label_counts = mem["used_action_labels_by_slot"].setdefault(sa.slot, {})
            label_counts[sa.label] = label_counts.get(sa.label, 0) + 1
            if mid in ONE_TIME_FIELD_MOVES:
                mem["field_moves_used"].add(mid)
            if mid in PROTECT_MOVES:
                mem["last_protect_turn_by_slot"][sa.slot] = turn
            status_target = self._action_target_for_memory(battle, sa)
            if status_target is not None and mid in REPEAT_BAD_STATUS_MOVES:
                key = (mid, safe_species(status_target))
                mem["used_status_on_target"][key] = mem["used_status_on_target"].get(key, 0) + 1
        mem["last_actions"].append((turn, action.label))
        mem["last_actions"] = mem["last_actions"][-8:]

    @staticmethod
    def _action_target_for_memory(battle: DoubleBattle, action: SlotAction) -> Optional[Any]:
        if action.target is not None:
            return action.target
        try:
            position = int(action.target_position or 0)
        except Exception:
            return None
        if position == 0:
            return None
        return LegalActionGenerator._target_from_position(battle, position)

    @staticmethod
    def _slot_damage_progress_score(battle: DoubleBattle, action: SlotAction) -> float:
        if action.kind != "move" or action.move is None or move_base_power(action.move) <= 0:
            return 0.0
        try:
            attacker = (
                battle.active_pokemon[action.slot]
                if action.slot < len(battle.active_pokemon)
                else None
            )
        except Exception:
            attacker = None
        if attacker is None:
            return 0.0
        if is_spread_move(action.move):
            return sum(
                _advanced_damage_ratio(battle, action.move, attacker, opp, spread=True)
                for opp in active_alive_mons(battle.opponent_active_pokemon)
                if damage_multiplier(opp, action.move) > 0
            )
        target = IndependentTwoSlotAgent._action_target_for_memory(battle, action)
        if (
            target is None
            or not _target_is_opponent(battle, target)
            or damage_multiplier(target, action.move) <= 0
        ):
            return 0.0
        return _advanced_damage_ratio(battle, action.move, attacker, target)

    def _contextual_action_is_bad(self, battle: DoubleBattle, action: JointAction) -> bool:
        mem = self._memory(battle)
        turn = int(safe_getattr(battle, "turn", 0) or 0)



        partner_is_placeholder = action.right.kind == "pass" and str(
            getattr(action.right, "label", "")
        ).startswith("partner-controlled")
        damage_score = self._joint_damage_score(battle, action)
        for sa in [action.left, action.right]:
            if sa.kind != "move" or sa.move is None:
                continue
            mid = move_id(sa.move)
            if mid in FIRST_TURN_ONLY_STYLE_MOVES and turn > 2:
                return True
            if known_target_is_immune(sa, battle):
                return True
            if mid in ONE_TIME_FIELD_MOVES and mid in mem["field_moves_used"]:
                return True
            status_target = self._action_target_for_memory(battle, sa)
            if (
                status_target is not None
                and mid in REPEAT_BAD_STATUS_MOVES
                and mem["used_status_on_target"].get((mid, safe_species(status_target)), 0) >= 1
            ):
                return True
            label_count = (
                mem.get("used_action_labels_by_slot", {}).get(sa.slot, {}).get(sa.label, 0)
            )
            if label_count >= 2 and move_base_power(sa.move) > 0:
                progress = self._slot_damage_progress_score(battle, sa)
                target = self._action_target_for_memory(battle, sa)
                target_hp = safe_hp_fraction(target) if target is not None else 1.0
                if progress < 0.30 and target_hp > 0.35:
                    return True
            if (
                not partner_is_placeholder
                and mid in LOW_PROGRESS_SUPPORT_MOVES
                and damage_score < 0.12
            ):
                return True
            if not partner_is_placeholder and mid in SETUP_MOVES and damage_score < 0.12:
                return True
            if mid in PROTECT_MOVES and mem["last_protect_turn_by_slot"].get(sa.slot) == turn - 1:
                return True
            if mid in SETUP_MOVES and mem["used_moves_by_slot"][sa.slot].get(mid, 0) >= 2:
                return True
        return [label for _, label in mem["last_actions"][-4:]].count(action.label) >= 2

    def _post_filter_actions(
        self, battle: DoubleBattle, actions: List[JointAction]
    ) -> List[JointAction]:
        result = [
            a
            for a in actions
            if self._no_pass_with_living_active(battle, a)
            and not self._contextual_action_is_bad(battle, a)
        ]
        result = result or actions[:]
        result.sort(key=lambda a: a.search_score, reverse=True)
        return result

    @staticmethod
    def _no_pass_with_living_active(battle: DoubleBattle, action: JointAction) -> bool:
        for i, slot_action in enumerate([action.left, action.right]):
            active = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            if slot_action.kind == "pass" and active is not None and not is_fainted(active):
                return False
        return True

    @staticmethod
    def _joint_damage_score(battle: DoubleBattle, action: JointAction) -> float:
        score = 0.0
        for sa in [action.left, action.right]:
            if sa.kind != "move" or sa.move is None or move_base_power(sa.move) <= 0:
                continue
            attacker = (
                battle.active_pokemon[sa.slot] if sa.slot < len(battle.active_pokemon) else None
            )
            if is_spread_move(sa.move):
                score += sum(
                    approximate_damage_points(sa.move, attacker, opp, spread=True)
                    / max(1.0, estimated_max_hp_points(opp))
                    for opp in active_alive_mons(battle.opponent_active_pokemon)
                )
            elif sa.target is not None:
                score += approximate_damage_points(sa.move, attacker, sa.target) / max(
                    1.0, estimated_max_hp_points(sa.target)
                )
        return score

    def _safe_forced_switch_order(self, battle: DoubleBattle, forced: Optional[List[bool]] = None):
        try:
            forced = forced or force_switch_list(battle)
            orders: List[SingleBattleOrder] = []
            used_switch_ids: set[str] = set()
            active_species = {
                safe_species(m)
                for m in safe_getattr(battle, "active_pokemon", []) or []
                if m is not None
            }
            for slot in range(2):
                if slot < len(forced) and forced[slot]:
                    switch_actions = [
                        a
                        for a in self.action_generator.generate_slot_actions(battle, slot)
                        if a.kind == "switch" and a.switch is not None
                    ]
                    filtered = [
                        a
                        for a in switch_actions
                        if safe_species(a.switch) not in used_switch_ids
                        and safe_species(a.switch) not in active_species
                    ]
                    filtered = filtered or [
                        a for a in switch_actions if safe_species(a.switch) not in used_switch_ids
                    ]
                    if not filtered:
                        return self._safe_random_order(battle)
                    filtered.sort(
                        key=lambda a: self._switch_in_pressure_score(a.switch, battle), reverse=True
                    )
                    chosen = filtered[0]
                    used_switch_ids.add(safe_species(chosen.switch))
                    orders.append(chosen.order)
                else:
                    orders.append(PassBattleOrder())
            return DoubleBattleOrder(orders[0], orders[1])
        except Exception:
            return self._safe_random_order(battle)

    @staticmethod
    def _switch_in_pressure_score(mon: Any, battle: Any = None) -> float:
        if mon is None:
            return -1e9
        hp = safe_hp_fraction(mon)
        moves = safe_getattr(mon, "moves", {}) or {}
        move_values = moves.values() if isinstance(moves, dict) else moves
        best_bp = 0.0
        spread_bp = 0.0
        priority = 0.0
        accuracy = 0.0
        for move in move_values:
            bp = float(move_base_power(move))
            if bp <= 0:
                continue
            acc = move_accuracy(move)
            stab = 1.5 if has_type(mon, move) else 1.0
            value = bp * acc * stab * move_expected_hits(move)
            best_bp = max(best_bp, value)
            if is_spread_move(move):
                spread_bp = max(spread_bp, value)
            priority = max(priority, float(move_priority(move)))
            accuracy = max(accuracy, acc)
        offenses = max(float(safe_stat(mon, "atk", 100)), float(safe_stat(mon, "spa", 100))) / 180.0
        bulk = (
            float(safe_stat(mon, "hp", 100))
            + float(safe_stat(mon, "def", 100))
            + float(safe_stat(mon, "spd", 100))
        ) / 360.0


        offensive_match = 0.0
        defensive_match = 0.0
        if battle is not None:
            try:
                opps = active_alive_mons(battle.opponent_active_pokemon[:2])
                if opps:

                    off_best = 0.0
                    cand_moves = (
                        list(move_values)
                        if move_values
                        else OpponentThreatModel._predicted_moves_for_opp(mon)
                    )
                    for move in cand_moves:
                        if move_base_power(move) <= 0:
                            continue
                        for opp in opps:
                            mult = damage_multiplier(opp, move)
                            if mult > off_best:
                                off_best = mult
                    offensive_match = min(4.0, off_best)

                    incoming = 0.0
                    cnt = 0
                    for opp in opps:
                        opp_moves_raw = safe_getattr(opp, "moves", {}) or {}
                        opp_moves = (
                            list(opp_moves_raw.values())
                            if isinstance(opp_moves_raw, dict)
                            else list(opp_moves_raw)
                        )
                        if not opp_moves:
                            opp_moves = OpponentThreatModel._predicted_moves_for_opp(opp)
                        worst = 0.0
                        for move in opp_moves:
                            if move_base_power(move) <= 0:
                                continue
                            mult = damage_multiplier(mon, move)
                            if mult > worst:
                                worst = mult
                        if worst > 0:
                            incoming += worst
                            cnt += 1
                    if cnt > 0:
                        avg_incoming = incoming / cnt

                        defensive_match = max(0.0, 2.0 - avg_incoming)
            except Exception:
                pass
        return (
            2.15 * hp
            + 0.62 * bulk
            + 0.006 * float(safe_speed(mon))
            + 0.018 * min(150.0, best_bp)
            + 0.010 * min(150.0, spread_bp)
            + 0.30 * max(0.0, priority)
            + 0.20 * accuracy
            + 0.45 * offenses
            + 1.20 * offensive_match
            + 1.00 * defensive_match
        )

    def _safe_non_switch_order(self, battle: DoubleBattle):
        try:
            forced = force_switch_list(battle)
            if any(forced):
                return self._safe_forced_switch_order(battle, forced)
            orders: List[SingleBattleOrder] = []
            for slot in range(2):
                active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
                if active is None or is_fainted(active):
                    orders.append(PassBattleOrder())
                    continue
                actions = self.action_generator.generate_slot_actions(battle, slot)
                move_actions = [a for a in actions if a.kind == "move" and a.move is not None]
                if move_actions:
                    scorer = self.left_agent if slot == 0 else self.right_agent
                    move_actions.sort(key=lambda a: scorer.score(battle, a), reverse=True)
                    orders.append(move_actions[0].order)
                else:
                    fallback = [a for a in actions if a.kind in {"default", "pass"}]
                    orders.append(fallback[0].order if fallback else DefaultBattleOrder())
            return DoubleBattleOrder(orders[0], orders[1])
        except Exception:
            return self._safe_random_order(battle)

    def _safe_random_order(self, battle: DoubleBattle):
        try:
            return self.choose_random_doubles_move(battle)
        except Exception:
            return DoubleBattleOrder(DefaultBattleOrder(), DefaultBattleOrder())

    def _online_update_from_current_state(self, battle: DoubleBattle) -> None:
        battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
        current_balance = self._hp_balance(battle)
        previous_balance = self.last_hp_balance.get(battle_key, current_balance)
        reward = max(-1.0, min(1.0, current_balance - previous_balance))
        next_value = self._estimate_current_state_value(battle)
        self.model.update_td(battle_key, reward, next_value, terminal=False)
        self.last_hp_balance[battle_key] = current_balance

    def _estimate_current_state_value(self, battle: DoubleBattle) -> float:
        left_top = self.left_agent.top_k(
            battle,
            self.action_generator.generate_slot_actions(battle, 0),
            min(6, self.config.top_k_slot_actions),
        )
        right_top = self.right_agent.top_k(
            battle,
            self.action_generator.generate_slot_actions(battle, 1),
            min(6, self.config.top_k_slot_actions),
        )
        joint_actions = self._post_filter_actions(
            battle, JointCoordinator.enumerate_joint_actions(left_top, right_top)
        )
        values = [
            self.model.predict(self.encoder.encode(battle, action)) for action in joint_actions[:24]
        ]
        return max(values) if values else -0.5

    def _hp_balance(self, battle: DoubleBattle) -> float:
        my_team = safe_getattr(battle, "team", {}) or {}
        opp_team = safe_getattr(battle, "opponent_team", {}) or {}
        return (
            sum(safe_hp_fraction(m) for m in my_team.values())
            - sum(safe_hp_fraction(m) for m in opp_team.values())
        ) / 6.0

    def _remember_hp_balance(self, battle: DoubleBattle) -> None:
        self.last_hp_balance[str(safe_getattr(battle, "battle_tag", "unknown"))] = self._hp_balance(
            battle
        )

    def _handle_terminal_if_needed(self, battle: DoubleBattle) -> None:
        battle_key = str(safe_getattr(battle, "battle_tag", "unknown"))
        if battle_key in self.finished_battles_seen:
            return
        self.finished_battles_seen.add(battle_key)
        terminal_reward = (
            1.0
            if safe_getattr(battle, "won", False)
            else -1.0
            if safe_getattr(battle, "lost", False)
            else 0.0
        )
        self.model.update_td(battle_key, terminal_reward, 0.0, terminal=True)
        self.save_model()

    def apply_external_terminal_reward(self, battle_key: str, terminal_reward: float) -> None:
        self.model.update_td(str(battle_key), float(terminal_reward), 0.0, terminal=True)
        self.save_model()

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        self._handle_terminal_if_needed(battle)

    def save_model(self) -> None:
        self.replay_logger.flush()
        if self.config.use_online_learning:
            self.model.save()

    def _log_unique(self, battle: DoubleBattle, msg: str) -> None:
        key = f"{safe_getattr(battle, 'battle_tag', 'unknown')}:{int(safe_getattr(battle, 'turn', 0) or 0)}"
        if self.last_logged_turn_action.get(key) == msg:
            return
        self.last_logged_turn_action[key] = msg
        logger.info(f"Turn {int(safe_getattr(battle, 'turn', 0) or 0)} | {msg}")





def clone_config(config: AgentConfig, **overrides: Any) -> AgentConfig:
    data = asdict(config)
    data.update(overrides)
    return AgentConfig(**data)


def reset_file(path: str) -> None:
    try:
        ensure_parent_dir(path)
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
