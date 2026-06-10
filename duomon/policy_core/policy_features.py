from __future__ import annotations

from .policy_context import *
from .policy_actions import *


class FeatureEncoder:
    FEATURE_NAMES = [
        "bias",
        "my_hp_mean",
        "opp_hp_mean",
        "hp_advantage",
        "my_alive",
        "opp_alive",
        "speed_advantage",
        "damage_sum",
        "damage_max",
        "ko_prob_sum",
        "ko_prob_max",
        "spread_count",
        "focus_fire",
        "split_targets",
        "immune_count",
        "accuracy_mean",
        "priority_count",
        "protect_count",
        "helping_hand_count",
        "fakeout_count",
        "speed_control_count",
        "redirection_count",
        "setup_count",
        "pivot_count",
        "switch_count",
        "double_switch",
        "pass_count",
        "recoil_low_hp",
        "overkill",
        "partner_damage_risk",
        "opp_pressure",
        "support_without_attack",
        "message_agreement",
        "message_conflict",
        "my_left_hp",
        "my_right_hp",
        "opp_left_hp",
        "opp_right_hp",
        "turn_norm",
        "forced_switch_left",
        "forced_switch_right",
    ]

    def encode(
        self,
        battle: DoubleBattle,
        action: JointAction,
        metrics: Optional[CoordinationMetrics] = None,
    ) -> np.ndarray:
        my_active = active_alive_mons(battle.active_pokemon)
        opp_active = active_alive_mons(battle.opponent_active_pokemon)
        my_hp_mean = float(np.mean([safe_hp_fraction(m) for m in my_active])) if my_active else 0.0
        opp_hp_mean = (
            float(np.mean([safe_hp_fraction(m) for m in opp_active])) if opp_active else 0.0
        )
        my_team = safe_getattr(battle, "team", {}) or {}
        opp_team = safe_getattr(battle, "opponent_team", {}) or {}
        my_alive = len([m for m in my_team.values() if not is_fainted(m)])
        opp_alive = len([m for m in opp_team.values() if not is_fainted(m)])
        my_speed = float(np.mean([safe_speed(m) for m in my_active])) if my_active else 0.0
        opp_speed = float(np.mean([safe_speed(m) for m in opp_active])) if opp_active else 0.0
        computed = metrics or compute_coordination_metrics(battle, action)

        spread_count = immune_count = priority_count = protect_count = helping_hand_count = 0
        fakeout_count = speed_control_count = redirection_count = setup_count = pivot_count = 0
        switch_count = pass_count = recoil_low_hp = 0
        accuracies: List[float] = []
        overkill = 0.0

        for sa in [action.left, action.right]:
            if sa.kind == "pass":
                pass_count += 1
                continue
            if sa.kind == "switch":
                switch_count += 1
                continue
            if sa.kind != "move" or sa.move is None:
                continue
            mid = move_id(sa.move)
            move = sa.move
            attacker = (
                battle.active_pokemon[sa.slot] if sa.slot < len(battle.active_pokemon) else None
            )
            bp = move_base_power(move)
            accuracies.append(move_accuracy(move))
            priority_count += int(move_priority(move) > 0)
            protect_count += int(mid in PROTECT_MOVES)
            helping_hand_count += int(mid in HELPING_HAND_MOVES)
            fakeout_count += int(mid in FAKE_OUT_MOVES)
            speed_control_count += int(mid in SPEED_CONTROL_MOVES)
            redirection_count += int(mid in REDIRECTION_MOVES)
            setup_count += int(mid in SETUP_MOVES)
            pivot_count += int(mid in PIVOT_MOVES)
            recoil_low_hp += int(
                mid in RECOIL_MOVES and attacker is not None and safe_hp_fraction(attacker) < 0.40
            )
            if bp <= 0:
                continue
            if is_spread_move(move):
                spread_count += 1
                for opp in opp_active:
                    dmg = approximate_damage_points(move, attacker, opp, spread=True)
                    overkill += max(0.0, dmg - estimated_hp_points(opp)) / 100.0
            elif sa.target is not None and _target_is_opponent(battle, sa.target):
                if damage_multiplier(sa.target, move) == 0:
                    immune_count += 1
                dmg = approximate_damage_points(move, attacker, sa.target, spread=False)
                overkill += max(0.0, dmg - estimated_hp_points(sa.target)) / 100.0

        forced = force_switch_list(battle)
        values = [
            1.0,
            my_hp_mean,
            opp_hp_mean,
            my_hp_mean - opp_hp_mean,
            my_alive / 6.0,
            opp_alive / 6.0,
            (my_speed - opp_speed) / 300.0,
            computed.damage_sum,
            computed.damage_max,
            computed.ko_prob_sum,
            computed.ko_prob_max,
            float(spread_count),
            float(computed.focus_fire),
            float(computed.split_targets),
            float(immune_count),
            float(np.mean(accuracies)) if accuracies else 1.0,
            float(priority_count),
            float(protect_count),
            float(helping_hand_count),
            float(fakeout_count),
            float(speed_control_count),
            float(redirection_count),
            float(setup_count),
            float(pivot_count),
            float(switch_count),
            float(switch_count >= 2),
            float(pass_count),
            float(recoil_low_hp),
            overkill,
            computed.partner_damage_risk,
            self._estimate_opp_pressure(battle),
            float(computed.support_without_attack),
            float(computed.message_agreement),
            float(computed.message_conflict),
            safe_hp_fraction(battle.active_pokemon[0]) if len(battle.active_pokemon) > 0 else 0.0,
            safe_hp_fraction(battle.active_pokemon[1]) if len(battle.active_pokemon) > 1 else 0.0,
            safe_hp_fraction(battle.opponent_active_pokemon[0])
            if len(battle.opponent_active_pokemon) > 0
            else 0.0,
            safe_hp_fraction(battle.opponent_active_pokemon[1])
            if len(battle.opponent_active_pokemon) > 1
            else 0.0,
            min(1.0, float(int(safe_getattr(battle, "turn", 0) or 0)) / 20.0),
            float(forced[0]),
            float(forced[1]),
        ]
        return np.array(values, dtype=np.float32)

    @staticmethod
    def _estimate_opp_pressure(battle: DoubleBattle) -> float:
        pressure = 0.0
        for opp in active_alive_mons(battle.opponent_active_pokemon):
            moves = safe_getattr(opp, "moves", {}) or {}
            move_list = list(moves.values()) if isinstance(moves, dict) else list(moves)
            best = 0.0
            for move in move_list:
                for mine in active_alive_mons(battle.active_pokemon):
                    best = max(
                        best,
                        approximate_damage_points(move, opp, mine)
                        / max(1.0, estimated_max_hp_points(mine)),
                    )
            pressure += best
        return pressure


class LinearPolicyValueModel:
    MODEL_VERSION = "linear_td_v2_damage_calibrated"

    def __init__(self, config: AgentConfig, encoder: FeatureEncoder):
        self.config = config
        self.feature_names = encoder.FEATURE_NAMES
        self.weights = np.zeros(len(self.feature_names), dtype=np.float32)
        self._init_prior()
        self.previous_features: Dict[str, np.ndarray] = {}
        self.previous_value: Dict[str, float] = {}
        self.load()

    def _init_prior(self) -> None:
        priors = {
            "my_hp_mean": 0.8,
            "opp_hp_mean": -0.8,
            "hp_advantage": 1.1,
            "my_alive": 2.0,
            "opp_alive": -2.1,
            "speed_advantage": 0.35,
            "damage_sum": 3.2,
            "damage_max": 1.8,
            "ko_prob_sum": 4.5,
            "ko_prob_max": 3.8,
            "spread_count": 0.35,
            "focus_fire": 0.25,
            "split_targets": 0.20,
            "immune_count": -5.5,
            "accuracy_mean": 0.50,
            "priority_count": 0.25,
            "protect_count": -0.35,
            "helping_hand_count": -0.15,
            "fakeout_count": 0.75,
            "speed_control_count": 0.60,
            "redirection_count": 0.10,
            "setup_count": -0.20,
            "pivot_count": 0.25,
            "switch_count": -0.45,
            "double_switch": -2.2,
            "pass_count": -2.0,
            "recoil_low_hp": -1.5,
            "overkill": -0.9,
            "partner_damage_risk": -3.0,
            "opp_pressure": -0.60,
            "support_without_attack": -1.0,
            "message_agreement": 0.60,
            "message_conflict": -0.70,
            "my_left_hp": 0.15,
            "my_right_hp": 0.15,
            "opp_left_hp": -0.15,
            "opp_right_hp": -0.15,
            "turn_norm": -0.20,
            "forced_switch_left": -0.10,
            "forced_switch_right": -0.10,
        }
        for i, name in enumerate(self.feature_names):
            self.weights[i] = priors.get(name, 0.0)

    def predict(self, features: np.ndarray) -> float:
        return math.tanh(float(np.dot(self.weights, features)) / 8.0)

    def update_td(
        self, battle_key: str, reward: float, next_best_value: float, terminal: bool
    ) -> None:
        if not self.config.use_online_learning or battle_key not in self.previous_features:
            return
        features = self.previous_features[battle_key]
        old_value = self.previous_value.get(battle_key, self.predict(features))
        target = float(
            np.clip(
                reward if terminal else reward + self.config.td_gamma * next_best_value, -1.0, 1.0
            )
        )
        update = np.clip(self.config.learning_rate * (target - old_value) * features, -0.05, 0.05)
        self.weights = np.clip(
            self.weights + update, -self.config.weight_clip, self.config.weight_clip
        )

    def remember(self, battle_key: str, features: np.ndarray, value: float) -> None:
        self.previous_features[battle_key] = features.copy()
        self.previous_value[battle_key] = value

    def save(self) -> None:
        ensure_parent_dir(self.config.model_path)
        path = os.path.abspath(self.config.model_path)
        lock = _MODEL_FILE_LOCKS.setdefault(path, threading.Lock())
        payload = {
            "model_version": self.MODEL_VERSION,
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
        }
        last_error: Optional[Exception] = None
        for attempt in range(6):
            tmp_path = f"{path}.{os.getpid()}.{id(self)}.{attempt}.tmp"
            try:
                with lock:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))
        logger.info(
            f"[model] action=save status=failed path={self.config.model_path} error={last_error}"
        )

    def load(self) -> None:
        if not os.path.exists(self.config.model_path):
            return
        try:
            with open(self.config.model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model_version") != self.MODEL_VERSION:
                logger.info(f"Ignored incompatible model version: {self.config.model_path}.")
                return
            if data.get("feature_names", []) == self.feature_names:
                weights = data.get("weights", [])
                if len(weights) == len(self.weights):
                    self.weights = np.array(weights, dtype=np.float32)
                    logger.info(f"Loaded model from {self.config.model_path}.")
        except Exception as exc:
            logger.info(f"Could not load model: {exc}")


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
