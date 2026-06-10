from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class AgentConfig:
    battle_format: str = "gen9randomdoublesbattle"
    eval_battles: int = 100
    max_concurrent_battles: int = 1
    seed: int = 42

    output_dir: str = "outputs"
    training_dir: str = "learned_weights"

    model_path: str = "independent_value_model.json"
    replay_path: str = "independent_replays.jsonl"
    metrics_path: str = "independent_metrics.jsonl"
    ctde_joint_reranker_path: str = "ctde_joint_reranker.json"
    simpleheuristics_ctde_joint_reranker_path: str = ""
    abyssal_ctde_joint_reranker_path: str = ""
    typeaware_ctde_joint_reranker_path: str = ""
    ctde_joint_dataset_path: str = "ctde_joint_examples.jsonl"
    ctde_outcomes_path: str = "ctde_outcomes.jsonl"
    ctde_split_path: str = "ctde_split.json"
    ctde_validation_fraction: float = 0.20
    benchmark_type: str = "default"
    ally_policy: str = ""
    simpleheuristics_ally_policy: str = ""
    abyssal_ally_policy: str = ""
    fixed_ally_team_enabled: bool = False
    fixed_ally_battle_format: str = "gen9duomonfixedalliesmultirandombattle"
    fixed_ally_team_p1_path: str = "teams/duomon_ally_p1.txt"
    fixed_ally_team_p3_path: str = "teams/duomon_ally_p3.txt"
    fixed_ally_team_p1: str = ""
    fixed_ally_team_p3: str = ""
    mirror_opponent_team_enabled: bool = False
    fixed_ally_optimize_leads_enabled: bool = False

    use_online_learning: bool = False
    benchmark_online_learning: bool = False
    learned_value_weight: float = 0.0
    abyssal_learned_value_weight: float = 0.0
    learning_rate: float = 0.012
    td_gamma: float = 0.96
    weight_clip: float = 8.0

    top_k_slot_actions: int = 10

    max_repeated_turn_calls: int = 2
    max_battle_turns_soft: int = 80
    log_every_turn: bool = True
    replay_flush_every: int = 64

    allow_voluntary_switches: bool = True
    allow_pivot_moves: bool = True
    use_advanced_policy: bool = True
    learned_value_clip: float = 0.45
    hard_benchmark_learned_value_cap: float = 0.15

    tactical_knowledge_enabled: bool = True
    type_effectiveness_weight: float = 0.38
    ko_pressure_weight: float = 1.05
    protect_tactical_weight: float = 0.82
    fake_out_tactical_weight: float = 0.85
    speed_control_tactical_weight: float = 0.58
    redirection_tactical_weight: float = 0.55
    status_control_tactical_weight: float = 0.95
    helping_hand_tactical_weight: float = 0.80
    switch_tactical_weight: float = 0.85
    spread_pressure_weight: float = 0.55
    simpleheuristics_spread_pressure_weight: float = 0.55
    abyssal_spread_pressure_weight: float = 0.55
    simpleheuristics_communication_enabled: str = ""
    abyssal_communication_enabled: str = ""
    typeaware_communication_enabled: str = ""
    use_shared_joint_selector: bool = True
    use_ctde_joint_reranker: bool = False
    log_ctde_joint_examples: bool = True
    ctde_joint_reranker_weight: float = 0.0
    ctde_joint_reranker_clip: float = 0.75
    simpleheuristics_use_ctde_joint_reranker: bool = False
    abyssal_use_ctde_joint_reranker: bool = False
    typeaware_use_ctde_joint_reranker: bool = False
    simpleheuristics_ctde_joint_reranker_weight: float = 0.0
    abyssal_ctde_joint_reranker_weight: float = 0.0
    typeaware_ctde_joint_reranker_weight: float = 0.0
    shared_joint_top_k: int = 12
    shared_partner_score_weight: float = 0.28
    shared_joint_combo_prebonus_weight: float = 0.0
    role_focus_weight: float = 1.0
    abyssal_role_focus_weight: float = 1.0
    shared_split_target_weight: float = 0.0
    typeaware_shared_split_target_weight: float = -1.0
    simpleheuristics_shared_split_target_weight: float = -1.0
    abyssal_shared_split_target_weight: float = -1.0
    shared_partial_split_penalty_weight: float = 0.0
    simpleheuristics_shared_partial_split_penalty_weight: float = 0.0
    abyssal_shared_partial_split_penalty_weight: float = 0.0
    hard_benchmark_pair_weight: float = 0.0
    simpleheuristics_hard_benchmark_pair_weight: float = -1.0
    abyssal_hard_benchmark_pair_weight: float = -1.0
    hard_benchmark_protect_weight: float = 0.0
    simpleheuristics_hard_benchmark_protect_weight: float = -1.0
    abyssal_hard_benchmark_protect_weight: float = -1.0
    hard_benchmark_survival_weight: float = 0.75
    decision_gate_enabled: bool = True
    tera_gate_enabled: bool = True
    tera_turn1_penalty: float = 3.75
    tera_reject_penalty: float = 4.50
    tera_min_offensive_gain: float = 0.18
    risk_gate_enabled: bool = True
    risk_gate_protect_bonus: float = 2.30
    risk_gate_attack_penalty: float = 1.75
    risk_gate_no_protect_penalty_scale: float = 0.40
    simpleheuristics_risk_gate_no_protect_penalty_scale: float = -1.0
    abyssal_risk_gate_no_protect_penalty_scale: float = 0.25
    protect_filter_predicted_danger_enabled: bool = True
    accuracy_risk_weight: float = 2.40
    unknown_threat_risk_floor_weight: float = 1.0
    low_hp_attack_risk_penalty: float = 1.15
    early_survival_mode_weight: float = 1.15
    decision_diagnostics_top_k: int = 8
    opponent_response_top_k: int = 1
    opponent_response_softmax_temp: float = 1.0
    opponent_response_margin: float = 0.0
    simpleheuristics_exact_response_score: bool = False
    self_damage_estimate_scale: float = 1.0
    opponent_damage_estimate_scale: float = 1.0
    simpleheuristics_self_damage_estimate_scale: float = 1.0
    simpleheuristics_opponent_damage_estimate_scale: float = 1.0
    abyssal_self_damage_estimate_scale: float = 1.0
    abyssal_opponent_damage_estimate_scale: float = 1.0
    blind_opening_policy: str = "focus_opp1"
    use_joint_plan_barrier: bool = False
    simpleheuristics_use_joint_plan_barrier: bool = False
    partner_intent_wait_seconds: float = 0.45
    simpleheuristics_partner_intent_wait_seconds: float = 0.45
    abyssal_partner_intent_wait_seconds: float = 0.45
    joint_plan_follow_wait_seconds: float = 0.08
    early_state_sync_wait_seconds: float = 0.18
    simpleheuristics_early_state_sync_wait_seconds: float = 0.18
    abyssal_early_state_sync_wait_seconds: float = 0.18
    simpleheuristics_disable_voluntary_switches: bool = True

    communication_enabled: bool = True
    communication_type: str = "structured"
    communication_ablation_mode: str = "normal"
    communication_use_gate: bool = False
    communication_gate_threshold: float = 0.0
    communication_dropout_prob: float = 0.0
    communication_noise_std: float = 0.0
    communication_delay_steps: int = 0
    communication_critic_uses_messages: bool = True
    communication_zero_agent: str = ""
    communication_solo_claims_enabled: bool = False
    communication_solo_claim_penalty: float = 1.0
    transformer_action_prior_enabled: bool = False
    transformer_action_prior_run_dir: str = "outputs/transformer_training/battle_transformer_v1"
    transformer_action_prior_weight: float = 0.0
    transformer_action_prior_clip: float = 2.0
    transformer_action_prior_device: str = "cpu"
    simpleheuristics_transformer_action_prior_run_dir: str = ""
    abyssal_transformer_action_prior_run_dir: str = ""
    typeaware_transformer_action_prior_run_dir: str = ""
    simpleheuristics_transformer_action_prior_weight: float = -1.0
    abyssal_transformer_action_prior_weight: float = -1.0
    typeaware_transformer_action_prior_weight: float = -1.0
    per_battle_timeout_seconds: float = 180.0
    inter_battle_sleep_seconds: float = 0.35

    profiling_enabled: bool = False
    artifact_root: str = "artifacts"
    run_name: str = "default"
    run_id: str = ""
    replay_retention: str = "all"
    compress_selected_battles: bool = True
    selected_battle_compression: str = "gzip"
    max_full_battles_per_run: int = 500
    keep_parallel_replay_shards: bool = False

    def __post_init__(self) -> None:
        self.output_dir = os.environ.get("DUOMON_OUTPUT_DIR", self.output_dir)
        self.training_dir = os.environ.get("DUOMON_MODEL_DIR", self.training_dir)
        self.artifact_root = os.environ.get("DUOMON_ARTIFACT_DIR", self.artifact_root)
        self.artifact_root = os.environ.get("DUOMON_ARTIFACT_ROOT", self.artifact_root)
        self.run_name = os.environ.get("DUOMON_RUN_NAME", self.run_name)
        self.run_id = os.environ.get("DUOMON_RUN_ID", self.run_id)
        if not self.run_id:
            self.run_id = time.strftime("%Y%m%d_%H%M%S")

        self.model_path = training_path(self, self.model_path)
        self.ctde_joint_reranker_path = training_path(self, self.ctde_joint_reranker_path)
        self.simpleheuristics_ctde_joint_reranker_path = training_path(
            self, self.simpleheuristics_ctde_joint_reranker_path
        )
        self.abyssal_ctde_joint_reranker_path = training_path(
            self, self.abyssal_ctde_joint_reranker_path
        )
        self.typeaware_ctde_joint_reranker_path = training_path(
            self, self.typeaware_ctde_joint_reranker_path
        )
        self.ctde_joint_dataset_path = training_path(self, self.ctde_joint_dataset_path)
        self.ctde_outcomes_path = training_path(
            self, os.environ.get("DUOMON_CTDE_OUTCOMES_PATH", self.ctde_outcomes_path)
        )
        self.ctde_split_path = training_path(
            self, os.environ.get("DUOMON_CTDE_SPLIT_PATH", self.ctde_split_path)
        )
        self.replay_path = output_path(self, self.replay_path)
        self.metrics_path = output_path(self, self.metrics_path)


def _path_in_dir(path: str, directory: str) -> bool:
    if not path or os.path.isabs(path):
        return True
    norm_path = os.path.normpath(path)
    norm_dir = os.path.normpath(directory)
    return norm_path == norm_dir or norm_path.startswith(norm_dir + os.sep)


def _scoped_path(directory: str, path: str) -> str:
    if not path or os.path.isabs(path) or _path_in_dir(path, directory):
        return path
    return os.path.join(directory, path)


def output_path(config: AgentConfig, path: str) -> str:
    return _scoped_path(config.output_dir, path)


def training_path(config: AgentConfig, path: str) -> str:
    if path and not os.path.isabs(path):
        norm_path = os.path.normpath(path)
        if (
            _path_in_dir(path, config.output_dir)
            or norm_path == "outputs"
            or norm_path.startswith("outputs" + os.sep)
        ):
            return path
    return _scoped_path(config.training_dir, path)


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) if path else ""
    if directory:
        os.makedirs(directory, exist_ok=True)


class AgentMode(str, Enum):
    INDEPENDENT = "independent"


def setup_logger() -> logging.Logger:
    for name in ["poke-env", "websockets", "httpx", "urllib3"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    logger = logging.getLogger("IndependentSOTA")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
