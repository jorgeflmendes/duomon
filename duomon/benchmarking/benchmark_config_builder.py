from __future__ import annotations

from ..core.env import env_enabled, env_float, env_int, env_text
from .benchmark_cli import *


def build_benchmark_config_from_env() -> AgentConfig:

    benchmark_online_learning = env_text("DUOMON_BENCHMARK_ONLINE_LEARNING", "0").strip() != "0"
    default_parallelism = "1" if benchmark_online_learning else "32"
    benchmark_parallelism = max(1, env_int("DUOMON_PARALLEL_BATTLES", default_parallelism))
    config = AgentConfig(
        battle_format=env_text("DUOMON_BATTLE_FORMAT", "gen9multirandombattle"),
        eval_battles=env_int("DUOMON_BATTLES_PER_OPPONENT", "50"),
        max_concurrent_battles=benchmark_parallelism,
        seed=env_int("DUOMON_SEED", "42"),
        model_path="multi_independent_value_model.json",
        replay_path="multi_independent_replays.jsonl",
        metrics_path="multi_independent_metrics.jsonl",
        ctde_joint_reranker_path=env_text(
            "DUOMON_CTDE_JOINT_RERANKER_PATH", "ctde_joint_reranker.json"
        ),
        simpleheuristics_ctde_joint_reranker_path=env_text(
            "DUOMON_SIMPLE_CTDE_JOINT_RERANKER_PATH", ""
        ),
        abyssal_ctde_joint_reranker_path=env_text("DUOMON_ABYSSAL_CTDE_JOINT_RERANKER_PATH", ""),
        typeaware_ctde_joint_reranker_path=env_text("DUOMON_TYPEAWARE_CTDE_JOINT_RERANKER_PATH", ""),
        ctde_joint_dataset_path=env_text(
            "DUOMON_CTDE_JOINT_DATASET_PATH", "ctde_joint_examples.jsonl"
        ),
        ctde_outcomes_path=env_text("DUOMON_CTDE_OUTCOMES_PATH", "ctde_outcomes.jsonl"),
        ctde_split_path=env_text("DUOMON_CTDE_SPLIT_PATH", "ctde_split.json"),
        ctde_validation_fraction=env_float("DUOMON_CTDE_VALIDATION_FRACTION", "0.20"),
        ally_policy=env_text("DUOMON_ALLY_POLICY", ""),
        simpleheuristics_ally_policy=env_text("DUOMON_SIMPLE_ALLY_POLICY", ""),
        abyssal_ally_policy=env_text("DUOMON_ABYSSAL_ALLY_POLICY", ""),
        fixed_ally_team_enabled=env_enabled("DUOMON_FIXED_ALLY_TEAMS", "0"),
        fixed_ally_battle_format=env_text(
            "DUOMON_FIXED_ALLY_BATTLE_FORMAT",
            "gen9duomonfixedalliesmultirandombattle",
        ),
        fixed_ally_team_p1_path=env_text(
            "DUOMON_FIXED_ALLY_TEAM_P1_PATH", "teams/duomon_ally_p1.txt"
        ),
        fixed_ally_team_p3_path=env_text(
            "DUOMON_FIXED_ALLY_TEAM_P3_PATH", "teams/duomon_ally_p3.txt"
        ),
        fixed_ally_team_p1=env_text("DUOMON_FIXED_ALLY_TEAM_P1", ""),
        fixed_ally_team_p3=env_text("DUOMON_FIXED_ALLY_TEAM_P3", ""),
        mirror_opponent_team_enabled=env_enabled("DUOMON_MIRROR_OPPONENT_TEAM_ENABLED", "0"),
        fixed_ally_optimize_leads_enabled=env_enabled("DUOMON_FIXED_ALLY_OPTIMIZE_LEADS", "0"),
        benchmark_online_learning=benchmark_online_learning,
        learned_value_weight=float(os.environ.get("DUOMON_LEARNED_VALUE_WEIGHT", "0.0")),
        abyssal_learned_value_weight=float(
            os.environ.get(
                "DUOMON_ABYSSAL_LEARNED_VALUE_WEIGHT",
                os.environ.get("DUOMON_LEARNED_VALUE_WEIGHT", "0.0"),
            )
        ),
        redirection_tactical_weight=float(
            os.environ.get("DUOMON_REDIRECTION_TACTICAL_WEIGHT", "0.55")
        ),
        status_control_tactical_weight=float(
            os.environ.get("DUOMON_STATUS_CONTROL_TACTICAL_WEIGHT", "0.95")
        ),
        helping_hand_tactical_weight=float(
            os.environ.get("DUOMON_HELPING_HAND_TACTICAL_WEIGHT", "0.80")
        ),
        hard_benchmark_pair_weight=float(os.environ.get("DUOMON_HARD_BENCHMARK_PAIR_WEIGHT", "0.0")),
        simpleheuristics_hard_benchmark_pair_weight=float(
            os.environ.get("DUOMON_SIMPLE_HARD_BENCHMARK_PAIR_WEIGHT", "-1.0")
        ),
        abyssal_hard_benchmark_pair_weight=float(
            os.environ.get("DUOMON_ABYSSAL_HARD_BENCHMARK_PAIR_WEIGHT", "-1.0")
        ),
        hard_benchmark_protect_weight=float(
            os.environ.get("DUOMON_HARD_BENCHMARK_PROTECT_WEIGHT", "0.0")
        ),
        simpleheuristics_hard_benchmark_protect_weight=float(
            os.environ.get(
                "DUOMON_SIMPLE_HARD_BENCHMARK_PROTECT_WEIGHT",
                os.environ.get("DUOMON_HARD_BENCHMARK_PROTECT_WEIGHT", "-1.0"),
            )
        ),
        abyssal_hard_benchmark_protect_weight=float(
            os.environ.get(
                "DUOMON_ABYSSAL_HARD_BENCHMARK_PROTECT_WEIGHT",
                os.environ.get("DUOMON_HARD_BENCHMARK_PROTECT_WEIGHT", "-1.0"),
            )
        ),
        hard_benchmark_survival_weight=float(
            os.environ.get("DUOMON_HARD_BENCHMARK_SURVIVAL_WEIGHT", "0.75")
        ),
        decision_gate_enabled=os.environ.get("DUOMON_DECISION_GATE_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
        tera_gate_enabled=os.environ.get("DUOMON_TERA_GATE_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
        tera_turn1_penalty=float(os.environ.get("DUOMON_TERA_TURN1_PENALTY", "3.75")),
        tera_reject_penalty=float(os.environ.get("DUOMON_TERA_REJECT_PENALTY", "4.50")),
        tera_min_offensive_gain=float(os.environ.get("DUOMON_TERA_MIN_OFFENSIVE_GAIN", "0.18")),
        risk_gate_enabled=os.environ.get("DUOMON_RISK_GATE_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
        risk_gate_protect_bonus=float(os.environ.get("DUOMON_RISK_GATE_PROTECT_BONUS", "2.30")),
        risk_gate_attack_penalty=float(os.environ.get("DUOMON_RISK_GATE_ATTACK_PENALTY", "1.75")),
        risk_gate_no_protect_penalty_scale=float(
            os.environ.get("DUOMON_RISK_GATE_NO_PROTECT_PENALTY_SCALE", "0.40")
        ),
        simpleheuristics_risk_gate_no_protect_penalty_scale=float(
            os.environ.get("DUOMON_SIMPLE_RISK_GATE_NO_PROTECT_PENALTY_SCALE", "-1.0")
        ),
        abyssal_risk_gate_no_protect_penalty_scale=float(
            os.environ.get("DUOMON_ABYSSAL_RISK_GATE_NO_PROTECT_PENALTY_SCALE", "0.25")
        ),
        protect_filter_predicted_danger_enabled=os.environ.get(
            "DUOMON_PROTECT_FILTER_PREDICTED_DANGER_ENABLED", "1"
        )
        .strip()
        .lower()
        not in {"0", "false", "no"},
        accuracy_risk_weight=float(os.environ.get("DUOMON_ACCURACY_RISK_WEIGHT", "2.40")),
        unknown_threat_risk_floor_weight=float(
            os.environ.get("DUOMON_UNKNOWN_THREAT_RISK_FLOOR_WEIGHT", "1.0")
        ),
        low_hp_attack_risk_penalty=float(
            os.environ.get("DUOMON_LOW_HP_ATTACK_RISK_PENALTY", "1.15")
        ),
        early_survival_mode_weight=float(
            os.environ.get("DUOMON_EARLY_SURVIVAL_MODE_WEIGHT", "1.15")
        ),
        decision_diagnostics_top_k=int(os.environ.get("DUOMON_DECISION_DIAGNOSTICS_TOP_K", "8")),
        spread_pressure_weight=float(os.environ.get("DUOMON_SPREAD_PRESSURE_WEIGHT", "0.55")),
        simpleheuristics_spread_pressure_weight=float(
            os.environ.get(
                "DUOMON_SIMPLE_SPREAD_PRESSURE_WEIGHT",
                os.environ.get("DUOMON_SPREAD_PRESSURE_WEIGHT", "0.55"),
            )
        ),
        abyssal_spread_pressure_weight=float(
            os.environ.get(
                "DUOMON_ABYSSAL_SPREAD_PRESSURE_WEIGHT",
                os.environ.get("DUOMON_SPREAD_PRESSURE_WEIGHT", "0.55"),
            )
        ),
        simpleheuristics_communication_enabled=os.environ.get(
            "DUOMON_SIMPLE_COMMUNICATION_ENABLED", ""
        ),
        abyssal_communication_enabled=os.environ.get("DUOMON_ABYSSAL_COMMUNICATION_ENABLED", ""),
        typeaware_communication_enabled=os.environ.get("DUOMON_TYPEAWARE_COMMUNICATION_ENABLED", ""),
        blind_opening_policy=os.environ.get("DUOMON_BLIND_OPENING_POLICY", "focus_opp1")
        .strip()
        .lower(),
        use_shared_joint_selector=os.environ.get("DUOMON_USE_SHARED_JOINT_SELECTOR", "1")
        .strip()
        .lower()
        not in {"0", "false", "no"},
        shared_joint_combo_prebonus_weight=float(
            os.environ.get("DUOMON_SHARED_JOINT_COMBO_PREBONUS_WEIGHT", "0.0")
        ),
        role_focus_weight=float(os.environ.get("DUOMON_ROLE_FOCUS_WEIGHT", "1.0")),
        abyssal_role_focus_weight=float(os.environ.get("DUOMON_ABYSSAL_ROLE_FOCUS_WEIGHT", "1.0")),
        shared_split_target_weight=float(os.environ.get("DUOMON_SHARED_SPLIT_TARGET_WEIGHT", "0.0")),
        typeaware_shared_split_target_weight=float(
            os.environ.get("DUOMON_TYPEAWARE_SHARED_SPLIT_TARGET_WEIGHT", "-1.0")
        ),
        simpleheuristics_shared_split_target_weight=float(
            os.environ.get("DUOMON_SIMPLE_SHARED_SPLIT_TARGET_WEIGHT", "-1.0")
        ),
        abyssal_shared_split_target_weight=float(
            os.environ.get("DUOMON_ABYSSAL_SHARED_SPLIT_TARGET_WEIGHT", "-1.0")
        ),
        shared_partial_split_penalty_weight=float(
            os.environ.get("DUOMON_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT", "0.0")
        ),
        simpleheuristics_shared_partial_split_penalty_weight=float(
            os.environ.get(
                "DUOMON_SIMPLE_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT",
                os.environ.get("DUOMON_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT", "0.0"),
            )
        ),
        abyssal_shared_partial_split_penalty_weight=float(
            os.environ.get(
                "DUOMON_ABYSSAL_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT",
                os.environ.get("DUOMON_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT", "0.0"),
            )
        ),
        opponent_response_top_k=int(os.environ.get("DUOMON_OPPONENT_RESPONSE_TOP_K", "1")),
        opponent_response_softmax_temp=float(
            os.environ.get("DUOMON_OPPONENT_RESPONSE_SOFTMAX_TEMP", "1.0")
        ),
        opponent_response_margin=float(os.environ.get("DUOMON_OPPONENT_RESPONSE_MARGIN", "0.0")),
        simpleheuristics_exact_response_score=os.environ.get(
            "DUOMON_SIMPLE_EXACT_RESPONSE_SCORE", "0"
        )
        .strip()
        .lower()
        in {"1", "true", "yes"},
        self_damage_estimate_scale=float(os.environ.get("DUOMON_SELF_DAMAGE_ESTIMATE_SCALE", "1.0")),
        opponent_damage_estimate_scale=float(
            os.environ.get("DUOMON_OPPONENT_DAMAGE_ESTIMATE_SCALE", "1.0")
        ),
        simpleheuristics_self_damage_estimate_scale=float(
            os.environ.get(
                "DUOMON_SIMPLE_SELF_DAMAGE_ESTIMATE_SCALE",
                os.environ.get("DUOMON_SELF_DAMAGE_ESTIMATE_SCALE", "1.0"),
            )
        ),
        simpleheuristics_opponent_damage_estimate_scale=float(
            os.environ.get(
                "DUOMON_SIMPLE_OPPONENT_DAMAGE_ESTIMATE_SCALE",
                os.environ.get("DUOMON_OPPONENT_DAMAGE_ESTIMATE_SCALE", "1.0"),
            )
        ),
        abyssal_self_damage_estimate_scale=float(
            os.environ.get(
                "DUOMON_ABYSSAL_SELF_DAMAGE_ESTIMATE_SCALE",
                os.environ.get("DUOMON_SELF_DAMAGE_ESTIMATE_SCALE", "1.0"),
            )
        ),
        abyssal_opponent_damage_estimate_scale=float(
            os.environ.get(
                "DUOMON_ABYSSAL_OPPONENT_DAMAGE_ESTIMATE_SCALE",
                os.environ.get("DUOMON_OPPONENT_DAMAGE_ESTIMATE_SCALE", "1.0"),
            )
        ),
        use_ctde_joint_reranker=os.environ.get("DUOMON_USE_CTDE_JOINT_RERANKER", "0").strip().lower()
        in {"1", "true", "yes"},
        log_ctde_joint_examples=os.environ.get("DUOMON_LOG_CTDE_JOINT_EXAMPLES", "1").strip().lower()
        not in {"0", "false", "no"},
        ctde_joint_reranker_weight=float(os.environ.get("DUOMON_CTDE_JOINT_RERANKER_WEIGHT", "0.0")),
        ctde_joint_reranker_clip=float(os.environ.get("DUOMON_CTDE_JOINT_RERANKER_CLIP", "0.75")),
        simpleheuristics_use_ctde_joint_reranker=os.environ.get(
            "DUOMON_SIMPLE_USE_CTDE_JOINT_RERANKER", "0"
        )
        .strip()
        .lower()
        in {"1", "true", "yes"},
        abyssal_use_ctde_joint_reranker=os.environ.get("DUOMON_ABYSSAL_USE_CTDE_JOINT_RERANKER", "0")
        .strip()
        .lower()
        in {"1", "true", "yes"},
        typeaware_use_ctde_joint_reranker=os.environ.get(
            "DUOMON_TYPEAWARE_USE_CTDE_JOINT_RERANKER", "0"
        )
        .strip()
        .lower()
        in {"1", "true", "yes"},
        simpleheuristics_ctde_joint_reranker_weight=float(
            os.environ.get(
                "DUOMON_SIMPLE_CTDE_JOINT_RERANKER_WEIGHT",
                os.environ.get("DUOMON_CTDE_JOINT_RERANKER_WEIGHT", "0.0"),
            )
        ),
        abyssal_ctde_joint_reranker_weight=float(
            os.environ.get(
                "DUOMON_ABYSSAL_CTDE_JOINT_RERANKER_WEIGHT",
                os.environ.get("DUOMON_CTDE_JOINT_RERANKER_WEIGHT", "0.0"),
            )
        ),
        typeaware_ctde_joint_reranker_weight=float(
            os.environ.get(
                "DUOMON_TYPEAWARE_CTDE_JOINT_RERANKER_WEIGHT",
                os.environ.get("DUOMON_CTDE_JOINT_RERANKER_WEIGHT", "0.0"),
            )
        ),
        use_joint_plan_barrier=os.environ.get("DUOMON_USE_JOINT_PLAN_BARRIER", "0").strip().lower()
        not in {"0", "false", "no"},
        communication_enabled=os.environ.get("DUOMON_COMMUNICATION_ENABLED", "1").strip().lower()
        not in {"0", "false", "no"},
        communication_type=os.environ.get("DUOMON_COMMUNICATION_TYPE", "structured").strip().lower(),
        communication_ablation_mode=os.environ.get("DUOMON_COMMUNICATION_ABLATION_MODE", "normal")
        .strip()
        .lower(),
        communication_use_gate=os.environ.get("DUOMON_COMMUNICATION_USE_GATE", "0").strip().lower()
        in {"1", "true", "yes"},
        communication_gate_threshold=float(
            os.environ.get("DUOMON_COMMUNICATION_GATE_THRESHOLD", "0.0")
        ),
        communication_dropout_prob=float(os.environ.get("DUOMON_COMMUNICATION_DROPOUT_PROB", "0.0")),
        communication_noise_std=float(os.environ.get("DUOMON_COMMUNICATION_NOISE_STD", "0.0")),
        communication_delay_steps=int(os.environ.get("DUOMON_COMMUNICATION_DELAY_STEPS", "0")),
        communication_critic_uses_messages=os.environ.get(
            "DUOMON_COMMUNICATION_CRITIC_USES_MESSAGES", "1"
        )
        .strip()
        .lower()
        not in {"0", "false", "no"},
        communication_zero_agent=os.environ.get("DUOMON_COMMUNICATION_ZERO_AGENT", ""),
        communication_solo_claims_enabled=os.environ.get(
            "DUOMON_COMMUNICATION_SOLO_CLAIMS_ENABLED", "0"
        )
        .strip()
        .lower()
        in {"1", "true", "yes"},
        communication_solo_claim_penalty=float(
            os.environ.get("DUOMON_COMMUNICATION_SOLO_CLAIM_PENALTY", "1.0")
        ),
        transformer_action_prior_enabled=os.environ.get(
            "DUOMON_TRANSFORMER_PRIOR_ENABLED", "0"
        )
        .strip()
        .lower()
        in {"1", "true", "yes"},
        transformer_action_prior_run_dir=os.environ.get(
            "DUOMON_TRANSFORMER_PRIOR_RUN_DIR",
            "outputs/transformer_training/battle_transformer_v1",
        ),
        transformer_action_prior_weight=float(
            os.environ.get("DUOMON_TRANSFORMER_PRIOR_WEIGHT", "0.0")
        ),
        transformer_action_prior_clip=float(
            os.environ.get("DUOMON_TRANSFORMER_PRIOR_CLIP", "2.0")
        ),
        transformer_action_prior_device=os.environ.get(
            "DUOMON_TRANSFORMER_PRIOR_DEVICE", "cpu"
        ),
        simpleheuristics_transformer_action_prior_run_dir=os.environ.get(
            "DUOMON_SIMPLE_TRANSFORMER_PRIOR_RUN_DIR", ""
        ),
        abyssal_transformer_action_prior_run_dir=os.environ.get(
            "DUOMON_ABYSSAL_TRANSFORMER_PRIOR_RUN_DIR", ""
        ),
        typeaware_transformer_action_prior_run_dir=os.environ.get(
            "DUOMON_TYPEAWARE_TRANSFORMER_PRIOR_RUN_DIR", ""
        ),
        simpleheuristics_transformer_action_prior_weight=float(
            os.environ.get("DUOMON_SIMPLE_TRANSFORMER_PRIOR_WEIGHT", "-1.0")
        ),
        abyssal_transformer_action_prior_weight=float(
            os.environ.get("DUOMON_ABYSSAL_TRANSFORMER_PRIOR_WEIGHT", "-1.0")
        ),
        typeaware_transformer_action_prior_weight=float(
            os.environ.get("DUOMON_TYPEAWARE_TRANSFORMER_PRIOR_WEIGHT", "-1.0")
        ),
        shared_joint_top_k=int(os.environ.get("DUOMON_SHARED_JOINT_TOP_K", "12")),
        top_k_slot_actions=int(os.environ.get("DUOMON_TOP_K_SLOT_ACTIONS", "10")),
        simpleheuristics_use_joint_plan_barrier=os.environ.get(
            "DUOMON_SIMPLE_USE_JOINT_PLAN_BARRIER", "0"
        )
        .strip()
        .lower()
        not in {"0", "false", "no"},
        partner_intent_wait_seconds=float(
            os.environ.get("DUOMON_PARTNER_INTENT_WAIT_SECONDS", "0.45")
        ),
        simpleheuristics_partner_intent_wait_seconds=float(
            os.environ.get(
                "DUOMON_SIMPLE_PARTNER_INTENT_WAIT_SECONDS",
                os.environ.get("DUOMON_PARTNER_INTENT_WAIT_SECONDS", "0.45"),
            )
        ),
        abyssal_partner_intent_wait_seconds=float(
            os.environ.get(
                "DUOMON_ABYSSAL_PARTNER_INTENT_WAIT_SECONDS",
                os.environ.get("DUOMON_PARTNER_INTENT_WAIT_SECONDS", "0.45"),
            )
        ),
        joint_plan_follow_wait_seconds=float(
            os.environ.get("DUOMON_JOINT_PLAN_FOLLOW_WAIT_SECONDS", "0.08")
        ),
        early_state_sync_wait_seconds=float(
            os.environ.get("DUOMON_EARLY_STATE_SYNC_WAIT_SECONDS", "0.18")
        ),
        simpleheuristics_early_state_sync_wait_seconds=float(
            os.environ.get(
                "DUOMON_SIMPLE_EARLY_STATE_SYNC_WAIT_SECONDS",
                os.environ.get("DUOMON_EARLY_STATE_SYNC_WAIT_SECONDS", "0.18"),
            )
        ),
        abyssal_early_state_sync_wait_seconds=float(
            os.environ.get(
                "DUOMON_ABYSSAL_EARLY_STATE_SYNC_WAIT_SECONDS",
                os.environ.get("DUOMON_EARLY_STATE_SYNC_WAIT_SECONDS", "0.18"),
            )
        ),
        allow_voluntary_switches=os.environ.get("DUOMON_ALLOW_VOLUNTARY_SWITCHES", "1")
        .strip()
        .lower()
        not in {"0", "false", "no"},
        simpleheuristics_disable_voluntary_switches=os.environ.get(
            "DUOMON_SIMPLE_DISABLE_VOLUNTARY_SWITCHES", "1"
        )
        .strip()
        .lower()
        not in {"0", "false", "no"},
        profiling_enabled=env_enabled("DUOMON_PROFILING_ENABLED", "0"),
        artifact_root=env_text("DUOMON_ARTIFACT_ROOT", env_text("DUOMON_ARTIFACT_DIR", "artifacts")),
        run_name=env_text("DUOMON_RUN_NAME", "benchmark"),
        run_id=env_text("DUOMON_RUN_ID", ""),
        replay_retention=env_text("DUOMON_REPLAY_RETENTION", "all").strip().lower(),
        compress_selected_battles=env_enabled("DUOMON_COMPRESS_SELECTED_BATTLES", "1"),
        selected_battle_compression=env_text("DUOMON_SELECTED_BATTLE_COMPRESSION", "gzip")
        .strip()
        .lower(),
        max_full_battles_per_run=max(0, env_int("DUOMON_MAX_FULL_BATTLES_PER_RUN", "500")),
        keep_parallel_replay_shards=env_enabled("DUOMON_KEEP_PARALLEL_REPLAY_SHARDS", "0"),
    )
    profile_name = os.environ.get("DUOMON_PROFILE", "ctde_mlp").strip().lower()
    if (
        profile_name in {"ctde_mlp", "league"}
        and "DUOMON_FIXED_ALLY_OPTIMIZE_LEADS" not in os.environ
    ):
        config.fixed_ally_optimize_leads_enabled = True
    return config


__all__ = ["build_benchmark_config_from_env"]
