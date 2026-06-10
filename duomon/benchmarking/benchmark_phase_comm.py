from __future__ import annotations

from .benchmark_context import *
from .benchmark_phase_env import *
from .benchmark_setup import _benchmark_logs_ctde_examples, _multi_battle_format


def _phase_transformer_action_prior_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    global_w = float(getattr(base_config, "transformer_action_prior_weight", 0.0) or 0.0)
    field_by_opponent = {
        "simpleheuristics": "simpleheuristics_transformer_action_prior_weight",
        "abyssal": "abyssal_transformer_action_prior_weight",
        "typeaware": "typeaware_transformer_action_prior_weight",
    }
    field = field_by_opponent.get(opponent_kind)
    if not field:
        return global_w
    value = float(getattr(base_config, field, -1.0) or -1.0)
    return global_w if value < 0.0 else value


def _phase_transformer_action_prior_run_dir(base_config: AgentConfig, opponent_kind: str) -> str:
    global_path = str(getattr(base_config, "transformer_action_prior_run_dir", "") or "")
    field_by_opponent = {
        "simpleheuristics": "simpleheuristics_transformer_action_prior_run_dir",
        "abyssal": "abyssal_transformer_action_prior_run_dir",
        "typeaware": "typeaware_transformer_action_prior_run_dir",
    }
    field = field_by_opponent.get(opponent_kind)
    if not field:
        return global_path
    return str(getattr(base_config, field, "") or "") or global_path


def _phase_spread_pressure_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_spread_pressure_weight",
                base_config.spread_pressure_weight,
            )
            or 0.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_spread_pressure_weight",
                base_config.spread_pressure_weight,
            )
            or 0.0
        )
    return float(getattr(base_config, "spread_pressure_weight", 0.0) or 0.0)


def _phase_partial_split_penalty_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_shared_partial_split_penalty_weight",
                base_config.shared_partial_split_penalty_weight,
            )
            or 0.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_shared_partial_split_penalty_weight",
                base_config.shared_partial_split_penalty_weight,
            )
            or 0.0
        )
    return float(getattr(base_config, "shared_partial_split_penalty_weight", 0.0) or 0.0)


def _phase_split_target_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    return _phase_nonnegative_override(
        base_config,
        "shared_split_target_weight",
        opponent_kind,
        {
            "abyssal": "abyssal_shared_split_target_weight",
            "simpleheuristics": "simpleheuristics_shared_split_target_weight",
            "typeaware": "typeaware_shared_split_target_weight",
        },
    )


def _phase_hard_benchmark_protect_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    return _phase_nonnegative_override(
        base_config,
        "hard_benchmark_protect_weight",
        opponent_kind,
        {
            "abyssal": "abyssal_hard_benchmark_protect_weight",
            "simpleheuristics": "simpleheuristics_hard_benchmark_protect_weight",
        },
    )


def _phase_hard_benchmark_pair_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    return _phase_nonnegative_override(
        base_config,
        "hard_benchmark_pair_weight",
        opponent_kind,
        {
            "abyssal": "abyssal_hard_benchmark_pair_weight",
            "simpleheuristics": "simpleheuristics_hard_benchmark_pair_weight",
        },
    )


def _phase_partner_intent_wait(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_partner_intent_wait_seconds",
                base_config.partner_intent_wait_seconds,
            )
            or 0.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_partner_intent_wait_seconds",
                base_config.partner_intent_wait_seconds,
            )
            or 0.0
        )
    return float(getattr(base_config, "partner_intent_wait_seconds", 0.0) or 0.0)


def _phase_early_state_sync_wait(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_early_state_sync_wait_seconds",
                base_config.early_state_sync_wait_seconds,
            )
            or 0.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_early_state_sync_wait_seconds",
                base_config.early_state_sync_wait_seconds,
            )
            or 0.0
        )
    return float(getattr(base_config, "early_state_sync_wait_seconds", 0.0) or 0.0)


def _phase_self_damage_scale(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_self_damage_estimate_scale",
                base_config.self_damage_estimate_scale,
            )
            or 1.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_self_damage_estimate_scale",
                base_config.self_damage_estimate_scale,
            )
            or 1.0
        )
    return float(getattr(base_config, "self_damage_estimate_scale", 1.0) or 1.0)


def _phase_opponent_damage_scale(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_opponent_damage_estimate_scale",
                base_config.opponent_damage_estimate_scale,
            )
            or 1.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_opponent_damage_estimate_scale",
                base_config.opponent_damage_estimate_scale,
            )
            or 1.0
        )
    return float(getattr(base_config, "opponent_damage_estimate_scale", 1.0) or 1.0)


def _common_phase_overrides(
    base_config: AgentConfig,
    opponent_kind: str,
    benchmark_type: str,
    model_path: str,
    replay_path: str,
) -> Dict[str, Any]:
    return {
        "battle_format": _multi_battle_format(base_config),
        "max_concurrent_battles": 1,
        "benchmark_type": benchmark_type,
        "model_path": model_path,
        "replay_path": replay_path,
        "communication_enabled": _phase_communication_enabled(base_config, opponent_kind),
        "ally_policy": _phase_ally_policy(base_config, opponent_kind),
        "use_ctde_joint_reranker": _phase_use_ctde_joint_reranker(base_config, opponent_kind),
        "ctde_joint_reranker_path": _phase_ctde_joint_reranker_path(base_config, opponent_kind),
        "ctde_joint_reranker_weight": _phase_ctde_joint_reranker_weight(base_config, opponent_kind),
        "transformer_action_prior_run_dir": _phase_transformer_action_prior_run_dir(
            base_config, opponent_kind
        ),
        "transformer_action_prior_weight": _phase_transformer_action_prior_weight(
            base_config, opponent_kind
        ),
        "spread_pressure_weight": _phase_spread_pressure_weight(base_config, opponent_kind),
        "shared_split_target_weight": _phase_split_target_weight(base_config, opponent_kind),
        "shared_partial_split_penalty_weight": _phase_partial_split_penalty_weight(
            base_config, opponent_kind
        ),
        "hard_benchmark_pair_weight": _phase_hard_benchmark_pair_weight(base_config, opponent_kind),
        "hard_benchmark_protect_weight": _phase_hard_benchmark_protect_weight(
            base_config, opponent_kind
        ),
        "risk_gate_no_protect_penalty_scale": _phase_risk_gate_no_protect_penalty_scale(
            base_config, opponent_kind
        ),
        "partner_intent_wait_seconds": _phase_partner_intent_wait(base_config, opponent_kind),
        "early_state_sync_wait_seconds": _phase_early_state_sync_wait(base_config, opponent_kind),
        "self_damage_estimate_scale": _phase_self_damage_scale(base_config, opponent_kind),
        "opponent_damage_estimate_scale": _phase_opponent_damage_scale(base_config, opponent_kind),
    }


def _collection_phase_config(
    base_config: AgentConfig,
    opponent_kind: str,
    benchmark_type: str,
    model_path: str,
    replay_path: str,
    online_learning: bool,
) -> AgentConfig:
    overrides = _common_phase_overrides(
        base_config, opponent_kind, benchmark_type, model_path, replay_path
    )
    overrides.update(
        use_online_learning=bool(online_learning),
    )
    return clone_config(base_config, **overrides)


def _eval_phase_config(
    base_config: AgentConfig,
    opponent_kind: str,
    benchmark_type: str,
    model_path: str,
    replay_path: str,
) -> AgentConfig:
    overrides = _common_phase_overrides(
        base_config, opponent_kind, benchmark_type, model_path, replay_path
    )
    overrides.update(
        use_online_learning=base_config.benchmark_online_learning,
        use_joint_plan_barrier=_phase_use_joint_plan_barrier(base_config, opponent_kind),
        role_focus_weight=_phase_role_focus_weight(base_config, opponent_kind),
        learned_value_weight=_phase_learned_value_weight(base_config, opponent_kind),
        log_ctde_joint_examples=_benchmark_logs_ctde_examples(base_config),
        allow_voluntary_switches=_phase_allow_voluntary_switches(base_config, opponent_kind),
    )
    return clone_config(base_config, **overrides)


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
