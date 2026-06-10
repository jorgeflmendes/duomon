from __future__ import annotations

from ..core.env import optional_bool
from .benchmark_context import *


def _phase_allow_voluntary_switches(base_config: AgentConfig, opponent_kind: str) -> bool:
    if opponent_kind == "simpleheuristics" and bool(
        getattr(base_config, "simpleheuristics_disable_voluntary_switches", True)
    ):
        return False
    return bool(getattr(base_config, "allow_voluntary_switches", True))


def _phase_use_joint_plan_barrier(base_config: AgentConfig, opponent_kind: str) -> bool:
    if bool(getattr(base_config, "use_joint_plan_barrier", False)):
        return True
    return opponent_kind == "simpleheuristics" and bool(
        getattr(base_config, "simpleheuristics_use_joint_plan_barrier", False)
    )


def _phase_role_focus_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(base_config, "abyssal_role_focus_weight", base_config.role_focus_weight) or 1.0
        )
    return float(getattr(base_config, "role_focus_weight", 1.0) or 1.0)


def _phase_learned_value_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_learned_value_weight",
                base_config.learned_value_weight,
            )
            or 0.0
        )
    return float(getattr(base_config, "learned_value_weight", 0.0) or 0.0)


def _phase_risk_gate_no_protect_penalty_scale(
    base_config: AgentConfig, opponent_kind: str
) -> float:
    return _phase_nonnegative_override(
        base_config,
        "risk_gate_no_protect_penalty_scale",
        opponent_kind,
        {
            "simpleheuristics": "simpleheuristics_risk_gate_no_protect_penalty_scale",
            "abyssal": "abyssal_risk_gate_no_protect_penalty_scale",
        },
        default_value=0.40,
    )


def _phase_nonnegative_override(
    base_config: AgentConfig,
    base_field: str,
    opponent_kind: str,
    field_by_opponent: Dict[str, str],
    default_value: float = 0.0,
) -> float:
    default = float(getattr(base_config, base_field, default_value) or default_value)
    field = field_by_opponent.get(opponent_kind)
    if not field:
        return default
    raw = getattr(base_config, field, -1.0)
    value = float(raw if raw is not None else -1.0)
    return default if value < 0.0 else value


def _phase_ally_policy(base_config: AgentConfig, opponent_kind: str) -> str:
    if opponent_kind == "abyssal":
        return (
            str(
                getattr(base_config, "abyssal_ally_policy", "")
                or getattr(base_config, "ally_policy", "")
                or ""
            )
            .strip()
            .lower()
        )
    if opponent_kind == "simpleheuristics":
        return (
            str(
                getattr(base_config, "simpleheuristics_ally_policy", "")
                or getattr(base_config, "ally_policy", "")
                or ""
            )
            .strip()
            .lower()
        )
    return str(getattr(base_config, "ally_policy", "") or "").strip().lower()


def _phase_communication_enabled(base_config: AgentConfig, opponent_kind: str) -> bool:
    field_by_opponent = {
        "simpleheuristics": "simpleheuristics_communication_enabled",
        "abyssal": "abyssal_communication_enabled",
        "typeaware": "typeaware_communication_enabled",
    }
    field = field_by_opponent.get(opponent_kind, "")
    raw = getattr(base_config, field, "") if field else ""
    return optional_bool(raw, bool(getattr(base_config, "communication_enabled", True)))


def _phase_use_ctde_joint_reranker(base_config: AgentConfig, opponent_kind: str) -> bool:
    if bool(getattr(base_config, "use_ctde_joint_reranker", False)):
        return True
    if opponent_kind == "abyssal":
        return bool(getattr(base_config, "abyssal_use_ctde_joint_reranker", False))
    if opponent_kind == "simpleheuristics":
        return bool(getattr(base_config, "simpleheuristics_use_ctde_joint_reranker", False))
    if opponent_kind == "typeaware":
        return bool(getattr(base_config, "typeaware_use_ctde_joint_reranker", False))
    return False


def _phase_ctde_joint_reranker_weight(base_config: AgentConfig, opponent_kind: str) -> float:
    if opponent_kind == "abyssal":
        return float(
            getattr(
                base_config,
                "abyssal_ctde_joint_reranker_weight",
                base_config.ctde_joint_reranker_weight,
            )
            or 0.0
        )
    if opponent_kind == "simpleheuristics":
        return float(
            getattr(
                base_config,
                "simpleheuristics_ctde_joint_reranker_weight",
                base_config.ctde_joint_reranker_weight,
            )
            or 0.0
        )
    if opponent_kind == "typeaware":
        return float(
            getattr(
                base_config,
                "typeaware_ctde_joint_reranker_weight",
                base_config.ctde_joint_reranker_weight,
            )
            or 0.0
        )
    return float(getattr(base_config, "ctde_joint_reranker_weight", 0.0) or 0.0)


def _phase_ctde_joint_reranker_path(base_config: AgentConfig, opponent_kind: str) -> str:
    if opponent_kind == "abyssal":
        path = str(getattr(base_config, "abyssal_ctde_joint_reranker_path", "") or "")
        return path or base_config.ctde_joint_reranker_path
    if opponent_kind == "simpleheuristics":
        path = str(getattr(base_config, "simpleheuristics_ctde_joint_reranker_path", "") or "")
        return path or base_config.ctde_joint_reranker_path
    if opponent_kind == "typeaware":
        path = str(getattr(base_config, "typeaware_ctde_joint_reranker_path", "") or "")
        return path or base_config.ctde_joint_reranker_path
    return base_config.ctde_joint_reranker_path


def _apply_post_profile_env_overrides(config: AgentConfig) -> None:

    bool_fields = {
        "DUOMON_USE_CTDE_JOINT_RERANKER": "use_ctde_joint_reranker",
        "DUOMON_SIMPLE_USE_CTDE_JOINT_RERANKER": "simpleheuristics_use_ctde_joint_reranker",
        "DUOMON_ABYSSAL_USE_CTDE_JOINT_RERANKER": "abyssal_use_ctde_joint_reranker",
        "DUOMON_TYPEAWARE_USE_CTDE_JOINT_RERANKER": "typeaware_use_ctde_joint_reranker",
        "DUOMON_COMMUNICATION_ENABLED": "communication_enabled",
        "DUOMON_COMMUNICATION_USE_GATE": "communication_use_gate",
        "DUOMON_COMMUNICATION_CRITIC_USES_MESSAGES": "communication_critic_uses_messages",
        "DUOMON_COMMUNICATION_SOLO_CLAIMS_ENABLED": "communication_solo_claims_enabled",
    }
    for env_name, field_name in bool_fields.items():
        if env_name in os.environ:
            setattr(
                config,
                field_name,
                os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes"},
            )

    float_fields = {
        "DUOMON_CTDE_JOINT_RERANKER_WEIGHT": "ctde_joint_reranker_weight",
        "DUOMON_SIMPLE_CTDE_JOINT_RERANKER_WEIGHT": "simpleheuristics_ctde_joint_reranker_weight",
        "DUOMON_ABYSSAL_CTDE_JOINT_RERANKER_WEIGHT": "abyssal_ctde_joint_reranker_weight",
        "DUOMON_TYPEAWARE_CTDE_JOINT_RERANKER_WEIGHT": "typeaware_ctde_joint_reranker_weight",
        "DUOMON_CTDE_JOINT_RERANKER_CLIP": "ctde_joint_reranker_clip",
        "DUOMON_SPREAD_PRESSURE_WEIGHT": "spread_pressure_weight",
        "DUOMON_SIMPLE_SPREAD_PRESSURE_WEIGHT": "simpleheuristics_spread_pressure_weight",
        "DUOMON_ABYSSAL_SPREAD_PRESSURE_WEIGHT": "abyssal_spread_pressure_weight",
        "DUOMON_ROLE_FOCUS_WEIGHT": "role_focus_weight",
        "DUOMON_ABYSSAL_ROLE_FOCUS_WEIGHT": "abyssal_role_focus_weight",
        "DUOMON_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT": "shared_partial_split_penalty_weight",
        "DUOMON_SIMPLE_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT": (
            "simpleheuristics_shared_partial_split_penalty_weight"
        ),
        "DUOMON_ABYSSAL_SHARED_PARTIAL_SPLIT_PENALTY_WEIGHT": (
            "abyssal_shared_partial_split_penalty_weight"
        ),
        "DUOMON_HARD_BENCHMARK_PROTECT_WEIGHT": "hard_benchmark_protect_weight",
        "DUOMON_SIMPLE_HARD_BENCHMARK_PROTECT_WEIGHT": (
            "simpleheuristics_hard_benchmark_protect_weight"
        ),
        "DUOMON_ABYSSAL_HARD_BENCHMARK_PROTECT_WEIGHT": ("abyssal_hard_benchmark_protect_weight"),
        "DUOMON_HARD_BENCHMARK_SURVIVAL_WEIGHT": "hard_benchmark_survival_weight",
        "DUOMON_RISK_GATE_NO_PROTECT_PENALTY_SCALE": "risk_gate_no_protect_penalty_scale",
        "DUOMON_SIMPLE_RISK_GATE_NO_PROTECT_PENALTY_SCALE": (
            "simpleheuristics_risk_gate_no_protect_penalty_scale"
        ),
        "DUOMON_ABYSSAL_RISK_GATE_NO_PROTECT_PENALTY_SCALE": (
            "abyssal_risk_gate_no_protect_penalty_scale"
        ),
        "DUOMON_COMMUNICATION_SOLO_CLAIM_PENALTY": "communication_solo_claim_penalty",
    }
    for env_name, field_name in float_fields.items():
        if env_name in os.environ:
            setattr(config, field_name, float(os.environ[env_name]))

    path_fields = {
        "DUOMON_CTDE_JOINT_RERANKER_PATH": "ctde_joint_reranker_path",
        "DUOMON_SIMPLE_CTDE_JOINT_RERANKER_PATH": "simpleheuristics_ctde_joint_reranker_path",
        "DUOMON_ABYSSAL_CTDE_JOINT_RERANKER_PATH": "abyssal_ctde_joint_reranker_path",
        "DUOMON_TYPEAWARE_CTDE_JOINT_RERANKER_PATH": "typeaware_ctde_joint_reranker_path",
    }
    for env_name, field_name in path_fields.items():
        if env_name in os.environ:
            setattr(config, field_name, training_path(config, os.environ[env_name]))


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
