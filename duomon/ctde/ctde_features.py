from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


CTDE_FEATURE_NAMES = [
    "bias",
    "pair_score_norm",
    "pair_bonus_norm",
    "local_score_norm",
    "partner_score_norm",
    "target_damage",
    "target_damage_sq",
    "ko_count",
    "split_target_value",
    "split_ko_count",
    "split_pressure_count",
    "trade_value",
    "survival_value",
    "hard_benchmark_value",
    "local_rank_norm",
    "partner_rank_norm",
    "same_target",
    "target_slot_0",
    "target_slot_1",
    "reason_joint_ko",
    "reason_joint_focus",
    "reason_joint_split",
    "reason_joint_pressure",
    "benchmark_simpleheuristics",
    "benchmark_abyssal",
    "benchmark_maxpower",
    "benchmark_random",
    "combined_total_damage",
    "combined_slot0_damage",
    "combined_slot1_damage",
    "combined_max_damage",
    "combined_min_damage",
    "combined_damage_balance",
    "ko_slot0",
    "ko_slot1",
    "is_split_pair",
    "weak_split_pair",
    "same_target_overkill",
    "protect_count",
    "support_move_count",
    "spread_like_reason",

    "state_self0_hp",
    "state_self1_hp",
    "state_opp0_hp",
    "state_opp1_hp",
    "state_self0_speed_adv",
    "state_self1_speed_adv",
    "state_weather_rain",
    "state_weather_sun",
    "state_weather_sand",
    "state_weather_snow",
    "state_terrain_electric",
    "state_terrain_grassy",
    "state_terrain_psychic",
    "state_terrain_misty",
    "state_trick_room",
    "state_tailwind_self",
    "state_tailwind_opp",
    "state_screens_self",
    "state_screens_opp",
    "state_hazards_self",
    "state_hazards_opp",
    "state_alive_count_self",
    "state_alive_count_opp",
    "state_min_self_hp",
    "state_team_hp_diff",
    "state_self0_burned",
    "state_self0_paralyzed",
    "state_self1_burned",
    "state_self1_paralyzed",
    "state_opp0_burned",
    "state_opp0_poisoned",
    "state_opp1_burned",
    "state_opp1_poisoned",
    "state_turn_norm",
]










_CTDE_PRIVILEGED_FEATURES = frozenset(
    {
        "benchmark_simpleheuristics",
        "benchmark_abyssal",
        "benchmark_maxpower",
        "benchmark_random",
    }
)



CTDE_POLICY_FEATURE_NAMES = [
    name for name in CTDE_FEATURE_NAMES if name not in _CTDE_PRIVILEGED_FEATURES
]


_CTDE_FEATURE_INDEX = {name: idx for idx, name in enumerate(CTDE_FEATURE_NAMES)}
_CTDE_POLICY_FEATURE_INDEX = {name: idx for idx, name in enumerate(CTDE_POLICY_FEATURE_NAMES)}



_CTDE_INTERACTION_PAIRS = [
    ("pair_score_norm", "combined_total_damage"),
    ("pair_score_norm", "combined_damage_balance"),
    ("pair_bonus_norm", "combined_max_damage"),
    ("local_score_norm", "partner_score_norm"),
    ("local_rank_norm", "partner_rank_norm"),
    ("same_target", "same_target_overkill"),
    ("is_split_pair", "combined_min_damage"),
    ("is_split_pair", "weak_split_pair"),
    ("split_target_value", "split_pressure_count"),
    ("split_target_value", "split_ko_count"),
    ("hard_benchmark_value", "survival_value"),
    ("hard_benchmark_value", "trade_value"),
    ("benchmark_simpleheuristics", "same_target_overkill"),
    ("benchmark_simpleheuristics", "weak_split_pair"),
    ("benchmark_simpleheuristics", "combined_damage_balance"),
    ("benchmark_abyssal", "protect_count"),
    ("benchmark_abyssal", "support_move_count"),
    ("benchmark_abyssal", "hard_benchmark_value"),
    ("benchmark_abyssal", "combined_total_damage"),
    ("benchmark_maxpower", "target_damage"),
]

_CTDE_POLICY_INTERACTION_PAIRS = [
    pair
    for pair in _CTDE_INTERACTION_PAIRS
    if pair[0] not in _CTDE_PRIVILEGED_FEATURES and pair[1] not in _CTDE_PRIVILEGED_FEATURES
]


def _ctde_transformed_feature_names(transform: str = "raw") -> List[str]:
    transform = str(transform or "raw").strip().lower()
    if transform in {"", "raw", "linear"}:
        return list(CTDE_FEATURE_NAMES)
    if transform not in {"compact_nonlinear", "compact_nonlinear_v1"}:
        return list(CTDE_FEATURE_NAMES)
    names = list(CTDE_FEATURE_NAMES)
    names.extend(f"{name}^2" for name in CTDE_FEATURE_NAMES if name != "bias")
    for left, right in _CTDE_INTERACTION_PAIRS:
        names.append(f"{left}*{right}")
    names.extend(
        [
            "abs_local_partner_score_gap",
            "abs_local_partner_rank_gap",
            "damage_focus_minus_balance",
            "split_pressure_per_damage",
        ]
    )
    return names


def _ctde_transform_features(features: Sequence[float], transform: str = "raw") -> np.ndarray:
    base = np.array(features, dtype=np.float32)
    if len(base) != len(CTDE_FEATURE_NAMES):
        return base
    transform = str(transform or "raw").strip().lower()
    if transform in {"", "raw", "linear"}:
        return base
    if transform not in {"compact_nonlinear", "compact_nonlinear_v1"}:
        return base

    parts = [base]
    parts.append(
        np.array(
            [
                base[idx] * base[idx]
                for idx, name in enumerate(CTDE_FEATURE_NAMES)
                if name != "bias"
            ],
            dtype=np.float32,
        )
    )
    interactions = []
    for left, right in _CTDE_INTERACTION_PAIRS:
        left_idx = _CTDE_FEATURE_INDEX.get(left)
        right_idx = _CTDE_FEATURE_INDEX.get(right)
        if left_idx is None or right_idx is None:
            interactions.append(0.0)
        else:
            interactions.append(float(base[left_idx] * base[right_idx]))
    local_score = base[_CTDE_FEATURE_INDEX["local_score_norm"]]
    partner_score = base[_CTDE_FEATURE_INDEX["partner_score_norm"]]
    local_rank = base[_CTDE_FEATURE_INDEX["local_rank_norm"]]
    partner_rank = base[_CTDE_FEATURE_INDEX["partner_rank_norm"]]
    combined_max = base[_CTDE_FEATURE_INDEX["combined_max_damage"]]
    combined_min = base[_CTDE_FEATURE_INDEX["combined_min_damage"]]
    split_pressure = base[_CTDE_FEATURE_INDEX["split_pressure_count"]]
    combined_total = base[_CTDE_FEATURE_INDEX["combined_total_damage"]]
    interactions.extend(
        [
            abs(float(local_score - partner_score)),
            abs(float(local_rank - partner_rank)),
            float(combined_max - combined_min),
            float(split_pressure / max(0.05, combined_total)),
        ]
    )
    parts.append(np.array(interactions, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _int_float_map(raw: Any) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        slot = _safe_int(key)
        if slot is None:
            continue
        try:
            result[slot] = float(value or 0.0)
        except Exception:
            result[slot] = 0.0
    return result


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def ctde_features_from_details(details: Dict[str, Any], benchmark_type: str = "") -> List[float]:
    reason = str(details.get("reason", "") or "")
    target_slot = details.get("target_slot")
    try:
        target_slot_int = int(target_slot) if target_slot is not None else -1
    except Exception:
        target_slot_int = -1

    local_rank = float(details.get("local_rank", 0.0) or 0.0)
    partner_rank = float(details.get("partner_rank", 0.0) or 0.0)
    target_damage = float(details.get("target_damage", 0.0) or 0.0)
    ko_slots = details.get("ko_slots", []) or []
    try:
        ko_count = float(len(ko_slots))
    except Exception:
        ko_count = 0.0

    benchmark = str(benchmark_type or details.get("benchmark_type", "") or "")
    combined_damage = _int_float_map(details.get("combined_damage_by_slot", {}) or {})
    slot0_damage = float(combined_damage.get(0, 0.0) or 0.0)
    slot1_damage = float(combined_damage.get(1, 0.0) or 0.0)
    combined_total = max(0.0, slot0_damage) + max(0.0, slot1_damage)
    combined_max = max(max(0.0, slot0_damage), max(0.0, slot1_damage))
    combined_min = min(max(0.0, slot0_damage), max(0.0, slot1_damage))
    damage_balance = combined_min / max(0.05, combined_max)
    ko_slot_ids = {_safe_int(slot) for slot in ko_slots}
    local_target = _safe_int(details.get("local_target_slot"))
    partner_target = _safe_int(details.get("partner_target_slot"))
    is_split_pair = float(
        local_target in {0, 1} and partner_target in {0, 1} and local_target != partner_target
    )
    split_ko_count = float(details.get("split_ko_count", 0.0) or 0.0)
    split_pressure_count = float(details.get("split_pressure_count", 0.0) or 0.0)
    weak_split = float(is_split_pair and (split_ko_count < 2.0 or split_pressure_count < 2.0))
    same_target_overkill = 0.0
    if local_target in {0, 1} and local_target == partner_target:
        same_target_overkill = max(0.0, target_damage - 1.15)
    local_mid = str(details.get("local_move_id", "") or "").lower()
    partner_mid = str(details.get("partner_move_id", "") or "").lower()
    protect_ids = {
        "protect",
        "detect",
        "spikyshield",
        "kingsshield",
        "banefulbunker",
        "silktrap",
        "burningbulwark",
        "maxguard",
    }
    support_ids = protect_ids | {
        "helpinghand",
        "followme",
        "ragepowder",
        "wideguard",
        "tailwind",
        "trickroom",
        "reflect",
        "lightscreen",
        "auroraveil",
        "stealthrock",
        "spikes",
        "toxicspikes",
        "stickyweb",
    }
    protect_count = float(int(local_mid in protect_ids) + int(partner_mid in protect_ids))
    support_count = float(int(local_mid in support_ids) + int(partner_mid in support_ids))

    state = details.get("state", {}) or {}
    if not isinstance(state, dict):
        state = {}

    def _s(key: str, lo: float = -4.0, hi: float = 4.0, default: float = 0.0) -> float:
        try:
            return _clip(float(state.get(key, default) or default), lo, hi)
        except Exception:
            return default

    return [
        1.0,
        _clip(float(details.get("pair_score", 0.0) or 0.0) / 24.0, -2.0, 2.0),
        _clip(float(details.get("pair_bonus", 0.0) or 0.0) / 16.0, -2.0, 2.0),
        _clip(float(details.get("local_score", 0.0) or 0.0) / 16.0, -2.0, 2.0),
        _clip(float(details.get("partner_score", 0.0) or 0.0) / 16.0, -2.0, 2.0),
        _clip(target_damage, 0.0, 3.0),
        _clip(target_damage * target_damage, 0.0, 5.0),
        _clip(ko_count, 0.0, 2.0),
        _clip(float(details.get("split_target_value", 0.0) or 0.0), -4.0, 6.0),
        _clip(float(details.get("split_ko_count", 0.0) or 0.0), 0.0, 2.0),
        _clip(float(details.get("split_pressure_count", 0.0) or 0.0), 0.0, 2.0),
        _clip(float(details.get("trade_value", 0.0) or 0.0), -4.0, 4.0),
        _clip(float(details.get("survival_value", 0.0) or 0.0), -4.0, 4.0),
        _clip(float(details.get("hard_benchmark_value", 0.0) or 0.0), -4.0, 4.0),
        _clip(local_rank / 12.0, 0.0, 2.0),
        _clip(partner_rank / 12.0, 0.0, 2.0),
        float(
            str(details.get("local_target_slot", "")) == str(details.get("partner_target_slot", ""))
        ),
        float(target_slot_int == 0),
        float(target_slot_int == 1),
        float(reason == "joint-ko"),
        float(reason == "joint-focus"),
        float(reason == "joint-split"),
        float(reason == "joint-pressure"),
        float(benchmark == "vs_simpleheuristics"),
        float(benchmark == "vs_abyssal"),
        float(benchmark == "vs_maxpower"),
        float(benchmark == "vs_random"),
        _clip(combined_total, 0.0, 4.0),
        _clip(slot0_damage, 0.0, 2.5),
        _clip(slot1_damage, 0.0, 2.5),
        _clip(combined_max, 0.0, 2.5),
        _clip(combined_min, 0.0, 2.5),
        _clip(damage_balance, 0.0, 1.0),
        float(0 in ko_slot_ids),
        float(1 in ko_slot_ids),
        is_split_pair,
        weak_split,
        _clip(same_target_overkill, 0.0, 2.0),
        _clip(protect_count, 0.0, 2.0),
        _clip(support_count, 0.0, 2.0),
        float(reason in {"joint-split", "joint-pressure"}),

        _s("self0_hp", 0.0, 1.0),
        _s("self1_hp", 0.0, 1.0),
        _s("opp0_hp", 0.0, 1.0),
        _s("opp1_hp", 0.0, 1.0),
        _s("self0_speed_adv", -3.0, 3.0),
        _s("self1_speed_adv", -3.0, 3.0),
        _s("weather_rain", 0.0, 1.0),
        _s("weather_sun", 0.0, 1.0),
        _s("weather_sand", 0.0, 1.0),
        _s("weather_snow", 0.0, 1.0),
        _s("terrain_electric", 0.0, 1.0),
        _s("terrain_grassy", 0.0, 1.0),
        _s("terrain_psychic", 0.0, 1.0),
        _s("terrain_misty", 0.0, 1.0),
        _s("trick_room", 0.0, 1.0),
        _s("tailwind_self", 0.0, 1.0),
        _s("tailwind_opp", 0.0, 1.0),
        _s("screens_self", 0.0, 1.0),
        _s("screens_opp", 0.0, 1.0),
        _s("hazards_self", 0.0, 4.0),
        _s("hazards_opp", 0.0, 4.0),
        _s("alive_count_self", 0.0, 2.0),
        _s("alive_count_opp", 0.0, 2.0),
        _s("min_self_hp", 0.0, 1.0),
        _s("team_hp_diff", -2.0, 2.0),
        _s("self0_burned", 0.0, 1.0),
        _s("self0_paralyzed", 0.0, 1.0),
        _s("self1_burned", 0.0, 1.0),
        _s("self1_paralyzed", 0.0, 1.0),
        _s("opp0_burned", 0.0, 1.0),
        _s("opp0_poisoned", 0.0, 1.0),
        _s("opp1_burned", 0.0, 1.0),
        _s("opp1_poisoned", 0.0, 1.0),
        _s("turn_norm", 0.0, 1.0),
    ]


def ctde_policy_features_from_details(
    details: Dict[str, Any],
) -> List[float]:

    full = ctde_features_from_details(details, benchmark_type="")

    return [v for v, name in zip(full, CTDE_FEATURE_NAMES) if name not in _CTDE_PRIVILEGED_FEATURES]


def ctde_runtime_features_from_details(details: Dict[str, Any]) -> List[float]:
    runtime_details = dict(details)
    runtime_details.pop("benchmark_type", None)
    return ctde_features_from_details(runtime_details, benchmark_type="")


__all__ = [
    "CTDE_FEATURE_NAMES",
    "CTDE_POLICY_FEATURE_NAMES",
    "_CTDE_PRIVILEGED_FEATURES",
    "_ctde_transform_features",
    "_ctde_transformed_feature_names",
    "ctde_features_from_details",
    "ctde_policy_features_from_details",
    "ctde_runtime_features_from_details",
]
