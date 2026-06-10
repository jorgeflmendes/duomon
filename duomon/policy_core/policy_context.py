from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import AgentConfig, AgentMode, ensure_parent_dir, logger
from ..heuristic import (
    BENEFICIAL_ALLY_TARGET_MOVES,
    FAKE_OUT_MOVES,
    FIRST_TURN_ONLY_STYLE_MOVES,
    HELPING_HAND_MOVES,
    LOW_PROGRESS_SUPPORT_MOVES,
    NO_EXPLICIT_TARGETS,
    ONE_TIME_FIELD_MOVES,
    PIVOT_MOVES,
    PROTECT_MOVES,
    RECHARGE_MOVES,
    RECOIL_MOVES,
    REDIRECTION_MOVES,
    REPEAT_BAD_STATUS_MOVES,
    SCREEN_MOVES,
    SELF_DEBUFF_MOVES,
    SELF_SACRIFICE_MOVES,
    SETUP_MOVES,
    SLEEP_CONTROL_MOVES,
    SPEED_CONTROL_MOVES,
    STATUS_CONTROL_MOVES,
    TERRAIN_BOOST_TYPES,
    TERRAIN_REMOVE_MOVES,
    TERRAIN_SET_MOVES,
    CoordinationMetrics,
    JointAction,
    OpponentThreatModel,
    SlotAction,
    TacticalKnowledgeEvaluator,
    ThreatEstimate,
    _advanced_damage_ratio,
    _advanced_ko_probability,
    _battle_field_names,
    _battle_side_condition_names,
    _is_grounded_approx,
    _ko_prob_from_effective,
    _mon_type_names,
    _move_hits_ally_activation,
    _target_is_ally,
    _target_is_opponent,
    action_uses_tera,
    active_alive_mons,
    approximate_damage_points,
    blind_ko_probability_from_ratio,
    blind_positional_damage_ratio,
    compute_coordination_metrics,
    damage_multiplier,
    estimate_partner_damage_risk,
    estimated_hp_points,
    estimated_ko_probability,
    estimated_max_hp_points,
    force_switch_list,
    has_type,
    is_fainted,
    is_spread_move,
    json_safe,
    known_target_is_immune,
    move_accuracy,
    move_base_power,
    move_expected_hits,
    move_id,
    move_name,
    move_priority,
    move_target_type,
    normalize_slot_list,
    safe_getattr,
    safe_hp_fraction,
    safe_species,
    safe_speed,
    safe_stat,
    trapped_list,
    utility_damage_ratio,
    utility_damage_sum,
)
from ..shared import (
    DefaultBattleOrder,
    DoubleBattle,
    DoubleBattleOrder,
    Move,
    PassBattleOrder,
    Player,
    SingleBattleOrder,
)

_MODEL_FILE_LOCKS: Dict[str, threading.Lock] = {}

__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
