from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import AgentConfig, ensure_parent_dir, logger
from ..heuristic import (
    ALLY_ACTIVATION_ABILITIES,
    ALLY_ACTIVATION_ITEMS,
    FAKE_OUT_MOVES,
    FIRST_TURN_ONLY_STYLE_MOVES,
    HAZARD_MOVES,
    HELPING_HAND_MOVES,
    LOW_PROGRESS_SUPPORT_MOVES,
    MULTI_INTENT_BLACKBOARD,
    MULTI_SHORT_MEMORY,
    NO_EXPLICIT_TARGETS,
    ONE_TIME_FIELD_MOVES,
    PIVOT_MOVES,
    PROTECT_MOVES,
    RECOIL_MOVES,
    RECOVERY_MOVES,
    REDIRECTION_MOVES,
    RELIABLE_TEMPO_SPEED_MOVES,
    REPEAT_BAD_STATUS_MOVES,
    SCREEN_MOVES,
    SELF_SACRIFICE_MOVES,
    SETUP_MOVES,
    SLEEP_CONTROL_MOVES,
    SPEED_CONTROL_MOVES,
    STATUS_CONTROL_MOVES,
    TERRAIN_BOOST_TYPES,
    TERRAIN_MOVES,
    TERRAIN_REMOVE_MOVES,
    TERRAIN_SET_MOVES,
    JointAction,
    OpponentThreatModel,
    SlotAction,
    TacticalKnowledgeEvaluator,
    ThreatEstimate,
    _advanced_damage_ratio,
    _battle_field_names,
    _battle_side_condition_names,
    _get_multi_short_memory,
    _is_grounded_approx,
    _ko_prob_from_effective,
    _lower_names,
    _mon_ability_name,
    _mon_item_name,
    _mon_type_names,
    _move_hits_ally_activation,
    _move_type_name,
    _multi_coordination_key,
    _multi_side_role,
    _psychic_terrain_blocks_priority,
    _role_num,
    _target_is_opponent,
    action_uses_tera,
    active_alive_mons,
    base_stat,
    blind_ko_probability_from_ratio,
    blind_positional_damage_ratio,
    damage_multiplier,
    force_switch_list,
    has_type,
    is_fainted,
    is_spread_move,
    json_safe,
    known_target_is_immune,
    move_accuracy,
    move_base_power,
    move_category,
    move_expected_hits,
    move_id,
    move_priority,
    move_target_type,
    normalize_slot_list,
    safe_compact_name,
    safe_getattr,
    safe_hp_fraction,
    safe_species,
    safe_speed,
    safe_stat,
    utility_damage_ratio,
    utility_damage_sum,
)
from ..ctde import (
    ctde_runtime_features_from_details,
    make_ctde_joint_reranker,
)
from ..policy_core import IndependentTwoSlotAgent, SlotAgent
from ..shared import (
    DefaultBattleOrder,
    PassBattleOrder,
    SimpleHeuristicsPlayer,
    SingleBattleOrder,
)
from .communication import (
    apply_packet_intervention,
    comm_mode,
    gate_packets,
    mode_disables_partner_messages,
    packet_gate_value,
    zero_message,
)
from ..models.transformer_prior import (
    build_action_prior_prompt,
    candidate_action_line,
    get_transformer_action_prior,
    normalized_prior_scores,
)
from .multi_base import MultiAwarePlayerMixin
from ..battle.multi_battle import MultiBattle
from poke_env.data import GenData

__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
