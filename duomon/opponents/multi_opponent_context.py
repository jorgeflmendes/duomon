from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..heuristic import (
    FAKE_OUT_MOVES,
    HAZARD_MOVES,
    NO_EXPLICIT_TARGETS,
    PROTECT_MOVES,
    RECOIL_MOVES,
    SETUP_MOVES,
    SPEED_CONTROL_MOVES,
    JointAction,
    OpponentThreatModel,
    SlotAction,
    _advanced_damage_ratio,
    _advanced_ko_probability,
    _ko_prob_from_effective,
    _target_is_ally,
    active_alive_mons,
    damage_multiplier,
    estimate_partner_damage_risk,
    force_switch_list,
    has_type,
    is_fainted,
    is_spread_move,
    move_accuracy,
    move_base_power,
    move_category,
    move_expected_hits,
    move_id,
    move_priority,
    move_target_type,
    normalize_slot_list,
    safe_getattr,
    safe_hp_fraction,
    safe_species,
    safe_speed,
    safe_stat,
    utility_damage_ratio,
)
from ..agents.multi_base import MultiAwarePlayerMixin
from ..battle.multi_battle import MultiBattle
from ..policy_core import IndependentTwoSlotAgent, LegalActionGenerator
from ..shared import (
    DefaultBattleOrder,
    Move,
    PassBattleOrder,
    Player,
    PseudoBattle,
    SimpleHeuristicsPlayer,
    SingleBattleOrder,
)

__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
