from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from ..config import logger
from ..heuristic import (
    NO_EXPLICIT_TARGETS,
    PROTECT_MOVES,
    _get_multi_short_memory,
    _mon_type_names,
    _multi_opponent_roles,
    _multi_partner_role,
    _multi_same_side,
    _multi_slot_letter,
    is_spread_move,
    json_safe,
    move_id,
    move_target_type,
    safe_compact_name,
    safe_getattr,
    safe_hp_fraction,
    safe_species,
)
from ..shared import (
    DefaultBattleOrder,
    DoubleBattle,
    Move,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.battle.pokemon import Pokemon


VALID_MULTI_TARGET_HEADS = {
    "p1: ",
    "p2: ",
    "p3: ",
    "p4: ",
    "p1a:",
    "p1b:",
    "p2a:",
    "p2b:",
    "p3a:",
    "p3b:",
    "p4a:",
    "p4b:",
}

__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
