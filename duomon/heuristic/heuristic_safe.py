from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import AgentConfig
from ..shared import (
    HISTORICAL_POKEMON_MOVE_DICT,
    DoubleBattle,
    GenData,
    Move,
    SingleBattleOrder,
)
from .heuristic_constants import *


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def safe_hp_fraction(mon: Any) -> float:
    if mon is None:
        return 0.0
    try:
        value = mon.current_hp_fraction
        return max(0.0, min(1.0, float(value))) if value is not None else 1.0
    except Exception:
        return 1.0


def safe_species(mon: Any) -> str:
    if mon is None:
        return "None"
    return str(safe_getattr(mon, "species", None) or safe_getattr(mon, "name", None) or "Unknown")


def species_key(raw: Any) -> str:
    try:
        text = safe_species(raw) if not isinstance(raw, str) else raw
    except Exception:
        text = str(raw)
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def safe_stats(mon: Any) -> Dict[str, int]:
    stats = safe_getattr(mon, "stats", {}) or {}
    return stats if isinstance(stats, dict) else {}


def safe_stat(mon: Any, stat: str, default: int = 100) -> int:
    try:
        value = safe_stats(mon).get(stat, None)
        if value not in {None, 0}:
            return int(value)
    except Exception:
        pass
    estimated = estimated_randombattle_stat(mon, stat)
    return int(estimated) if estimated is not None else default


_POKEDEX_BASE_STATS: Optional[Dict[str, Dict[str, Any]]] = None


def _pokedex_base_stats() -> Dict[str, Dict[str, Any]]:
    global _POKEDEX_BASE_STATS
    if _POKEDEX_BASE_STATS is not None:
        return _POKEDEX_BASE_STATS
    try:
        from poke_env.data import GenData as _GenData

        _POKEDEX_BASE_STATS = _GenData.from_format("gen9randombattle").pokedex
    except Exception:
        _POKEDEX_BASE_STATS = {}
    return _POKEDEX_BASE_STATS


def base_stat(mon: Any, stat: str, default: int = 100) -> int:
    try:
        species = species_key(mon)
        data = _pokedex_base_stats().get(species, {})
        stats = data.get("baseStats", {}) if isinstance(data, dict) else {}
        return int(stats.get(stat, default) or default)
    except Exception:
        return default


def known_base_stat(mon: Any, stat: str) -> Optional[float]:
    raw_base_stats = (
        safe_getattr(mon, "base_stats", None) or safe_getattr(mon, "baseStats", None) or {}
    )
    try:
        if isinstance(raw_base_stats, dict) and raw_base_stats.get(stat) is not None:
            return float(raw_base_stats.get(stat))
    except Exception:
        pass
    try:
        data = _pokedex_base_stats().get(species_key(mon), {})
        stats = data.get("baseStats", {}) if isinstance(data, dict) else {}
        if stats.get(stat) is not None:
            return float(stats.get(stat))
    except Exception:
        pass
    return None


def _estimated_randombattle_stat_from_base(base: float, level: int) -> float:
    return float(int(((2.0 * float(base) + 52.0) * float(level)) / 100.0) + 5)


def estimated_randombattle_stat(mon: Any, stat: str) -> Optional[float]:
    base = known_base_stat(mon, stat)
    if base is None:
        return None
    level = safe_level(mon)
    if stat == "hp":
        return _estimated_randombattle_hp_from_base(base, level)
    return _estimated_randombattle_stat_from_base(base, level)


def safe_compact_name(raw: Any) -> str:
    if raw is None:
        return "unknown"
    try:
        return (
            str(safe_getattr(raw, "name", None) or safe_getattr(raw, "id", None) or raw)
            .replace(" ", "")
            .replace("-", "")
            .lower()
        )
    except Exception:
        return "unknown"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return safe_compact_name(value)


def safe_speed(mon: Any) -> int:
    stats = safe_stats(mon)
    try:
        if "spe" in stats and stats.get("spe") not in {None, 0}:
            return int(stats.get("spe") or 100)
    except Exception:
        pass
    base = base_stat(mon, "spe", 100)
    if base == 100 and not stats:
        return 100
    level = safe_level(mon)



    return int(((2 * base + 52) * level) / 100 + 5)


def known_target_is_immune(action: "SlotAction", battle: Optional[DoubleBattle] = None) -> bool:
    from .heuristic_damage import _move_hits_ally_activation
    from .heuristic_moves import move_base_power
    from .heuristic_types import _target_is_ally, damage_multiplier

    if action.kind != "move" or action.move is None or action.target is None:
        return False
    if move_base_power(action.move) <= 0:
        return False
    if (
        battle is not None
        and _target_is_ally(battle, action.target)
        and _move_hits_ally_activation(action.move, action.target)
    ):
        return False
    return damage_multiplier(action.target, action.move) == 0


def safe_level(mon: Any) -> int:
    try:
        return int(safe_getattr(mon, "level", 80) or 80)
    except Exception:
        return 80


def is_fainted(mon: Any) -> bool:
    if mon is None or bool(safe_getattr(mon, "fainted", False)):
        return True
    raw_hp = safe_getattr(mon, "current_hp_fraction", None)
    if raw_hp is None:
        return False
    try:
        return float(raw_hp) <= 0.0
    except Exception:
        return False


def active_alive_mons(mons: Sequence[Any]) -> List[Any]:
    return [m for m in mons if m is not None and not is_fainted(m)]


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
