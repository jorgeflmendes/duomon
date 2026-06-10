from __future__ import annotations

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
from .heuristic_safe import *


def move_id(move: Any) -> str:
    return (
        str(
            safe_getattr(move, "id", None)
            or safe_getattr(move, "move_id", None)
            or safe_getattr(move, "name", "")
        )
        .replace(" ", "")
        .replace("-", "")
        .lower()
    )


def move_name(move: Any) -> str:
    return str(safe_getattr(move, "name", None) or safe_getattr(move, "id", None) or "unknown")


def move_base_power(move: Any) -> float:
    try:
        bp = float(safe_getattr(move, "base_power", 0) or 0)
        if bp > 0:
            return bp
        mid = move_id(move)
        if mid in {"grassknot", "lowkick", "heavyslam", "heatcrash"}:
            return 80.0
        if mid in {"superfang", "naturesmadness", "ruination"}:
            return 70.0
        if mid == "beatup":
            return 45.0
        if mid in {"seismictoss", "nightshade"}:
            return float(safe_level(safe_getattr(move, "pokemon", None)) or 80)
        if mid == "finalgambit":
            return 100.0
        return 0.0
    except Exception:
        return 0.0


def move_accuracy(move: Any) -> float:
    acc = safe_getattr(move, "accuracy", 1.0)
    if acc is True or acc is None:
        return 1.0
    try:
        acc = float(acc)
        if acc > 1:
            acc /= 100.0
        return max(0.0, min(1.0, acc))
    except Exception:
        return 1.0


def move_priority(move: Any) -> int:
    try:
        return int(safe_getattr(move, "priority", 0) or 0)
    except Exception:
        return 0


def move_expected_hits(move: Any) -> float:
    try:
        return max(1.0, float(safe_getattr(move, "expected_hits", 1.0) or 1.0))
    except Exception:
        return 1.0


def move_target_type(move: Any) -> str:
    raw = safe_getattr(move, "target", "")
    value = safe_getattr(raw, "value", None)
    name = safe_getattr(raw, "name", None)


    text = str(name or value or raw or "")
    key = (
        text.replace("MoveTarget.", "")
        .replace("movetarget.", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )
    mapping = {
        "self": "self",
        "all": "all",
        "field": "field",
        "foeside": "foeSide",
        "allyside": "allySide",
        "alladjacent": "allAdjacent",
        "alladjacentfoes": "allAdjacentFoes",
        "normal": "normal",
        "any": "any",
        "adjacentfoe": "adjacentFoe",
        "adjacentally": "adjacentAlly",
        "adjacentallyorself": "adjacentAllyOrSelf",
        "allies": "allies",
        "allyteam": "allyTeam",
        "randomnormal": "randomNormal",
        "scripted": "scripted",
    }
    return mapping.get(key, text)


def is_spread_move(move: Any) -> bool:
    target_type = move_target_type(move)
    if target_type in SPREAD_TARGETS:
        return True
    known_targets = NO_EXPLICIT_TARGETS | {
        "normal",
        "any",
        "adjacentFoe",
        "adjacentAlly",
        "adjacentAllyOrSelf",
        "allyTeam",
    }
    if target_type in known_targets:
        return False
    return move_id(move) in SPREAD_MOVE_IDS


def move_category(move: Any) -> str:
    raw = str(safe_getattr(move, "category", "") or "").lower()
    if "physical" in raw:
        return "physical"
    if "special" in raw:
        return "special"
    if "status" in raw:
        return "status"
    return "unknown"


def damage_multiplier(target: Any, move: Any) -> float:
    try:
        move_type = _move_type_name(move)
        ability = _mon_ability_name(target)
        mid = move_id(move)
        if move_type in TYPE_ABSORB_ABILITIES.get(ability, set()):
            return 0.0
        if ability == "levitate" and move_type == "ground" and mid not in {"thousandarrows"}:
            return 0.0
        if ability == "wellbakedbody" and move_type == "fire":
            return 0.0
        if ability == "purifyingsalt" and move_type == "ghost":
            return 0.5
        if mid in {"freezedry", "flyingpress", "thousandarrows"}:
            return _special_type_multiplier(target, move)
    except Exception:
        pass
    try:
        value = target.damage_multiplier(move)
        if value is not None:
            return float(value)
    except Exception:
        pass
    try:
        type_chart = GenData.from_format("gen9randombattle").type_chart
        move_type = _move_type_name(move).upper()
        if not move_type:
            return 1.0
        multiplier = 1.0
        for target_type in _mon_type_names(target):
            row = type_chart.get(str(target_type).upper(), {})
            multiplier *= float(row.get(move_type, 1.0))
        return multiplier
    except Exception:
        return 1.0


def _special_type_multiplier(target: Any, move: Any) -> float:
    type_chart = GenData.from_format("gen9randombattle").type_chart
    target_types = [str(t).upper() for t in _mon_type_names(target)]
    mid = move_id(move)
    if mid == "freezedry":
        multiplier = 1.0
        for target_type in target_types:
            if target_type == "WATER":
                multiplier *= 2.0
            else:
                multiplier *= float(type_chart.get(target_type, {}).get("ICE", 1.0))
        return multiplier
    if mid == "thousandarrows":
        multiplier = 1.0
        for target_type in target_types:
            multiplier *= (
                1.0
                if target_type == "FLYING"
                else float(type_chart.get(target_type, {}).get("GROUND", 1.0))
            )
        return multiplier
    if mid == "flyingpress":
        multiplier = 1.0
        for target_type in target_types:
            row = type_chart.get(target_type, {})
            multiplier *= float(row.get("FIGHTING", 1.0)) * float(row.get("FLYING", 1.0))
        return multiplier
    return 1.0


def has_type(attacker: Any, move: Any) -> bool:
    try:
        move_type = safe_getattr(move, "type", None)
        return move_type in (safe_getattr(attacker, "types", []) or [])
    except Exception:
        return False


def normalize_slot_list(raw: Any, slot: int) -> List[Any]:
    if raw is None or not isinstance(raw, list) or not raw:
        return []
    first = raw[0]
    if isinstance(first, list):
        return list(raw[slot] or []) if 0 <= slot < len(raw) else []
    return list(raw)


def force_switch_list(battle: DoubleBattle) -> List[bool]:
    fs = safe_getattr(battle, "force_switch", False)
    if isinstance(fs, list):
        result = [bool(x) for x in fs[:2]]
        return result + [False] * (2 - len(result))
    return [True, True] if bool(fs) else [False, False]


def trapped_list(battle: DoubleBattle) -> List[bool]:
    trapped = safe_getattr(battle, "trapped", False)
    if isinstance(trapped, list):
        result = [bool(x) for x in trapped[:2]]
        return result + [False] * (2 - len(result))
    return [True, True] if bool(trapped) else [False, False]


def _estimated_randombattle_hp_from_base(base_hp: float, level: int) -> float:
    return float(int(((2.0 * float(base_hp) + 52.0) * float(level)) / 100.0) + int(level) + 10)


def _known_base_hp(mon: Any) -> Optional[float]:
    raw_base_stats = (
        safe_getattr(mon, "base_stats", None) or safe_getattr(mon, "baseStats", None) or {}
    )
    try:
        if isinstance(raw_base_stats, dict) and raw_base_stats.get("hp") is not None:
            return float(raw_base_stats.get("hp"))
    except Exception:
        pass
    try:
        data = _pokedex_base_stats().get(species_key(mon), {})
        stats = data.get("baseStats", {}) if isinstance(data, dict) else {}
        if stats.get("hp") is not None:
            return float(stats.get("hp"))
    except Exception:
        pass
    return None


def estimated_max_hp_points(mon: Any) -> float:
    max_hp = safe_getattr(mon, "max_hp", None)
    try:
        if max_hp:
            return max(1.0, float(max_hp))
    except Exception:
        pass

    current_hp = safe_getattr(mon, "current_hp", None)
    hp_fraction = safe_hp_fraction(mon)
    try:
        if current_hp is not None and hp_fraction > 0.0:
            return max(1.0, float(current_hp) / hp_fraction)
    except Exception:
        pass
    hp_stat = safe_stats(mon).get("hp", None)
    try:
        if hp_stat and float(hp_stat) >= 120.0:
            return max(1.0, float(hp_stat))
    except Exception:
        pass




    base_hp = _known_base_hp(mon) or float(base_stat(mon, "hp", 90))
    level = safe_level(mon)
    try:
        return max(1.0, _estimated_randombattle_hp_from_base(base_hp, level))
    except Exception:
        return 300.0


def estimated_hp_points(mon: Any) -> float:
    raw_hp = safe_getattr(mon, "current_hp_fraction", None)
    fraction = 1.0 if raw_hp is None else safe_hp_fraction(mon)
    return max(1.0, estimated_max_hp_points(mon) * fraction)


def approximate_damage_points(
    move: Any, attacker: Any, target: Any, spread: bool = False, include_accuracy: bool = True
) -> float:
    if move is None or attacker is None or target is None or is_fainted(target):
        return 0.0
    bp = move_base_power(move)
    if bp <= 0:
        return 0.0

    category = move_category(move)
    atk = safe_stat(attacker, "spa" if category == "special" else "atk", 100)
    defense = safe_stat(target, "spd" if category == "special" else "def", 100)
    defense = max(1, defense)
    base = (((2 * safe_level(attacker) / 5 + 2) * bp * atk / defense) / 50) + 2
    mult = damage_multiplier(target, move)
    if mult == 0:
        return 0.0
    stab = 1.5 if has_type(attacker, move) else 1.0
    spread_mod = 0.75 if spread else 1.0
    accuracy_mod = move_accuracy(move) if include_accuracy else 1.0
    return max(0.0, base * stab * mult * accuracy_mod * spread_mod * move_expected_hits(move))


def estimated_ko_probability(move: Any, attacker: Any, target: Any, spread: bool = False) -> float:
    ratio = approximate_damage_points(move, attacker, target, spread=spread) / max(
        1.0, estimated_hp_points(target)
    )
    if ratio >= 1.0:
        return 0.95
    if ratio >= 0.85:
        return 0.65
    if ratio >= 0.70:
        return 0.35
    if ratio >= 0.50:
        return 0.12
    return 0.02


def _lower_names(raw: Any) -> List[str]:
    if raw is None:
        return []
    values = (
        list(raw.keys()) + list(raw.values())
        if isinstance(raw, dict)
        else list(raw)
        if isinstance(raw, (list, tuple, set))
        else [raw]
    )
    names = []
    for item in values:
        if item is None:
            continue
        try:
            names.append(str(getattr(item, "name", item)).lower().replace(" ", "").replace("-", ""))
        except Exception:
            pass
    return names


def _battle_weather_names(battle: DoubleBattle) -> List[str]:
    return _lower_names(safe_getattr(battle, "weather", None))


def _battle_field_names(battle: DoubleBattle) -> List[str]:
    names: List[str] = []
    for attr in ["fields", "field"]:
        names.extend(_lower_names(safe_getattr(battle, attr, None)))
    return names


def _battle_side_condition_names(battle: DoubleBattle, own_side: bool) -> List[str]:
    attr = "side_conditions" if own_side else "opponent_side_conditions"
    return _lower_names(safe_getattr(battle, attr, None))


def _target_is_ally(battle: DoubleBattle, target: Any) -> bool:
    return target is not None and any(
        target is ally for ally in active_alive_mons(battle.active_pokemon)
    )


def _target_is_opponent(battle: DoubleBattle, target: Any) -> bool:
    return target is not None and any(
        target is opp for opp in active_alive_mons(battle.opponent_active_pokemon)
    )


def _target_side_condition_names(battle: DoubleBattle, target: Any) -> List[str]:
    if _target_is_ally(battle, target):
        return _battle_side_condition_names(battle, own_side=True)
    if _target_is_opponent(battle, target):
        return _battle_side_condition_names(battle, own_side=False)
    return []


def _psychic_terrain_blocks_priority(battle: DoubleBattle, target: Any) -> bool:
    return "psychicterrain" in _battle_field_names(battle) and _is_grounded_approx(target)


def _move_type_name(move: Any) -> str:
    mt = safe_getattr(move, "type", None)
    return str(getattr(mt, "name", mt)).lower().replace(" ", "").replace("-", "")


def _mon_type_names(mon: Any) -> List[str]:
    try:
        return [
            str(getattr(t, "name", t)).lower().replace(" ", "").replace("-", "")
            for t in (safe_getattr(mon, "types", []) or [])
            if t
        ]
    except Exception:
        return []


def _mon_status_name(mon: Any) -> str:
    st = safe_getattr(mon, "status", None)
    return str(getattr(st, "name", st)).lower().replace(" ", "").replace("-", "")


def _is_grounded_approx(mon: Any) -> bool:
    if mon is None:
        return False
    if "flying" in _mon_type_names(mon):
        return False
    if _mon_ability_name(mon) == "levitate":
        return False
    if _mon_item_name(mon) == "airballoon":
        return False
    return True


def _mon_item_name(mon: Any) -> str:
    return safe_compact_name(safe_getattr(mon, "item", None))


def _mon_ability_name(mon: Any) -> str:
    return safe_compact_name(safe_getattr(mon, "ability", None))


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
