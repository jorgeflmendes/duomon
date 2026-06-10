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
from .heuristic_safe import *
from .heuristic_moves import *


def _damage_context_modifiers_enabled() -> bool:
    return os.environ.get("DUOMON_DAMAGE_CONTEXT_MODIFIERS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _item_damage_modifier(
    attacker: Any, target: Any, move: Any, category: str, mult: float
) -> float:
    item = _mon_item_name(attacker)
    mtype = _move_type_name(move)
    modifier = 1.0
    if item == "lifeorb":
        modifier *= 1.3
    if item == "choiceband" and category == "physical":
        modifier *= 1.5
    if item == "choicespecs" and category == "special":
        modifier *= 1.5
    if item == "muscleband" and category == "physical":
        modifier *= 1.1
    if item == "wiseglasses" and category == "special":
        modifier *= 1.1
    if item == "expertbelt" and mult > 1.0:
        modifier *= 1.2
    if item in _TYPE_BOOST_ITEMS.get(mtype, set()):
        modifier *= 1.2

    target_item = _mon_item_name(target)
    if target_item == "assaultvest" and category == "special":
        modifier /= 1.5
    if target_item == "eviolite":
        modifier /= 1.5
    return modifier


def _ability_damage_modifier(
    attacker: Any, target: Any, move: Any, category: str, mult: float, dmg: float
) -> float:
    ability = _mon_ability_name(attacker)
    target_ability = _mon_ability_name(target)
    mid = move_id(move)
    bp = move_base_power(move)
    mtype = _move_type_name(move)
    modifier = 1.0

    if ability in {"hugepower", "purepower"} and category == "physical":
        modifier *= 2.0
    if ability == "adaptability" and has_type(attacker, move):
        modifier *= 4.0 / 3.0
    if ability == "technician" and 0 < bp <= 60:
        modifier *= 1.5
    if ability == "tintedlens" and 0.0 < mult < 1.0:
        modifier *= 2.0
    if ability == "guts" and category == "physical" and _mon_status_name(attacker):
        modifier *= 1.5
    if ability == "hustle" and category == "physical":
        modifier *= 1.5
    if ability == "flareboost" and category == "special" and "brn" in _mon_status_name(attacker):
        modifier *= 1.5
    if ability == "toxicboost" and category == "physical" and "psn" in _mon_status_name(attacker):
        modifier *= 1.5
    if ability == "sharpness" and mid in SLICING_MOVES:
        modifier *= 1.5
    if ability == "ironfist" and mid in PUNCHING_MOVES:
        modifier *= 1.2
    if ability == "reckless" and mid in RECOIL_MOVES:
        modifier *= 1.2

    if target_ability in {"multiscale", "shadowshield"} and safe_hp_fraction(target) >= 0.999:
        modifier *= 0.5
    if target_ability == "furcoat" and category == "physical":
        modifier *= 0.5
    if target_ability == "thickfat" and mtype in {"fire", "ice"}:
        modifier *= 0.5
    if target_ability == "heatproof" and mtype == "fire":
        modifier *= 0.5
    if target_ability in {"filter", "solidrock", "prismarmor"} and mult > 1.0:
        modifier *= 0.75

    return modifier


def _move_hits_ally_activation(move: Any, ally: Any) -> bool:
    if move is None or ally is None:
        return False
    mid = move_id(move)
    ability = _mon_ability_name(ally)
    item = _mon_item_name(ally)
    mtype = _move_type_name(move)
    damaging = move_base_power(move) > 0 or mid in ALLY_MULTI_HIT_ACTIVATION_MOVES
    if not damaging:
        return False
    if mtype in TYPE_ABSORB_ABILITIES.get(ability, set()):
        return ability in ALLY_ACTIVATION_ABILITIES
    if (
        item == "weaknesspolicy"
        and mid not in FIXED_DAMAGE_MOVES
        and damage_multiplier(ally, move) > 1.0
    ):
        return True
    if ITEM_ACTIVATION_TYPES.get(item) == mtype and damage_multiplier(ally, move) > 0:
        return True
    if ability == "justified" and mtype == "dark":
        return True
    if ability in {"weakarmor", "stamina"} and move_category(move) == "physical":
        return True
    if ability == "stamina" and move_base_power(move) > 0:
        return True
    if ability == "steamengine" and mtype in {"water", "fire"}:
        return True
    if ability == "watercompaction" and mtype == "water":
        return True
    if ability in {"motordrive", "voltabsorb", "lightningrod"} and mtype == "electric":
        return True
    if ability in {"waterabsorb", "stormdrain"} and mtype == "water":
        return True
    if ability == "flashfire" and mtype == "fire":
        return True
    if ability == "sapsipper" and mtype == "grass":
        return True
    if ability == "eartheater" and mtype == "ground":
        return True
    return mid in ALLY_MULTI_HIT_ACTIVATION_MOVES and (
        item in ALLY_ACTIVATION_ITEMS or ability in ALLY_ACTIVATION_ABILITIES
    )


def _boost_stage(mon: Any, stat: str) -> int:
    boosts = safe_getattr(mon, "boosts", None) or safe_getattr(mon, "stat_boosts", None) or {}
    try:
        return int(boosts.get(stat, 0) or 0) if isinstance(boosts, dict) else 0
    except Exception:
        return 0


def _stage_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage)))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def _advanced_damage_points(
    battle: DoubleBattle,
    move: Any,
    attacker: Any,
    target: Any,
    spread: bool = False,
    helping_hand: bool = False,
    include_accuracy: bool = True,
) -> float:
    dmg = approximate_damage_points(
        move, attacker, target, spread=spread, include_accuracy=include_accuracy
    )
    if dmg <= 0:
        return 0.0
    category = move_category(move)
    mtype = _move_type_name(move)
    weather_blob = " ".join(_battle_weather_names(battle))
    fields_blob = " ".join(_battle_field_names(battle))

    if category == "physical":
        dmg *= _stage_multiplier(_boost_stage(attacker, "atk"))
        dmg /= max(0.35, _stage_multiplier(_boost_stage(target, "def")))
        if _mon_ability_name(attacker) != "guts" and (
            "brn" in _mon_status_name(attacker) or "burn" in _mon_status_name(attacker)
        ):
            dmg *= 0.55
    elif category == "special":
        dmg *= _stage_multiplier(_boost_stage(attacker, "spa"))
        dmg /= max(0.35, _stage_multiplier(_boost_stage(target, "spd")))

    if mtype == "water" and ("raindance" in weather_blob or "rain" in weather_blob):
        dmg *= 1.5
    if mtype == "fire" and ("raindance" in weather_blob or "rain" in weather_blob):
        dmg *= 0.5
    if mtype == "fire" and (
        "sunnyday" in weather_blob or "sun" in weather_blob or "desolateland" in weather_blob
    ):
        dmg *= 1.5
    if mtype == "water" and (
        "sunnyday" in weather_blob or "sun" in weather_blob or "desolateland" in weather_blob
    ):
        dmg *= 0.5
    attacker_grounded = _is_grounded_approx(attacker)
    target_grounded = _is_grounded_approx(target)
    if attacker_grounded and mtype == "electric" and "electricterrain" in fields_blob:
        dmg *= 1.3
    if attacker_grounded and mtype == "grass" and "grassyterrain" in fields_blob:
        dmg *= 1.3
    if attacker_grounded and mtype == "psychic" and "psychicterrain" in fields_blob:
        dmg *= 1.3
    if target_grounded and mtype == "dragon" and "mistyterrain" in fields_blob:
        dmg *= 0.5
    if (
        target_grounded
        and mtype == "ground"
        and move_id(move) in {"earthquake", "bulldoze", "magnitude"}
        and "grassyterrain" in fields_blob
    ):
        dmg *= 0.5
    target_side_blob = " ".join(_target_side_condition_names(battle, target))
    if category == "physical" and "reflect" in target_side_blob:
        dmg *= 2.0 / 3.0
    if category == "special" and "lightscreen" in target_side_blob:
        dmg *= 2.0 / 3.0
    if "auroraveil" in target_side_blob:
        dmg *= 2.0 / 3.0
    if helping_hand:
        dmg *= 1.5
    if _damage_context_modifiers_enabled():
        mult = damage_multiplier(target, move)
        dmg *= _item_damage_modifier(attacker, target, move, category, mult)
        dmg *= _ability_damage_modifier(attacker, target, move, category, mult, dmg)
    return max(0.0, dmg)


def _advanced_damage_ratio(
    battle: DoubleBattle,
    move: Any,
    attacker: Any,
    target: Any,
    spread: bool = False,
    helping_hand: bool = False,
    include_accuracy: bool = True,
) -> float:
    return _advanced_damage_points(
        battle,
        move,
        attacker,
        target,
        spread=spread,
        helping_hand=helping_hand,
        include_accuracy=include_accuracy,
    ) / max(1.0, estimated_max_hp_points(target))


def _ko_prob_from_effective(effective: float) -> float:
    if effective >= 1.18:
        return 0.97
    if effective >= 1.00:
        return 0.06 + 0.91 * (effective - 1.00) / 0.18
    if effective >= 0.80:
        return 0.02 + 0.04 * (effective - 0.80) / 0.20
    return max(0.01, 0.02 * effective / 0.80)


def _advanced_ko_probability(
    battle: DoubleBattle,
    move: Any,
    attacker: Any,
    target: Any,
    spread: bool = False,
    helping_hand: bool = False,
) -> float:
    ratio = _advanced_damage_ratio(
        battle, move, attacker, target, spread=spread, helping_hand=helping_hand
    )
    raw_hp = safe_getattr(target, "current_hp_fraction", None)
    target_hp = 1.0 if raw_hp is None else safe_hp_fraction(target)
    if target_hp <= 0.01:
        return 0.97
    return _ko_prob_from_effective(ratio / target_hp)


def blind_positional_damage_ratio(move: Any, attacker: Any, spread: bool = False) -> float:
    bp = move_base_power(move)
    if bp <= 0:
        return 0.0
    stab = 1.25 if attacker is not None and has_type(attacker, move) else 1.0
    spread_mod = 0.72 if spread else 1.0
    ratio = 0.18 + 0.0042 * min(160.0, bp * move_expected_hits(move))
    ratio *= stab * spread_mod * move_accuracy(move)
    if move_priority(move) > 0:
        ratio += 0.05
    return max(0.06, min(0.92, float(ratio)))


def blind_ko_probability_from_ratio(ratio: float) -> float:
    if ratio >= 0.86:
        return 0.45
    if ratio >= 0.64:
        return 0.16
    if ratio >= 0.42:
        return 0.06
    return 0.02


def utility_damage_ratio(value: Any, cap: float = 1.35) -> float:
    try:
        return max(0.0, min(float(cap), float(value)))
    except Exception:
        return 0.0


def utility_damage_sum(values: Any, cap: float = 1.35) -> float:
    try:
        return float(sum(utility_damage_ratio(value, cap=cap) for value in values))
    except Exception:
        return 0.0


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
