from __future__ import annotations

from .multi_opponent_context import *


def _opponent_slot_index(battle: MultiBattle, target: Any) -> Optional[int]:
    for idx, opp in enumerate(battle.opponent_active_pokemon[:2]):
        if opp is target:
            return idx
    for idx, opp in enumerate(battle.opponent_active_pokemon[:2]):
        if safe_species(opp) == safe_species(target):
            return idx
    return None


def _raw_multi_request(battle: MultiBattle) -> Dict[str, Any]:
    raw = safe_getattr(battle, "_last_raw_request", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _multi_request_must_switch(battle: MultiBattle) -> bool:
    raw_force = _raw_multi_request(battle).get("forceSwitch", None)
    if isinstance(raw_force, list):
        return any(bool(x) for x in raw_force)
    if raw_force is not None:
        return bool(raw_force)
    return bool(force_switch_list(battle)[0])


def _multi_switch_orders(player: Player, battle: MultiBattle) -> List[SingleBattleOrder]:
    orders: List[SingleBattleOrder] = []
    try:
        switches = normalize_slot_list(safe_getattr(battle, "available_switches", []), 0)
        for mon in switches:
            if mon is not None and not is_fainted(mon):
                orders.append(player.create_order(mon))
    except Exception:
        pass
    if orders:
        return orders

    raw = _raw_multi_request(battle)
    side = raw.get("side", {}) if isinstance(raw.get("side", {}), dict) else {}
    active_species = {safe_species(m) for m in battle.active_pokemon if m is not None}
    for pokemon in side.get("pokemon", []) or []:
        if pokemon.get("active"):
            continue
        ident = pokemon.get("ident")
        condition = str(pokemon.get("condition", ""))
        if not ident or condition.startswith("0 fnt") or condition == "0":
            continue
        mon = (safe_getattr(battle, "team", {}) or {}).get(ident) or safe_getattr(
            battle, "_team", {}
        ).get(ident)
        if mon is not None and safe_species(mon) not in active_species and not is_fainted(mon):
            try:
                orders.append(player.create_order(mon))
            except Exception:
                pass
    return orders


def _multi_safe_random_order(player: Player, battle: MultiBattle) -> SingleBattleOrder:
    try:
        if _multi_request_must_switch(battle):
            switches = _multi_switch_orders(player, battle)
            return random.choice(switches) if switches else DefaultBattleOrder()
        orders = []
        try:
            orders.extend(battle.valid_orders[0])
        except Exception:
            pass
        orders = [o for o in orders if not isinstance(o, (PassBattleOrder, DefaultBattleOrder))]
        return random.choice(orders) if orders else DefaultBattleOrder()
    except Exception:
        return DefaultBattleOrder()


def _legal_showdown_positions(battle: MultiBattle, move: Any, active: Any) -> List[int]:
    try:
        return [int(p) for p in battle.get_possible_showdown_targets(move, active)] or [0]
    except Exception:
        return [0]


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
