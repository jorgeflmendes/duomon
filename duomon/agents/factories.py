from __future__ import annotations

import os
import random
import time
from typing import Any, Dict

from ..config import AgentConfig
from .multi_agent import SingleSlotMultiAgent
from ..opponents.multi_opponents import (
    AbyssalStyleMultiSlotOpponent,
    CounterAbyssalMultiSlotOpponent,
    DamageRaceMultiSlotAgent,
    MaxPowerMultiSlotOpponent,
    RandomMultiSlotOpponent,
    SimpleDamageMultiSlotAgent,
    SimpleHeuristicsMultiSlotOpponent,
    SimplePlusMultiSlotAgent,
    SimpleSmartSwitchMultiSlotAgent,
    TypeAwareMultiSlotOpponent,
)
from ..shared import AccountConfiguration, LocalhostServerConfiguration, Player, to_id_str
from ..core.teams import _fixed_ally_team_for_agent, _team_kwargs_for_opponent


def _named_account(name: str):
    base = to_id_str(name)[:8] or "bot"
    suffix = f"{int(time.time() * 1000) % 1000000:06d}{os.getpid() % 1000:03d}{random.SystemRandom().randint(0, 999):03d}"
    username = (base + suffix)[:18]
    return AccountConfiguration(username, None)


def _multi_player_kwargs(config: AgentConfig, agent_name: str) -> Dict[str, Any]:
    return {
        "account_configuration": _named_account(agent_name),
        "battle_format": config.battle_format,
        "max_concurrent_battles": config.max_concurrent_battles,
        "server_configuration": LocalhostServerConfiguration,
        "open_timeout": 8.0,
        "ping_interval": None,
        "ping_timeout": None,
    }


def _make_player(player_cls: Any, config: AgentConfig, agent_name: str, **extra: Any) -> Player:
    kwargs = _multi_player_kwargs(config, agent_name)
    kwargs.update(extra)
    return player_cls(**kwargs)


def _make_multi_opponent(player_cls: Any, config: AgentConfig, agent_name: str) -> Player:
    return _make_player(
        player_cls,
        config,
        agent_name,
        **_team_kwargs_for_opponent(config, agent_name),
    )


ALLY_POLICY_CLASSES: Dict[str, Any] = {
    "simple": SimpleHeuristicsMultiSlotOpponent,
    "simpleplus": SimplePlusMultiSlotAgent,
    "simplesmart": SimpleSmartSwitchMultiSlotAgent,
    "simpledamage": SimpleDamageMultiSlotAgent,
    "maxpower": MaxPowerMultiSlotOpponent,
    "abyssal": AbyssalStyleMultiSlotOpponent,
    "counterabyssal": CounterAbyssalMultiSlotOpponent,
    "damagerace": DamageRaceMultiSlotAgent,
}


def make_multi_agent(config: AgentConfig, agent_name: str, save_replays=False) -> Player:
    ally_policy = (
        str(getattr(config, "ally_policy", "") or os.environ.get("DUOMON_ALLY_POLICY", ""))
        .strip()
        .lower()
    )
    ally_team = _fixed_ally_team_for_agent(config, agent_name)
    team_kwargs = {"team": ally_team} if ally_team else {}
    if (
        ally_policy in {"hybridabyssal", "hybrid-abyssal", "hybrid_abyssal"}
        and "p3" in agent_name.lower()
    ):
        return _make_player(AbyssalStyleMultiSlotOpponent, config, agent_name, **team_kwargs)
    policy_cls = ALLY_POLICY_CLASSES.get(ally_policy)
    if policy_cls is not None:
        return _make_player(policy_cls, config, agent_name, **team_kwargs)
    kwargs = _multi_player_kwargs(config, agent_name)
    kwargs.update(team_kwargs)
    return SingleSlotMultiAgent(
        config=config,
        agent_name=agent_name,
        **kwargs,
        save_replays=save_replays,
    )


def make_multi_random_opponent(
    config: AgentConfig, agent_name: str = "random-multi"
) -> RandomMultiSlotOpponent:
    return _make_multi_opponent(RandomMultiSlotOpponent, config, agent_name)


def make_multi_maxpower_opponent(
    config: AgentConfig, agent_name: str = "maxpower-multi"
) -> MaxPowerMultiSlotOpponent:
    return _make_multi_opponent(MaxPowerMultiSlotOpponent, config, agent_name)


def make_multi_simpleheuristics_opponent(
    config: AgentConfig, agent_name: str = "simpleheuristics-multi"
) -> SimpleHeuristicsMultiSlotOpponent:
    return _make_multi_opponent(SimpleHeuristicsMultiSlotOpponent, config, agent_name)


def make_multi_typeaware_opponent(
    config: AgentConfig, agent_name: str = "typeaware-multi"
) -> TypeAwareMultiSlotOpponent:
    return _make_multi_opponent(TypeAwareMultiSlotOpponent, config, agent_name)


def make_multi_abyssal_opponent(
    config: AgentConfig, agent_name: str = "abyssal-multi"
) -> AbyssalStyleMultiSlotOpponent:
    return _make_multi_opponent(AbyssalStyleMultiSlotOpponent, config, agent_name)


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
