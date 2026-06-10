from __future__ import annotations

from typing import Tuple

from ..agents.factories import (
    make_multi_abyssal_opponent,
    make_multi_maxpower_opponent,
    make_multi_random_opponent,
    make_multi_simpleheuristics_opponent,
    make_multi_typeaware_opponent,
)
from ..config import AgentConfig
from ..shared import Player


def make_opponent_pair(config: AgentConfig, opponent_kind: str) -> Tuple[Player, Player]:
    if opponent_kind == "maxpower":
        return (
            make_multi_maxpower_opponent(config, "opp-p2-max"),
            make_multi_maxpower_opponent(config, "opp-p4-max"),
        )
    if opponent_kind == "abyssal":
        return (
            make_multi_abyssal_opponent(config, "opp-p2-aby"),
            make_multi_abyssal_opponent(config, "opp-p4-aby"),
        )
    if opponent_kind == "typeaware":
        return (
            make_multi_typeaware_opponent(config, "opp-p2-type"),
            make_multi_typeaware_opponent(config, "opp-p4-type"),
        )
    if opponent_kind == "simpleheuristics":
        return (
            make_multi_simpleheuristics_opponent(config, "opp-p2"),
            make_multi_simpleheuristics_opponent(config, "opp-p4"),
        )
    return (
        make_multi_random_opponent(config, "opp-p2-rand"),
        make_multi_random_opponent(config, "opp-p4-rand"),
    )
