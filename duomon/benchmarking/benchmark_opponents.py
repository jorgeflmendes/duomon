from __future__ import annotations


OPPONENT_DISPLAY_NAMES = {
    "random": "Random",
    "maxpower": "MaxPower",
    "simpleheuristics": "SimpleHeuristics",
    "abyssal": "Abyssal",
    "typeaware": "TypeAware",
}

OPPONENT_DESCRIPTIONS = {
    "random": "Selects a legal move or switch at random; useful as the baseline sanity check.",
    "maxpower": "Prioritizes high base-power attacking moves, with small bonuses for accuracy and priority.",
    "simpleheuristics": "Uses poke-env SimpleHeuristics-style scoring with type-aware targeting and double-battle target multipliers.",
    "abyssal": "Uses a stronger damage-focused heuristic with expected damage, KO pressure, STAB, accuracy, priority, status moves and safer switches.",
    "typeaware": "Scores each visible opponent with SimpleHeuristics and targets the best type-aware option.",
}


def opponent_name(opponent_kind: str) -> str:
    return OPPONENT_DISPLAY_NAMES.get(opponent_kind, str(opponent_kind or "unknown"))


def opponent_description(opponent_kind: str) -> str:
    return OPPONENT_DESCRIPTIONS.get(opponent_kind, "Custom benchmark opponent.")
