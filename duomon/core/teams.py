from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from poke_env.data import to_id_str
from poke_env.teambuilder.teambuilder import Teambuilder

from ..config import AgentConfig, logger
from ..heuristic import (
    FAKE_OUT_MOVES,
    PIVOT_MOVES,
    PROTECT_MOVES,
    REDIRECTION_MOVES,
    RELIABLE_TEMPO_SPEED_MOVES,
    SLEEP_CONTROL_MOVES,
    base_stat,
)


def _read_team_text(raw_team: str, path: str) -> str:
    def coerce(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("value") or value.get("text") or value.get("team") or ""
        text = str(value or "").strip()
        if text.startswith("{") and "'value'" in text:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    text = str(
                        parsed.get("value") or parsed.get("text") or parsed.get("team") or text
                    ).strip()
            except (SyntaxError, ValueError):
                pass
        return text

    raw_team = coerce(raw_team)
    if raw_team:
        return raw_team
    path = str(path or "").strip()
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as handle:
            return coerce(handle.read())
    except FileNotFoundError:
        logger.warning("[team] status=missing path=%s", path)
    except OSError as exc:
        logger.warning("[team] status=unreadable path=%s error=%s", path, exc)
    return ""


def _pack_team_text(team_text: str) -> str:
    team_text = str(team_text or "").strip()
    if not team_text:
        return ""
    if "|" in team_text and "\n" not in team_text:
        return team_text
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(team_text))


def _team_chunk_species(chunk: str) -> str:
    first = next((line.strip() for line in str(chunk or "").splitlines() if line.strip()), "")
    if "@" in first:
        first = first.split("@", 1)[0]
    if "(" in first:
        first = first.split("(", 1)[0]
    return first.strip()


def _team_chunk_move_ids(chunk: str) -> List[str]:
    moves: List[str] = []
    for line in str(chunk or "").splitlines():
        text = line.strip()
        if text.startswith("-"):
            moves.append(to_id_str(text[1:].strip()))
    return moves


def _fixed_lead_score(chunk: str) -> float:
    species = _team_chunk_species(chunk)
    moves = set(_team_chunk_move_ids(chunk))
    offense = max(base_stat(species, "atk", 90), base_stat(species, "spa", 90))
    speed = base_stat(species, "spe", 80)
    bulk = (
        base_stat(species, "hp", 90) + base_stat(species, "def", 90) + base_stat(species, "spd", 90)
    ) / 3.0
    score = 0.0
    score += 0.0050 * float(offense)
    score += 0.0042 * float(speed)
    score += 0.0015 * float(bulk)
    if offense >= 125 and speed >= 120:
        score += 2.40
    elif offense >= 120 and speed >= 90:
        score += 1.25
    if moves & FAKE_OUT_MOVES:
        score += 3.80
    if moves & SLEEP_CONTROL_MOVES:
        score += 2.00
    if moves & REDIRECTION_MOVES:
        score += 0.85
    if moves & RELIABLE_TEMPO_SPEED_MOVES:
        score += 0.90
    if moves & PIVOT_MOVES:
        score += 0.75
    if moves & PROTECT_MOVES:
        score += 0.25
    priority_moves = {
        "aquajet",
        "grassyglide",
        "suckerpunch",
        "bulletpunch",
        "machpunch",
        "iceshard",
        "extremespeed",
    }
    if moves & priority_moves:
        score += 0.95
    high_power = {
        "surgingstrikes",
        "wickedblow",
        "moonblast",
        "closecombat",
        "woodhammer",
        "flareblitz",
    }
    if moves & high_power:
        score += 0.55
    return float(score)


def _optimize_fixed_team_lead_text(team_text: str) -> str:
    team_text = str(team_text or "").strip()
    if not team_text or "|" in team_text and "\n" not in team_text:
        return team_text
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", team_text) if chunk.strip()]
    if len(chunks) <= 1:
        return team_text
    scored = [(_fixed_lead_score(chunk), idx, chunk) for idx, chunk in enumerate(chunks)]
    best_score, best_idx, _best_chunk = max(scored, key=lambda item: (item[0], -item[1]))
    current_score = scored[0][0]
    if best_idx == 0 or best_score < current_score + 0.85:
        return team_text
    reordered = [chunks[best_idx]] + [chunk for idx, chunk in enumerate(chunks) if idx != best_idx]
    return "\n\n".join(reordered)


def _fixed_team_text_for_role(config: AgentConfig, role: str) -> str:
    role = str(role or "p1").lower()
    if role == "p3":
        raw_team = getattr(config, "fixed_ally_team_p3", "")
        path = getattr(config, "fixed_ally_team_p3_path", "")
    else:
        raw_team = getattr(config, "fixed_ally_team_p1", "")
        path = getattr(config, "fixed_ally_team_p1_path", "")
    return _read_team_text(raw_team, path)


def _team_role_for_agent(agent_name: str) -> str:
    name = str(agent_name or "").lower()
    if "p3" in name or "p4" in name:
        return "p3"
    return "p1"


def _fixed_team_for_agent(config: AgentConfig, agent_name: str, opponent: bool = False) -> str:
    if not bool(getattr(config, "fixed_ally_team_enabled", False)):
        return ""
    if opponent and not bool(getattr(config, "mirror_opponent_team_enabled", False)):
        return ""
    try:
        team_text = _fixed_team_text_for_role(config, _team_role_for_agent(agent_name))
        if bool(getattr(config, "fixed_ally_optimize_leads_enabled", False)):
            team_text = _optimize_fixed_team_lead_text(team_text)
        return _pack_team_text(team_text)
    except Exception as exc:
        logger.warning("[team] status=invalid agent=%s error=%s", agent_name, exc)
        return ""


def _fixed_ally_team_for_agent(config: AgentConfig, agent_name: str) -> str:
    return _fixed_team_for_agent(config, agent_name, opponent=False)


def _fixed_opponent_team_for_agent(config: AgentConfig, agent_name: str) -> str:
    return _fixed_team_for_agent(config, agent_name, opponent=True)


def _team_kwargs_for_opponent(config: AgentConfig, agent_name: str) -> Dict[str, str]:
    team = _fixed_opponent_team_for_agent(config, agent_name)
    return {"team": team} if team else {}


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
