from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .jsonl_utils import iter_jsonl
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from duomon.core.jsonl import iter_jsonl

SUPPORTISH_MOVES = {
    "protect",
    "detect",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "silktrap",
    "burningbulwark",
    "maxguard",
    "recover",
    "roost",
    "slackoff",
    "morningsun",
    "synthesis",
    "softboiled",
    "milkdrink",
    "shoreup",
    "strengthsap",
    "rest",
    "toxicspikes",
    "stealthrock",
    "spikes",
    "stickyweb",
    "swordsdance",
    "nastyplot",
    "dragondance",
    "calmmind",
    "cosmicpower",
    "substitute",
    "leechseed",
    "tailwind",
    "thunderwave",
    "encore",
}

PROTECT_MOVES = {
    "protect",
    "detect",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "silktrap",
    "burningbulwark",
    "maxguard",
}

SETUP_MOVES = {
    "swordsdance",
    "nastyplot",
    "dragondance",
    "calmmind",
    "bulkup",
    "irondefense",
    "quiverdance",
    "shellsmash",
    "coil",
    "agility",
    "rockpolish",
    "takeheart",
    "acidarmor",
    "cosmicpower",
    "amnesia",
    "autotomize",
    "shiftgear",
    "substitute",
    "bellydrum",
    "victorydance",
    "workup",
    "growth",
    "tailglow",
    "honeclaws",
    "clangoroussoul",
    "howl",
}

RECOVERY_MOVES = {
    "recover",
    "roost",
    "slackoff",
    "morningsun",
    "synthesis",
    "softboiled",
    "milkdrink",
    "shoreup",
    "strengthsap",
    "rest",
}

HAZARD_MOVES = {"toxicspikes", "stealthrock", "spikes", "stickyweb"}

SPEED_CONTROL_MOVES = {
    "tailwind",
    "icywind",
    "electroweb",
    "thunderwave",
    "glare",
    "stringshot",
    "bulldoze",
    "snarl",
    "strugglebug",
    "trickroom",
}


def _safe_rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _clean_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def move_from_action(action: str) -> str:
    match = re.search(r"[LR]:([^| ]+)", action or "")
    token = (match.group(1) if match else action or "").strip().lower()
    if "->" in token:
        token = token.split("->", 1)[0]
    if token.startswith("switch"):
        return "switch"
    return _clean_token(token)


def target_from_action(action: str) -> str:
    match = re.search(r"->([^| ]+)", action or "")
    return _clean_token(match.group(1)) if match else ""


def load_results(output_dir: str, opponent: str) -> Dict[str, bool]:
    path = os.path.join(
        output_dir,
        f"multi_independent_vs_{opponent}_multi_eval_vs_{opponent}_results.jsonl",
    )
    results: Dict[str, bool] = {}
    for row in iter_jsonl(path, strict=True):
        battle_tag = str(row.get("battle_tag") or "")
        if battle_tag:
            results[battle_tag] = bool(row.get("p1_won"))
    return results


def load_replays(
    output_dir: str, opponent: str, battle_tags: Iterable[str]
) -> List[Dict[str, Any]]:
    wanted = set(battle_tags)
    path = os.path.join(output_dir, f"multi_independent_vs_{opponent}_replays.jsonl")
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(path, strict=True):
        if row.get("battle_tag") in wanted:
            rows.append(row)
    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counters: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    values: Dict[str, List[float]] = defaultdict(list)
    by_battle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    early_rows = 0
    early_blind_rows = 0

    bool_keys = [
        "focus_fire",
        "split_targets",
        "double_switch",
        "support_without_attack",
        "message_agreement",
        "message_conflict",
        "useful_focus_fire",
        "saved_partner_attempt",
        "threat_response",
        "plan_consistency",
        "offensive_opportunity",
        "high_threat_state",
    ]
    numeric_keys = [
        "communication_gain",
        "overkill_risk",
        "damage_sum",
        "ko_prob_sum",
        "partner_damage_risk",
    ]

    for row in rows:
        by_battle[str(row.get("battle_tag") or "")].append(row)
        action = str(row.get("action") or "")
        move = move_from_action(action)
        target = target_from_action(action)
        actions[move] += 1
        if target:
            targets[target] += 1

        coordination = row.get("coordination") or {}
        if isinstance(coordination, dict):
            for key in bool_keys:
                if coordination.get(key):
                    counters[key] += 1
            for key in numeric_keys:
                try:
                    values[key].append(float(coordination.get(key, 0.0) or 0.0))
                except (TypeError, ValueError):
                    pass

        turn = int(row.get("turn") or 0)
        if turn <= 2:
            early_rows += 1
            if not row.get("opp_active"):
                early_blind_rows += 1

        if move in PROTECT_MOVES:
            counters["protect_move"] += 1
        if move in SPEED_CONTROL_MOVES:
            counters["speed_control_move"] += 1
        if move in SETUP_MOVES:
            counters["setup_move"] += 1
        if move in RECOVERY_MOVES:
            counters["recovery_move"] += 1
        if move in HAZARD_MOVES:
            counters["hazard_move"] += 1
        if move in SUPPORTISH_MOVES:
            counters["supportish_move"] += 1
        if move == "switch":
            counters["switch_action"] += 1
        if target in {"opp1", "opp2"}:
            counters["blind_target_action"] += 1

    last_turns = [
        max(int(item.get("turn") or 0) for item in battle_rows)
        for battle_rows in by_battle.values()
        if battle_rows
    ]

    return {
        "battles": len(by_battle),
        "rows": len(rows),
        "avg_last_turn": _mean([float(value) for value in last_turns]),
        "rates": {
            key: {
                "count": int(counters[key]),
                "total": len(rows),
                "rate": _safe_rate(counters[key], len(rows)),
            }
            for key in [
                "focus_fire",
                "split_targets",
                "support_without_attack",
                "message_conflict",
                "protect_move",
                "speed_control_move",
                "setup_move",
                "recovery_move",
                "hazard_move",
                "switch_action",
                "blind_target_action",
            ]
        },
        "early_blind": {
            "count": early_blind_rows,
            "total": early_rows,
            "rate": _safe_rate(early_blind_rows, early_rows),
        },
        "averages": {key: _mean(items) for key, items in values.items()},
        "top_actions": actions.most_common(12),
        "top_targets": targets.most_common(12),
    }


def _selected_from_messages(row: Dict[str, Any]) -> Dict[str, Any]:
    for message in reversed(row.get("messages") or []):
        if not isinstance(message, dict):
            continue
        selected = (message.get("content") or {}).get("selected") or {}
        if selected:
            return {
                "agent": str(message.get("agent") or ""),
                "target_slot": selected.get("target_slot"),
                "spread": bool(selected.get("spread")),
                "protect": bool(selected.get("protect")),
                "label": str(selected.get("label") or ""),
            }
    return {}


def summarize_paired_turns(rows: List[Dict[str, Any]], results: Dict[str, bool]) -> Dict[str, Any]:
    by_turn: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_turn[(str(row.get("battle_tag") or ""), int(row.get("turn") or 0))].append(row)

    counters: Dict[str, Counter[str]] = {
        "all": Counter(),
        "wins": Counter(),
        "losses": Counter(),
    }
    examples: List[Dict[str, Any]] = []
    for (battle_tag, turn), turn_rows in by_turn.items():
        selected = [_selected_from_messages(row) for row in turn_rows]
        selected = [item for item in selected if item]
        agents: Dict[str, Dict[str, Any]] = {}
        for item in selected:
            agent = str(item.get("agent") or f"agent-{len(agents)}")
            agents[agent] = item
        if len(agents) < 2:
            continue

        values = list(agents.values())[:2]
        outcome = "wins" if results.get(battle_tag, False) else "losses"
        if any(item.get("protect") for item in values):
            kind = "protect_present"
        elif any(item.get("spread") for item in values):
            kind = "spread_present"
        else:
            left = values[0].get("target_slot")
            right = values[1].get("target_slot")
            if left is not None and right is not None and str(left) == str(right):
                kind = "actual_focus"
            elif left is not None and right is not None:
                kind = "actual_split"
            else:
                kind = "unknown"
        counters["all"][kind] += 1
        counters[outcome][kind] += 1
        if kind in {"actual_split", "protect_present"} and len(examples) < 5:
            examples.append(
                {
                    "battle_tag": battle_tag,
                    "turn": turn,
                    "kind": kind,
                    "actions": [
                        {
                            "agent": agent,
                            "label": item.get("label"),
                            "target_slot": item.get("target_slot"),
                            "spread": item.get("spread"),
                            "protect": item.get("protect"),
                        }
                        for agent, item in agents.items()
                    ],
                }
            )

    def block(counter: Counter[str]) -> Dict[str, Any]:
        total = sum(counter.values())
        return {
            "total": total,
            "counts": dict(counter),
            "rates": {key: _safe_rate(value, total) for key, value in counter.items()},
        }

    return {
        "all": block(counters["all"]),
        "wins": block(counters["wins"]),
        "losses": block(counters["losses"]),
        "examples": examples,
    }


def analyze_opponent(output_dir: str, opponent: str) -> Dict[str, Any]:
    results = load_results(output_dir, opponent)
    rows = load_replays(output_dir, opponent, results.keys())
    rows_by_outcome = {
        "wins": [row for row in rows if results.get(str(row.get("battle_tag") or ""), False)],
        "losses": [row for row in rows if not results.get(str(row.get("battle_tag") or ""), False)],
    }
    wins = sum(1 for won in results.values() if won)
    return {
        "opponent": opponent,
        "result_file_battles": len(results),
        "wins": wins,
        "losses": len(results) - wins,
        "winrate": _safe_rate(wins, len(results)),
        "rows": len(rows),
        "wins_rows": summarize_rows(rows_by_outcome["wins"]),
        "losses_rows": summarize_rows(rows_by_outcome["losses"]),
        "paired_turns": summarize_paired_turns(rows, results),
    }


def _format_rate(item: Dict[str, Any]) -> str:
    return f"{item['count']}/{item['total']}={100.0 * item['rate']:.1f}%"


def print_summary(summary: Dict[str, Any]) -> None:
    print()
    print(f"== {summary['opponent']} ==")
    print(
        f"battles={summary['result_file_battles']} "
        f"wins={summary['wins']} losses={summary['losses']} "
        f"winrate={100.0 * summary['winrate']:.1f}% rows={summary['rows']}"
    )
    for label in ("wins_rows", "losses_rows"):
        block = summary[label]
        print(
            f"  {label.replace('_rows', '')}: battles={block['battles']} "
            f"rows={block['rows']} avg_last_turn={block['avg_last_turn']:.2f}"
        )
        keys = [
            "focus_fire",
            "split_targets",
            "support_without_attack",
            "message_conflict",
            "protect_move",
            "speed_control_move",
            "setup_move",
            "recovery_move",
            "hazard_move",
            "switch_action",
            "blind_target_action",
        ]
        print("    rates:", ", ".join(f"{key}:{_format_rate(block['rates'][key])}" for key in keys))
        print("    early_blind:", _format_rate(block["early_blind"]))
        averages = ", ".join(
            f"{key}:{value:.3f}" for key, value in sorted(block["averages"].items())
        )
        print("    averages:", averages or "none")
        print(
            "    top_actions:", ", ".join(f"{name}:{count}" for name, count in block["top_actions"])
        )
        print(
            "    top_targets:", ", ".join(f"{name}:{count}" for name, count in block["top_targets"])
        )
    paired = summary["paired_turns"]
    print("  paired_turns:")
    for label in ("all", "wins", "losses"):
        block = paired[label]
        counts = ", ".join(f"{key}:{value}" for key, value in sorted(block["counts"].items()))
        rates = ", ".join(
            f"{key}:{100.0 * value:.1f}%" for key, value in sorted(block["rates"].items())
        )
        print(f"    {label}: total={block['total']} counts=[{counts}] rates=[{rates}]")
    if paired["examples"]:
        print("    examples:", json.dumps(paired["examples"][:2], ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DuoMon benchmark replay failure modes.")
    parser.add_argument("--output-dir", default=os.environ.get("DUOMON_OUTPUT_DIR", "outputs"))
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["simpleheuristics", "abyssal"],
        help="Opponent result/replay suffixes to analyze.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [analyze_opponent(args.output_dir, opponent) for opponent in args.opponents]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return
    for summary in summaries:
        print_summary(summary)


if __name__ == "__main__":
    main()
