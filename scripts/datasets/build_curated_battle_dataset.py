from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


PLAYER_RE = re.compile(r"^\|player\|(p[12])\|([^|]*)\|[^|]*\|([^|]*)", re.M)
WIN_RE = re.compile(r"^\|win\|(.+)$", re.M)
TIER_RE = re.compile(r"^\|tier\|(.+)$", re.M)
TURN_RE = re.compile(r"^\|turn\|(\d+)", re.M)


def _default_duomon_results_globs() -> List[str]:
    raw = os.environ.get("DUOMON_RESULTS_GLOBS", "").strip()
    if not raw:
        raw = os.environ.get("DUOMON_DUOMON_RESULTS_GLOBS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    output_root = os.environ.get("DUOMON_OUTPUT_DIR", "outputs").rstrip("/\\")
    return [f"{output_root}/**/runs/*/*results.jsonl"]


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _winner_side(players: Dict[str, Dict[str, Any]], winner_name: str) -> Optional[str]:
    for side, payload in players.items():
        if str(payload.get("name", "")).strip() == winner_name:
            return side
    return None


def _quality_tier(winner_rating: int) -> str:
    if winner_rating >= 1600:
        return "elite_1600_plus"
    if winner_rating >= 1500:
        return "expert_1500_plus"
    if winner_rating >= 1400:
        return "strong_1400_plus"
    return "good_1300_plus"


def _turn_count_from_log(log: str) -> int:
    turns = [_safe_int(match.group(1)) for match in TURN_RE.finditer(log)]
    return max(turns, default=0)


def _parse_vgc_battle(
    battle_id: str,
    value: Any,
    source_file: Path,
    min_winner_rating: int,
    min_loser_rating: int,
    min_turns: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    timestamp = _safe_int(value[0])
    log = str(value[1] or "")
    win_match = WIN_RE.search(log)
    if not win_match:
        return None
    if "forfeited." in log.lower() or "|forfeit|" in log.lower():
        return None

    players: Dict[str, Dict[str, Any]] = {}
    for match in PLAYER_RE.finditer(log):
        side, name, rating = match.groups()
        players[side] = {"name": name.strip(), "rating": _safe_int(rating)}
    if "p1" not in players or "p2" not in players:
        return None

    winner_name = win_match.group(1).strip()
    side = _winner_side(players, winner_name)
    if side is None:
        return None
    loser_side = "p2" if side == "p1" else "p1"
    winner_rating = int(players[side]["rating"])
    loser_rating = int(players[loser_side]["rating"])
    if winner_rating < min_winner_rating or loser_rating < min_loser_rating:
        return None

    turn_count = _turn_count_from_log(log)
    if turn_count < min_turns:
        return None

    tier_match = TIER_RE.search(log)
    return {
        "schema": "duomon.battle_dataset.v1",
        "source": "vgc_bench",
        "source_file": str(source_file.as_posix()),
        "battle_id": battle_id,
        "format": source_file.stem.replace("logs_", ""),
        "tier": tier_match.group(1).strip() if tier_match else "",
        "timestamp": timestamp,
        "winner_side": side,
        "winner_name": winner_name,
        "winner_rating": winner_rating,
        "loser_side": loser_side,
        "loser_name": players[loser_side]["name"],
        "loser_rating": loser_rating,
        "turn_count": turn_count,
        "quality_tier": _quality_tier(winner_rating),
        "quality_filters": {
            "rated": True,
            "non_forfeit": True,
            "min_winner_rating": min_winner_rating,
            "min_loser_rating": min_loser_rating,
            "min_turns": min_turns,
        },
        "players": players,
        "text_log": log,
    }


def iter_vgc_bench_records(
    roots: Iterable[Path],
    min_winner_rating: int,
    min_loser_rating: int,
    min_turns: int,
    max_records: int,
) -> Iterator[Dict[str, Any]]:
    emitted = 0
    files: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".json":
            files.append(root)
        elif root.is_dir():
            files.extend(root.glob("*.json"))
    files = sorted(files, key=lambda path: path.name)
    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        for battle_id, value in data.items():
            record = _parse_vgc_battle(
                str(battle_id),
                value,
                path,
                min_winner_rating,
                min_loser_rating,
                min_turns,
            )
            if record is None:
                continue
            yield record
            emitted += 1
            if max_records and emitted >= max_records:
                return


def _winning_results_by_replay(
    results_globs: Iterable[str],
) -> Dict[Path, Dict[str, Dict[str, Any]]]:
    by_replay: Dict[Path, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for pattern in results_globs:
        for path in Path().glob(pattern):
            for row in _read_jsonl(path):
                if not bool(row.get("finished")) or not bool(row.get("p1_won")):
                    continue
                replay_path = Path(str(row.get("replay_path") or ""))
                battle_tag = str(row.get("battle_tag") or "")
                if replay_path and battle_tag:
                    by_replay[replay_path][battle_tag] = row
    return by_replay


def iter_duomon_records(
    results_globs: Iterable[str],
    min_turns: int,
    max_records: int,
) -> Iterator[Dict[str, Any]]:
    emitted = 0
    by_replay = _winning_results_by_replay(results_globs)
    for replay_path, wins in sorted(by_replay.items(), key=lambda item: str(item[0])):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(replay_path):
            battle_tag = str(row.get("battle_tag") or "")
            if battle_tag in wins:
                grouped[battle_tag].append(row)
        for battle_tag, turns in sorted(grouped.items()):
            turn_count = max((_safe_int(row.get("turn")) for row in turns), default=0)
            if turn_count < min_turns:
                continue
            result = wins[battle_tag]
            yield {
                "schema": "duomon.battle_dataset.v1",
                "source": "duomon_selfplay_benchmark",
                "source_file": str(replay_path.as_posix()),
                "battle_id": battle_tag,
                "format": "gen9duomonfixedalliesmultirandombattle",
                "benchmark_type": result.get("benchmark_type"),
                "opponent_kind": result.get("opponent_kind"),
                "winner_side": "p1+p3",
                "winner_name": "duomon_curated_allies",
                "turn_count": turn_count,
                "quality_tier": "duomon_win",
                "quality_filters": {
                    "finished": True,
                    "p1_won": True,
                    "fixed_ally_team_hash": result.get("fixed_ally_team_hash"),
                    "min_turns": min_turns,
                },
                "result": result,
                "trajectory": turns,
            }
            emitted += 1
            if max_records and emitted >= max_records:
                return


class ShardWriter:
    def __init__(self, output_dir: Path, records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.records_per_shard = max(1, records_per_shard)
        self.shard_index = -1
        self.records_in_shard = 0
        self.handle: Optional[gzip.GzipFile] = None
        self.shards: List[Dict[str, Any]] = []

    def __enter__(self) -> "ShardWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _open_next(self) -> None:
        self.close()
        self.shard_index += 1
        self.records_in_shard = 0
        path = self.output_dir / f"battle_shard_{self.shard_index:05d}.jsonl.gz"
        self.handle = gzip.open(path, "wt", encoding="utf-8")
        self.shards.append({"path": path.name, "records": 0})

    def write(self, record: Dict[str, Any]) -> None:
        if self.handle is None or self.records_in_shard >= self.records_per_shard:
            self._open_next()
        assert self.handle is not None
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.records_in_shard += 1
        self.shards[-1]["records"] += 1


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    source_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    total_records = 0

    with ShardWriter(output_dir, args.records_per_shard) as writer:
        if args.include_vgc_bench:
            roots = [Path(path) for path in args.vgc_bench_roots]
            for record in iter_vgc_bench_records(
                roots,
                args.min_winner_rating,
                args.min_loser_rating,
                args.min_turns,
                args.max_vgc_records,
            ):
                writer.write(record)
                total_records += 1
                source_counts[record["source"]] += 1
                tier_counts[record["quality_tier"]] += 1
                format_counts[record["format"]] += 1

        if args.include_duomon:
            for record in iter_duomon_records(
                args.duomon_results_globs,
                args.min_turns,
                args.max_duomon_records,
            ):
                writer.write(record)
                total_records += 1
                source_counts[record["source"]] += 1
                tier_counts[record["quality_tier"]] += 1
                format_counts[str(record.get("opponent_kind") or record["format"])] += 1

        manifest = {
            "schema": "duomon.battle_dataset_manifest.v1",
            "output_dir": str(output_dir.as_posix()),
            "records": total_records,
            "shards": writer.shards,
            "source_counts": dict(source_counts),
            "quality_tier_counts": dict(tier_counts),
            "format_counts": dict(format_counts),
            "filters": {
                "min_winner_rating": args.min_winner_rating,
                "min_loser_rating": args.min_loser_rating,
                "min_turns": args.min_turns,
                "include_vgc_bench": args.include_vgc_bench,
                "include_duomon": args.include_duomon,
                "max_vgc_records": args.max_vgc_records,
                "max_duomon_records": args.max_duomon_records,
            },
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/datasets/curated_battle_transformer_v1")
    parser.add_argument("--records-per-shard", type=int, default=5000)
    parser.add_argument("--min-winner-rating", type=int, default=1300)
    parser.add_argument("--min-loser-rating", type=int, default=1200)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-vgc-records", type=int, default=0)
    parser.add_argument("--max-duomon-records", type=int, default=0)
    parser.add_argument("--include-vgc-bench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-duomon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vgc-bench-roots",
        nargs="+",
        default=["external/VGC-Bench/battle_logs"],
    )
    parser.add_argument(
        "--duomon-results-globs",
        nargs="+",
        default=_default_duomon_results_globs(),
    )
    return parser.parse_args()


def main() -> None:
    manifest = build_dataset(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
