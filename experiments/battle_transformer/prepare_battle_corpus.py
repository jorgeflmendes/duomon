from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from tokenizers import ByteLevelBPETokenizer


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<battle>",
    "</battle>",
    "<vgc>",
    "<duomon>",
    "<winner>",
    "<loser>",
    "<context>",
    "<coord>",
    "<turn>",
    "<action>",
]


VOLATILE_LINE_RE = re.compile(
    r"^\|(j|l|html|uhtml|inactive|inactiveoff|raw|c|c:|chatmsg|pm|queryresponse)\|"
)


def _iter_records(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    for shard in sorted(dataset_dir.glob("battle_shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _stable_split(battle_id: str, validation_fraction: float) -> str:
    digest = hashlib.sha1(battle_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    threshold = int(validation_fraction * ((1 << 64) - 1))
    return "val" if value < threshold else "train"


def _clean_vgc_log(log: str, max_lines: int) -> str:
    cleaned: List[str] = []
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not line or VOLATILE_LINE_RE.match(line):
            continue
        cleaned.append(line)
        if len(cleaned) >= max_lines:
            break
    return "\n".join(cleaned)


def _safe_token(value: Any, default: str = "none") -> str:
    text = str(value if value is not None else default).strip().lower()
    if not text:
        return default
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^a-z0-9_.:+|>,=-]", "", text) or default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return default


def _line_fragment(value: Any, default: str = "none") -> str:
    text = str(value if value is not None else default).strip()
    return re.sub(r"\s+", " ", text) or default


def _active_context(row: Dict[str, Any]) -> str:
    def slot(label: str, mon: Any) -> str:
        if not isinstance(mon, dict):
            return f"{label}=none:1.00:s0"
        species = _safe_token(mon.get("species"))
        hp = max(0.0, min(1.0, _safe_float(mon.get("hp"), 1.0)))
        speed = int(max(0.0, _safe_float(mon.get("spe"), 0.0)))
        return f"{label}={species}:{hp:.2f}:s{speed}"

    my_active = row.get("my_active") if isinstance(row.get("my_active"), list) else []
    opp_active = row.get("opp_active") if isinstance(row.get("opp_active"), list) else []
    return " ".join(
        [
            slot("self", my_active[0] if len(my_active) > 0 else None),
            slot("partner", my_active[1] if len(my_active) > 1 else None),
            slot("opp0", opp_active[0] if len(opp_active) > 0 else None),
            slot("opp1", opp_active[1] if len(opp_active) > 1 else None),
        ]
    )


def _message_summary(messages: Any) -> str:
    if not isinstance(messages, list):
        return "none"
    parts: List[str] = []
    for message in messages[:2]:
        if not isinstance(message, dict):
            continue
        top_strategy = _safe_token(message.get("top_strategy"), "")
        if top_strategy:
            parts.append(f"top:{top_strategy}")
        for proposal in (message.get("proposals") or [])[:3]:
            if not isinstance(proposal, dict):
                continue
            strategy = _safe_token(proposal.get("strategy"), "")
            role = _safe_token(proposal.get("communication_role"), "")
            move = _safe_token(
                proposal.get("recommended_move") or proposal.get("recommended_label"),
                "",
            )
            target = _safe_token(proposal.get("target_species"), "")
            slot = proposal.get("target_slot")
            item = ",".join(
                part
                for part in [
                    strategy,
                    f"role:{role}" if role else "",
                    f"move:{move}" if move else "",
                    f"target:{target}" if target else "",
                    f"slot:{slot}" if slot is not None else "",
                ]
                if part
            )
            if item:
                parts.append(item)
        if len(parts) >= 8:
            break
    return "|".join(parts[:8]) if parts else "none"


def _message_agent(messages: Any) -> str:
    if not isinstance(messages, list):
        return "unknown"
    for message in messages:
        if isinstance(message, dict) and message.get("agent"):
            return _safe_token(message.get("agent"))
    return "unknown"


def _coordination_summary(coord: Any) -> str:
    if not isinstance(coord, dict):
        coord = {}
    bool_fields = [
        "focus_fire",
        "split_targets",
        "message_agreement",
        "message_conflict",
        "support_without_attack",
        "double_switch",
    ]
    numeric_fields = [
        "damage_sum",
        "damage_max",
        "ko_prob_sum",
        "partner_damage_risk",
        "overkill_risk",
        "communication_gain",
    ]
    parts = [
        f"protocol={_safe_token(coord.get('protocol_used'))}",
        f"reason={_safe_token(coord.get('protocol_reason'))}",
    ]
    parts.extend(f"{name}={int(bool(coord.get(name)))}" for name in bool_fields)
    parts.extend(f"{name}={_safe_float(coord.get(name)):.3f}" for name in numeric_fields)
    role_left = _safe_token(coord.get("role_left"), "")
    role_right = _safe_token(coord.get("role_right"), "")
    if role_left or role_right:
        parts.append(f"roles={role_left or 'none'}>{role_right or 'none'}")
    return " ".join(parts)


def _duomon_header(record: Dict[str, Any]) -> str:
    quality_filters = record.get("quality_filters") or {}
    if not isinstance(quality_filters, dict):
        quality_filters = {}
    fixed_hash = _safe_token(quality_filters.get("fixed_ally_team_hash"))
    fixed_allies = fixed_hash != "none"
    return " ".join(
        [
            "<duomon>",
            f"battle_id={_safe_token(record.get('battle_id'))}",
            f"format={_safe_token(record.get('format'))}",
            f"benchmark={_safe_token(record.get('benchmark_type'))}",
            f"opponent={_safe_token(record.get('opponent_kind'))}",
            f"turns={int(_safe_float(record.get('turn_count'), 0.0))}",
            f"fixed_allies={int(fixed_allies)}",
            f"ally_team={fixed_hash}",
            "winner=p1+p3",
        ]
    )


def _duomon_text(record: Dict[str, Any], max_turns: int) -> str:
    lines = [_duomon_header(record)]
    for row in (record.get("trajectory") or [])[:max_turns]:
        coord = row.get("coordination") if isinstance(row.get("coordination"), dict) else {}
        messages = row.get("messages")
        message_summary = _message_summary(messages)
        lines.append(
            " ".join(
                [
                    "<context>",
                    f"agent={_message_agent(messages)}",
                    f"turn={row.get('turn')}",
                    _active_context(row),
                    f"messages={message_summary}",
                ]
            )
        )
        lines.append(" ".join(["<coord>", _coordination_summary(coord)]))
        lines.append(
            " ".join(
                [
                    "<turn>",
                    f"t={row.get('turn')}",
                    f"action={_line_fragment(row.get('action'))}",
                    f"value={_safe_float(row.get('value')):.3f}",
                    f"protocol={_safe_token(coord.get('protocol_used'))}",
                    f"reason={_safe_token(coord.get('protocol_reason'))}",
                    f"messages={message_summary}",
                ]
            )
        )
    return "\n".join(lines)


def _record_to_text(record: Dict[str, Any], args: argparse.Namespace) -> Optional[str]:
    source = str(record.get("source") or "")
    if source == "vgc_bench":
        rating = int(record.get("winner_rating") or 0)
        turn_count = int(record.get("turn_count") or 0)
        if args.profile == "high_precision" and (
            rating < args.high_precision_min_rating or turn_count < args.high_precision_min_turns
        ):
            return None
        log = _clean_vgc_log(str(record.get("text_log") or ""), args.max_log_lines)
        if not log:
            return None
        return "\n".join(
            [
                "<battle>",
                f"<vgc> format={record.get('format')} tier={record.get('tier')}",
                f"<winner> side={record.get('winner_side')} rating={record.get('winner_rating')} name={record.get('winner_name')}",
                f"<loser> side={record.get('loser_side')} rating={record.get('loser_rating')} name={record.get('loser_name')}",
                log,
                "</battle>",
            ]
        )
    if source == "duomon_selfplay_benchmark":
        return "\n".join(["<battle>", _duomon_text(record, args.max_duomon_turns), "</battle>"])
    return None


def _write_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    stats = Counter()
    format_counts = Counter()

    with (
        train_path.open("w", encoding="utf-8") as train,
        val_path.open("w", encoding="utf-8") as val,
    ):
        for record in _iter_records(Path(args.dataset_dir)):
            if args.max_records and stats["accepted"] >= args.max_records:
                break
            text = _record_to_text(record, args)
            stats["seen"] += 1
            if text is None:
                stats["filtered"] += 1
                continue
            if args.duomon_oversample > 1 and record.get("source") == "duomon_selfplay_benchmark":
                copies = args.duomon_oversample
            else:
                copies = 1
            split = _stable_split(str(record.get("battle_id")), args.validation_fraction)
            if rng.random() < args.validation_fraction * 0.02:
                split = "val"
            handle = val if split == "val" else train
            for _ in range(copies):
                handle.write(text)
                handle.write("\n\n")
                stats[f"{split}_records"] += 1
            stats["accepted"] += 1
            stats[f"source:{record.get('source')}"] += 1
            format_counts[str(record.get("format") or record.get("opponent_kind"))] += 1

    manifest = {
        "schema": "duomon.battle_transformer_corpus.v1",
        "profile": args.profile,
        "dataset_dir": args.dataset_dir,
        "train_path": str(train_path.as_posix()),
        "val_path": str(val_path.as_posix()),
        "stats": dict(stats),
        "format_counts": dict(format_counts.most_common(50)),
        "args": vars(args),
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _train_tokenizer(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(output_dir / "train.txt")],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(tokenizer_dir))
    payload = {
        "tokenizer_dir": str(tokenizer_dir.as_posix()),
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": SPECIAL_TOKENS,
    }
    (output_dir / "tokenizer_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="outputs/datasets/curated_battle_transformer_v1")
    parser.add_argument(
        "--output-dir", default="outputs/transformer_training/battle_transformer_v1"
    )
    parser.add_argument("--profile", choices=["broad", "high_precision"], default="high_precision")
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-log-lines", type=int, default=260)
    parser.add_argument("--max-duomon-turns", type=int, default=80)
    parser.add_argument("--duomon-oversample", type=int, default=4)
    parser.add_argument("--high-precision-min-rating", type=int, default=1500)
    parser.add_argument("--high-precision-min-turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = _write_corpus(args)
    tokenizer = _train_tokenizer(args)
    print(json.dumps({"corpus": corpus, "tokenizer": tokenizer}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
