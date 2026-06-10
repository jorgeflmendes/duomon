from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

Number = (int, float)


def comm_mode(config: Any) -> str:
    if not bool(getattr(config, "communication_enabled", True)):
        return "no_comm"
    mode = str(getattr(config, "communication_ablation_mode", "normal") or "normal")
    mode = mode.strip().lower().replace("-", "_")
    aliases = {
        "none": "normal",
        "off": "no_comm",
        "disabled": "no_comm",
        "zero": "zero_messages",
        "zeroed": "zero_messages",
        "noise": "noisy_messages",
        "noisy": "noisy_messages",
        "delay": "delayed_messages",
        "delayed": "delayed_messages",
        "shuffle": "shuffled_messages",
        "shuffled": "shuffled_messages",
        "drop": "message_dropout",
        "dropout": "message_dropout",
    }
    return aliases.get(mode, mode)


def mode_disables_partner_messages(mode: str) -> bool:
    return mode in {"no_comm", "local_only"}


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool) and math.isfinite(float(value))


def _zero_numeric(value: Any) -> Any:
    if _is_numeric(value):
        return 0.0
    if isinstance(value, list):
        return [_zero_numeric(item) for item in value]
    if isinstance(value, tuple):
        return [_zero_numeric(item) for item in value]
    if isinstance(value, dict):
        return {key: _zero_numeric(item) for key, item in value.items()}
    return value


def zero_message(message: Dict[str, Any]) -> Dict[str, Any]:
    msg = copy.deepcopy(message)
    msg["speech_act"] = "ablated"
    msg["top_strategy"] = "zeroed"
    msg["facts"] = {}
    msg["memory"] = {}
    msg["capabilities"] = {}
    msg["proposals"] = []
    msg["top_candidates"] = []
    msg["vector"] = [0.0 for _ in list(msg.get("vector") or [])]
    msg["rationale"] = []
    content = msg.get("content")
    if isinstance(content, dict):
        msg["content"] = _zero_numeric(content)
    return msg


def zero_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    pkt = copy.deepcopy(packet)
    pkt["candidates"] = []
    message = pkt.get("message")
    if isinstance(message, dict):
        pkt["message"] = zero_message(message)
    return pkt


def _add_noise(value: Any, rng: random.Random, std: float) -> Any:
    if _is_numeric(value):
        return float(value) + rng.gauss(0.0, std)
    if isinstance(value, list):
        return [_add_noise(item, rng, std) for item in value]
    if isinstance(value, tuple):
        return [_add_noise(item, rng, std) for item in value]
    if isinstance(value, dict):
        return {key: _add_noise(item, rng, std) for key, item in value.items()}
    return value


def noisy_packet(packet: Dict[str, Any], rng: random.Random, std: float) -> Dict[str, Any]:
    if std <= 0.0:
        return copy.deepcopy(packet)
    pkt = copy.deepcopy(packet)
    candidates = pkt.get("candidates")
    if isinstance(candidates, list):
        pkt["candidates"] = [_add_noise(candidate, rng, std) for candidate in candidates]
    message = pkt.get("message")
    if isinstance(message, dict):
        pkt["message"] = _add_noise(message, rng, std)
        pkt["message"]["schema"] = message.get("schema", "vgc_structured_comm_v1")
        pkt["message"]["speech_act"] = message.get("speech_act", "inform_propose")
        pkt["message"]["agent"] = message.get("agent", pkt.get("agent_name", "partner"))
    return pkt


def packet_gate_value(packet: Dict[str, Any]) -> float:
    message = packet.get("message")
    if not isinstance(message, dict):
        return 0.0
    proposals = message.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        vector = message.get("vector")
        if isinstance(vector, list) and vector:
            vals = [float(v) for v in vector if _is_numeric(v)]
            return max(0.0, min(1.0, max(vals) if vals else 0.0))
        return 0.0
    best = 0.0
    for proposal in proposals:
        if isinstance(proposal, dict):
            try:
                best = max(best, float(proposal.get("confidence", 0.0) or 0.0))
            except Exception:
                continue
    return max(0.0, min(1.0, best))


def gate_packets(packets: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    if threshold <= 0.0:
        return packets
    return [packet for packet in packets if packet_gate_value(packet) >= threshold]


def dropout_packets(
    packets: List[Dict[str, Any]],
    rng: random.Random,
    dropout_prob: float,
) -> List[Dict[str, Any]]:
    prob = max(0.0, min(1.0, float(dropout_prob or 0.0)))
    if prob <= 0.0:
        return packets
    return [packet for packet in packets if rng.random() >= prob]


def apply_packet_intervention(
    packets: List[Dict[str, Any]],
    *,
    mode: str,
    rng: random.Random,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    zero_agent: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not packets:
        return [], {
            "mode": mode,
            "received_packets": 0,
            "used_packets": 0,
            "dropped_packets": 0,
            "zeroed_packets": 0,
            "noisy_packets": 0,
        }
    zero_agent = str(zero_agent or "").strip().lower()
    result: List[Dict[str, Any]] = []
    zeroed = 0
    noisy = 0
    for packet in packets:
        source = str(
            packet.get("agent_name") or (packet.get("message") or {}).get("agent") or ""
        ).lower()
        pkt = packet
        if mode == "zero_messages" or (zero_agent and zero_agent in source):
            pkt = zero_packet(packet)
            zeroed += 1
        elif mode == "noisy_messages" or noise_std > 0.0:
            pkt = noisy_packet(packet, rng, float(noise_std or 0.0))
            noisy += 1
        else:
            pkt = copy.deepcopy(packet)
        result.append(pkt)
    before_dropout = len(result)
    if mode == "message_dropout" or dropout_prob > 0.0:
        result = dropout_packets(result, rng, dropout_prob)
    return result, {
        "mode": mode,
        "received_packets": len(packets),
        "used_packets": len(result),
        "dropped_packets": before_dropout - len(result),
        "zeroed_packets": zeroed,
        "noisy_packets": noisy,
    }


def summarize_messages(messages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    count = 0
    vector_count = 0
    vector_sums: Dict[int, float] = defaultdict(float)
    vector_sq_sums: Dict[int, float] = defaultdict(float)
    proposal_counts: List[int] = []
    gate_values: List[float] = []
    strategies: Dict[str, int] = defaultdict(int)
    speech_acts: Dict[str, int] = defaultdict(int)
    agents: Dict[str, int] = defaultdict(int)
    top_candidate_counts: List[int] = []

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        count += 1
        agents[str(raw.get("agent", "unknown"))] += 1
        speech_acts[str(raw.get("speech_act", "unknown"))] += 1
        proposals = raw.get("proposals")
        if isinstance(proposals, list):
            proposal_counts.append(len(proposals))
            for proposal in proposals:
                if isinstance(proposal, dict):
                    strategies[str(proposal.get("strategy", "unknown"))] += 1
        candidates = raw.get("top_candidates")
        if isinstance(candidates, list):
            top_candidate_counts.append(len(candidates))
        vector = raw.get("vector")
        if isinstance(vector, list):
            vector_count += 1
            for idx, value in enumerate(vector):
                if _is_numeric(value):
                    v = float(value)
                    vector_sums[idx] += v
                    vector_sq_sums[idx] += v * v
        gate_values.append(packet_gate_value({"message": raw}))

    means: Dict[str, float] = {}
    variances: Dict[str, float] = {}
    for idx, total in vector_sums.items():
        mean = total / max(1, vector_count)
        sq = vector_sq_sums[idx] / max(1, vector_count)
        means[str(idx)] = mean
        variances[str(idx)] = max(0.0, sq - mean * mean)

    return {
        "message_count": count,
        "agents": dict(sorted(agents.items())),
        "speech_acts": dict(sorted(speech_acts.items())),
        "strategy_counts": dict(sorted(strategies.items(), key=lambda item: item[1], reverse=True)),
        "avg_proposals": sum(proposal_counts) / len(proposal_counts) if proposal_counts else 0.0,
        "avg_top_candidates": sum(top_candidate_counts) / len(top_candidate_counts)
        if top_candidate_counts
        else 0.0,
        "avg_gate": sum(gate_values) / len(gate_values) if gate_values else 0.0,
        "vector_dim": max((int(k) for k in means.keys()), default=-1) + 1,
        "vector_mean": means,
        "vector_variance": variances,
        "collapsed_vector_dims": [key for key, value in variances.items() if float(value) < 1.0e-6],
    }
