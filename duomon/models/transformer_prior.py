from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_PRIORS: Dict[tuple[str, str], "TransformerActionPrior"] = {}


def get_transformer_action_prior(run_dir: str, device: str = "cpu") -> "TransformerActionPrior":
    key = (str(Path(run_dir)), str(device or "cpu"))
    prior = _PRIORS.get(key)
    if prior is None:
        prior = TransformerActionPrior(run_dir, device=device)
        _PRIORS[key] = prior
    return prior


def _safe_name(mon: Any) -> str:
    if mon is None:
        return "none"
    for attr in ("species", "_species", "base_species"):
        value = getattr(mon, attr, "")
        if value:
            return str(value).replace(" ", "").lower()
    return str(mon).split()[0].replace(" ", "").lower()


def _safe_hp(mon: Any) -> float:
    try:
        value = float(getattr(mon, "current_hp_fraction", 1.0))
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    except Exception:
        pass
    return 1.0


def _message_strategies(partner_messages: Sequence[Dict[str, Any]]) -> str:
    strategies: List[str] = []
    for message in partner_messages or []:
        if not isinstance(message, dict):
            continue
        intent = message.get("transformer_intent")
        if isinstance(intent, dict):
            target = intent.get("target_slot")
            gate = intent.get("gate")
            top_actions = intent.get("top_actions") or []
            if target is not None:
                strategies.append(f"transformer_target_{target}")
            if gate is not None:
                try:
                    strategies.append(f"transformer_gate_{float(gate):.2f}")
                except Exception:
                    pass
            for item in top_actions[:3]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("move_id") or item.get("label") or "").strip()
                slot = item.get("target_slot")
                if label:
                    strategies.append(f"transformer_action_{label}_t{slot}")
                if len(strategies) >= 8:
                    break
        for proposal in message.get("proposals") or []:
            if isinstance(proposal, dict):
                strategy = str(proposal.get("strategy") or "").strip()
                if strategy:
                    strategies.append(strategy)
            if len(strategies) >= 8:
                break
        if len(strategies) >= 8:
            break
    return ",".join(strategies) if strategies else "none"


def build_action_prior_prompt(
    battle: Any,
    *,
    agent_name: str,
    benchmark_type: str,
    partner_messages: Sequence[Dict[str, Any]],
) -> str:
    turn = int(getattr(battle, "turn", 0) or 0)
    active = list(getattr(battle, "active_pokemon", []) or [])
    opponents = list(getattr(battle, "opponent_active_pokemon", []) or [])
    self_mon = active[0] if active else None
    partner = active[1] if len(active) > 1 else None
    opp0 = opponents[0] if opponents else None
    opp1 = opponents[1] if len(opponents) > 1 else None
    opponent = str(benchmark_type or "default").replace("vs_", "")
    messages = _message_strategies(partner_messages)
    return (
        "<battle>\n"
        f"<duomon> opponent={opponent} winner=p1+p3\n"
        f"<context> agent={agent_name} turn={turn} "
        f"self={_safe_name(self_mon)}:{_safe_hp(self_mon):.2f} "
        f"partner={_safe_name(partner)}:{_safe_hp(partner):.2f} "
        f"opp0={_safe_name(opp0)}:{_safe_hp(opp0):.2f} "
        f"opp1={_safe_name(opp1)}:{_safe_hp(opp1):.2f} "
        f"messages={messages}\n"
    )


def candidate_action_line(
    battle: Any,
    candidate_summary: Dict[str, Any],
    partner_messages: Sequence[Dict[str, Any]],
) -> str:
    turn = int(getattr(battle, "turn", 0) or 0)
    label = str(candidate_summary.get("label") or candidate_summary.get("move_id") or "unknown")
    score = float(candidate_summary.get("score", 0.0) or 0.0)
    messages = _message_strategies(partner_messages)
    return (
        f"<turn> t={turn} action=L:{label} | R:partner-controlled-by-other-player "
        f"value={max(-1.0, min(1.0, score / 12.0)):.3f} "
        f"protocol=transformer-prior messages={messages}\n"
    )


class TransformerActionPrior:
    def __init__(self, run_dir: str, device: str = "cpu") -> None:
        self.run_dir = Path(run_dir)
        self.device_name = str(device or "cpu")
        self.available = False
        self.error = ""
        self.model = None
        self.tokenizer = None
        self.torch = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from tokenizers import ByteLevelBPETokenizer
            from experiments.battle_transformer.train_battle_lm import BattleGPT, ModelConfig

            checkpoint_path = self.run_dir / "checkpoints" / "battle_gpt.pt"
            tokenizer_dir = self.run_dir / "tokenizer"
            if not checkpoint_path.exists():
                self.error = f"missing checkpoint: {checkpoint_path}"
                return
            self.tokenizer = ByteLevelBPETokenizer(
                str(tokenizer_dir / "vocab.json"),
                str(tokenizer_dir / "merges.txt"),
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = BattleGPT(ModelConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            if self.device_name == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            model.to(device)
            model.eval()
            self.torch = torch
            self.model = model
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def score_lines(self, prefix: str, lines: Sequence[str]) -> List[float]:
        if not self.available or self.model is None or self.tokenizer is None or self.torch is None:
            return [0.0 for _ in lines]
        torch = self.torch
        device = next(self.model.parameters()).device
        pad_id = int(self.tokenizer.token_to_id("<pad>") or 0)
        prefix_ids = list(self.tokenizer.encode(prefix).ids)
        rows: List[List[int]] = []
        line_lengths: List[int] = []
        max_len = 0
        for line in lines:
            line_ids = list(self.tokenizer.encode(line).ids)
            ids = (prefix_ids + line_ids)[-int(self.model.cfg.block_size) :]
            rows.append(ids)
            line_lengths.append(min(len(line_ids), max(0, len(ids) - 1)))
            max_len = max(max_len, len(ids))
        if not rows:
            return []
        batch = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
        for idx, ids in enumerate(rows):
            batch[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = self.model(batch[:, :-1])
            log_probs = torch.log_softmax(logits, dim=-1)
        scores: List[float] = []
        for row_idx, ids in enumerate(rows):
            targets = batch[row_idx, 1 : len(ids)]
            row_log_probs = log_probs[row_idx, : len(ids) - 1]
            target_scores = row_log_probs.gather(1, targets[:, None]).squeeze(1)
            n_line = max(1, int(line_lengths[row_idx]))
            line_scores = target_scores[-n_line:]
            scores.append(float(line_scores.mean().item()))
        return scores


def normalized_prior_scores(raw_scores: Sequence[float], clip: float) -> List[float]:
    values = np.array([float(v) for v in raw_scores], dtype=np.float64)
    if len(values) == 0:
        return []
    std = float(values.std())
    if std <= 1.0e-8:
        return [0.0 for _ in values]
    z = (values - float(values.mean())) / std
    bound = abs(float(clip or 2.0))
    return [float(max(-bound, min(bound, v))) for v in z.tolist()]
