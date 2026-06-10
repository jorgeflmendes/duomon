from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_battle_lm import BattleGPT, ModelConfig, _load_tokenizer


@torch.no_grad()
def generate(
    model: BattleGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(0.05, temperature)
        if top_k > 0:
            values, _indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/transformer_training/battle_transformer_v1")
    parser.add_argument("--prompt", default="<battle>\n<vgc>")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tokenizer = _load_tokenizer(run_dir / "tokenizer")
    checkpoint = torch.load(run_dir / "checkpoints" / "battle_gpt.pt", map_location="cpu")
    cfg = ModelConfig(**checkpoint["model_config"])
    model = BattleGPT(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
