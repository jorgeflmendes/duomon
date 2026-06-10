from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 384
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.10


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BattleGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, steps = idx.shape
        if steps > self.cfg.block_size:
            raise ValueError("sequence too long")
        pos = torch.arange(0, steps, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def _load_tokenizer(path: Path) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(str(path / "vocab.json"), str(path / "merges.txt"))


def _encode_file(tokenizer: ByteLevelBPETokenizer, path: Path, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    text = path.read_text(encoding="utf-8", errors="replace")
    ids = np.array(tokenizer.encode(text).ids, dtype=np.uint16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, ids)
    return np.load(cache_path, mmap_mode="r")


def _batch(
    data: np.ndarray, batch_size: int, block_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(np.asarray(data[i : i + block_size], dtype=np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(np.asarray(data[i + 1 : i + 1 + block_size], dtype=np.int64)) for i in ix]
    )
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def _estimate_loss(
    model: BattleGPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(args.eval_iters):
            x, y = _batch(data, args.batch_size, args.block_size, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
        mean_loss = float(sum(losses) / max(1, len(losses)))
        out[f"{split}_loss"] = mean_loss
        out[f"{split}_perplexity"] = float(math.exp(min(20.0, mean_loss)))
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/transformer_training/battle_transformer_v1")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional checkpoint to initialize from without overwriting the source run.",
    )
    parser.add_argument("--block-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def _load_initial_checkpoint(
    model: BattleGPT,
    checkpoint_path: str,
    expected_cfg: ModelConfig,
) -> Dict[str, object]:
    if not checkpoint_path:
        return {"loaded": False}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_cfg = checkpoint.get("model_config", {})
    source_cfg = ModelConfig(**raw_cfg)
    comparable = asdict(source_cfg)
    target = asdict(expected_cfg)
    if comparable != target:
        raise ValueError(
            "init checkpoint config does not match target config: "
            f"source={comparable} target={target}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "loaded": True,
        "checkpoint_path": str(path.as_posix()),
        "source_metrics": checkpoint.get("metrics", {}),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    tokenizer = _load_tokenizer(run_dir / "tokenizer")
    train_data = _encode_file(
        tokenizer, run_dir / "train.txt", run_dir / "cache" / "train_tokens.npy"
    )
    val_data = _encode_file(tokenizer, run_dir / "val.txt", run_dir / "cache" / "val_tokens.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = BattleGPT(cfg).to(device)
    init_info = _load_initial_checkpoint(model, args.init_checkpoint, cfg)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    metrics = {
        "schema": "duomon.battle_transformer_training.v1",
        "device": str(device),
        "model_config": asdict(cfg),
        "args": vars(args),
        "init_checkpoint": init_info,
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "history": [],
    }
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = _batch(train_data, args.batch_size, args.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            _, loss = model(x, y)
        assert loss is not None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            evals = _estimate_loss(model, train_data, val_data, args, device)
            row = {
                "step": step,
                "loss": float(loss.item()),
                "elapsed_seconds": time.time() - t0,
                **evals,
            }
            metrics["history"].append(row)
            print(json.dumps(row, sort_keys=True))

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "battle_gpt.pt"
    torch.save(
        {
            "model_config": asdict(cfg),
            "model_state_dict": model.state_dict()
            if not hasattr(model, "_orig_mod")
            else model._orig_mod.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics["checkpoint_path"] = str(checkpoint_path.as_posix())
    (run_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"checkpoint_path": str(checkpoint_path), "final": metrics["history"][-1]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
