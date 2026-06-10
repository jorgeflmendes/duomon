from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np

from ..config import ensure_parent_dir
from .ctde_data import (
    _collect_ctde_pairwise_examples,
    _normalize_benchmark_filter,
    _public_split_stats,
    _write_ctde_split_manifest,
    load_benchmark_outcomes,
)
from .ctde_eval import evaluate_ctde_joint_reranker_records
from .ctde_features import (
    CTDE_FEATURE_NAMES,
    _ctde_transform_features,
    _ctde_transformed_feature_names,
)
from .ctde_models import CTDEJointReranker
from .ctde_train_value import _train_ctde_value_regression


def train_ctde_joint_mlp_reranker(
    dataset_path: str,
    output_path: str,
    results_dir: str = "outputs",
    outcomes_path: str = "",
    epochs: int = 24,
    learning_rate: float = 0.0015,
    margin: float = 0.15,
    objective: str = "outcome_margin",
    loss_weight: float = 0.70,
    max_win_alternatives: int = 8,
    max_loss_alternatives: int = 5,
    hidden_sizes: Sequence[int] = (96, 48),
    batch_size: int = 512,
    weight_decay: float = 1.0e-4,
    transform: str = "compact_nonlinear",
    activation: str = "tanh",
    benchmarks: Any = None,
    seed: int = 42,
    max_pairs: int = 0,
    device: str = "auto",
    validation_fraction: float = 0.20,
    split_path: str = "",
    allow_empty_overwrite: bool = False,
    dropout: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1.0e-4,
    lr_scheduler_patience: int = 0,
    lr_scheduler_factor: float = 0.5,
) -> Dict[str, Any]:

    try:
        import torch
        import torch.nn.functional as functional
    except Exception as exc:
        raise RuntimeError("PyTorch is required to train the CTDE MLP offline.") from exc

    objective = str(objective or "outcome_margin").strip().lower()
    if objective in {"outcome", "outcome-aware", "outcome_aware", "outcome-margin"}:
        objective = "outcome_margin"
    if objective in {
        "value",
        "win_prob",
        "winprob",
        "win-probability",
        "value-regression",
    }:
        objective = "value_regression"
    if objective not in {"outcome_margin", "value_regression"}:
        objective = "outcome_margin"

    allowed_benchmarks = _normalize_benchmark_filter(benchmarks)
    outcomes = load_benchmark_outcomes(results_dir, outcomes_path)

    if objective == "value_regression":
        return _train_ctde_value_regression(
            dataset_path=dataset_path,
            output_path=output_path,
            outcomes=outcomes,
            results_dir=results_dir,
            outcomes_path=outcomes_path,
            allowed_benchmarks=allowed_benchmarks,
            epochs=epochs,
            learning_rate=learning_rate,
            loss_weight=loss_weight,
            max_candidates_per_record=max_win_alternatives,
            hidden_sizes=hidden_sizes,
            batch_size=batch_size,
            weight_decay=weight_decay,
            transform=transform,
            activation=activation,
            seed=seed,
            max_pairs=max_pairs,
            device=device,
            validation_fraction=validation_fraction,
            split_path=split_path,
            allow_empty_overwrite=allow_empty_overwrite,
            dropout=dropout,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )

    pair_data = _collect_ctde_pairwise_examples(
        dataset_path=dataset_path,
        outcomes=outcomes,
        objective=objective,
        loss_weight=loss_weight,
        max_win_alternatives=max_win_alternatives,
        max_loss_alternatives=max_loss_alternatives,
        benchmarks=allowed_benchmarks,
        validation_fraction=validation_fraction,
        split_seed=seed,
    )
    positives = pair_data["positives"]
    negatives = pair_data["negatives"]
    sample_weights = pair_data["sample_weights"]
    pair_count = len(positives)
    if pair_count <= 0:
        _write_ctde_split_manifest(split_path, pair_data["stats"], dataset_path, outcomes_path)
        result = {
            "examples": 0,
            "updates": 0,
            "output_path": output_path,
            "eval": {},
            "pair_stats": _public_split_stats(pair_data["stats"]),
        }
        if not allow_empty_overwrite:
            result.update({"output_written": False, "status": "skipped_no_examples"})
            return result
        payload = {
            "model_type": "mlp",
            "feature_names": CTDE_FEATURE_NAMES,
            "transform": transform,
            "input_feature_names": _ctde_transformed_feature_names(transform),
            "layers": [],
            "biases": [],
            "activation": activation,
            "examples": 0,
            "updates": 0,
            "metadata": {
                "dataset_path": dataset_path,
                "results_dir": results_dir,
                "outcomes_path": outcomes_path,
                "objective": objective,
                "benchmarks": sorted(allowed_benchmarks) if allowed_benchmarks else [],
                "pair_stats": _public_split_stats(pair_data["stats"]),
                "eval": {},
            },
        }
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        result["output_written"] = True
        return result

    rng = np.random.default_rng(int(seed))
    if max_pairs and int(max_pairs) > 0 and pair_count > int(max_pairs):
        keep = rng.choice(pair_count, size=int(max_pairs), replace=False)
        positives = [positives[int(idx)] for idx in keep]
        negatives = [negatives[int(idx)] for idx in keep]
        sample_weights = [sample_weights[int(idx)] for idx in keep]
        pair_count = len(positives)

    pos = np.stack([_ctde_transform_features(row, transform) for row in positives]).astype(
        np.float32
    )
    neg = np.stack([_ctde_transform_features(row, transform) for row in negatives]).astype(
        np.float32
    )
    all_inputs = np.vstack([pos, neg])
    input_mean = all_inputs.mean(axis=0).astype(np.float32)
    input_std = all_inputs.std(axis=0).astype(np.float32)
    input_std[input_std < 1.0e-4] = 1.0
    pos = ((pos - input_mean) / input_std).astype(np.float32)
    neg = ((neg - input_mean) / input_std).astype(np.float32)
    weights = np.array(sample_weights, dtype=np.float32)
    weights = np.maximum(weights, 0.05)

    requested_device = str(device or "auto").strip().lower()
    if requested_device in {"", "auto"}:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(requested_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DUOMON_CTDE_DEVICE pediu CUDA, mas torch.cuda.is_available() e falso.")

    torch.manual_seed(int(seed))
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    try:
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass

    hidden = tuple(int(size) for size in hidden_sizes if int(size) > 0)
    activation = str(activation or "tanh").strip().lower()
    if activation not in {"tanh", "relu"}:
        activation = "tanh"
    modules: List[Any] = []
    previous_size = int(pos.shape[1])
    activation_module = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
    dropout_rate = max(0.0, min(0.80, float(dropout or 0.0)))
    for size in hidden:
        modules.append(torch.nn.Linear(previous_size, int(size)))
        modules.append(activation_module())
        if dropout_rate > 0.0:
            modules.append(torch.nn.Dropout(dropout_rate))
        previous_size = int(size)
    modules.append(torch.nn.Linear(previous_size, 1))
    model = torch.nn.Sequential(*modules).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    pos_tensor = torch.from_numpy(pos).to(torch_device)
    neg_tensor = torch.from_numpy(neg).to(torch_device)
    weight_tensor = torch.from_numpy(weights).to(torch_device)
    batch_size = max(16, int(batch_size))
    updates = 0
    last_loss = 0.0
    val_pos = (
        np.stack(
            [
                _ctde_transform_features(row, transform)
                for row in pair_data.get("validation_positives", [])
            ]
        ).astype(np.float32)
        if pair_data.get("validation_positives")
        else None
    )
    val_neg = (
        np.stack(
            [
                _ctde_transform_features(row, transform)
                for row in pair_data.get("validation_negatives", [])
            ]
        ).astype(np.float32)
        if pair_data.get("validation_negatives")
        else None
    )
    val_weight = np.array(pair_data.get("validation_sample_weights", []), dtype=np.float32)
    if val_pos is not None and val_neg is not None and len(val_pos) > 0:
        val_pos = ((val_pos - input_mean) / input_std).astype(np.float32)
        val_neg = ((val_neg - input_mean) / input_std).astype(np.float32)
        val_pos_tensor = torch.from_numpy(val_pos).to(torch_device)
        val_neg_tensor = torch.from_numpy(val_neg).to(torch_device)
        val_weight_tensor = torch.from_numpy(np.maximum(val_weight, 0.05)).to(torch_device)
    else:
        val_pos_tensor = None
        val_neg_tensor = None
        val_weight_tensor = None
    patience = max(0, int(early_stopping_patience or 0))
    min_delta = max(0.0, float(early_stopping_min_delta or 0.0))
    scheduler = None
    if int(lr_scheduler_patience or 0) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=max(0.05, min(0.95, float(lr_scheduler_factor or 0.5))),
            patience=max(1, int(lr_scheduler_patience)),
        )
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_run = 0
    epochs_without_improvement = 0

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            if (
                val_pos_tensor is not None
                and val_neg_tensor is not None
                and val_weight_tensor is not None
            ):
                diff = model(val_pos_tensor).squeeze(-1) - model(val_neg_tensor).squeeze(-1)
                loss_terms = functional.softplus(-(diff - float(margin))) * val_weight_tensor
                return float(loss_terms.mean().detach().cpu().item())
        return float(last_loss)

    for epoch in range(max(1, int(epochs))):
        model.train()
        permutation = rng.permutation(pair_count)
        for start in range(0, pair_count, batch_size):
            idx = torch.as_tensor(
                permutation[start : start + batch_size],
                dtype=torch.long,
                device=torch_device,
            )
            pos_score = model(pos_tensor[idx]).squeeze(-1)
            neg_score = model(neg_tensor[idx]).squeeze(-1)
            diff = pos_score - neg_score
            loss_terms = functional.softplus(-(diff - float(margin))) * weight_tensor[idx]
            loss = loss_terms.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            updates += 1
        epochs_run = epoch + 1
        val_loss = validation_loss()
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epochs_run
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    exported_layers = []
    exported_biases = []
    for module in model:
        if isinstance(module, torch.nn.Linear):
            exported_layers.append(module.weight.detach().cpu().numpy().astype(float).tolist())
            exported_biases.append(module.bias.detach().cpu().numpy().astype(float).tolist())

    metadata = {
        "dataset_path": dataset_path,
        "results_dir": results_dir,
        "outcomes_path": outcomes_path,
        "epochs": int(epochs),
        "epochs_run": int(epochs_run),
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss if np.isfinite(best_val_loss) else last_loss),
        "learning_rate": float(learning_rate),
        "margin": float(margin),
        "objective": objective,
        "loss_weight": float(loss_weight),
        "max_win_alternatives": int(max_win_alternatives),
        "max_loss_alternatives": int(max_loss_alternatives),
        "hidden_sizes": list(hidden),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "transform": transform,
        "activation": activation,
        "dropout": float(dropout_rate),
        "lr_scheduler_patience": int(lr_scheduler_patience or 0),
        "lr_scheduler_factor": float(lr_scheduler_factor or 0.5),
        "benchmarks": sorted(allowed_benchmarks) if allowed_benchmarks else [],
        "seed": int(seed),
        "max_pairs": int(max_pairs),
        "device_requested": requested_device,
        "device_used": str(torch_device),
        "cuda_name": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else "",
        "pair_examples": int(pair_count),
        "last_loss": float(last_loss),
        "pair_stats": _public_split_stats(pair_data["stats"]),
    }
    payload = {
        "model_type": "mlp",
        "feature_names": CTDE_FEATURE_NAMES,
        "transform": transform,
        "input_feature_names": _ctde_transformed_feature_names(transform),
        "input_mean": input_mean.astype(float).tolist(),
        "input_std": input_std.astype(float).tolist(),
        "layers": exported_layers,
        "biases": exported_biases,
        "activation": activation,
        "examples": int(pair_count),
        "updates": int(updates),
        "metadata": metadata,
    }
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    eval_stats = evaluate_ctde_joint_reranker_records(
        pair_data["eval_records"], outcomes, CTDEJointReranker(output_path)
    )
    payload["metadata"]["eval"] = eval_stats
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _write_ctde_split_manifest(split_path, pair_data["stats"], dataset_path, outcomes_path)
    return {
        "examples": int(pair_count),
        "updates": int(updates),
        "objective": objective,
        "output_path": output_path,
        "output_written": True,
        "eval": eval_stats,
        "pair_stats": _public_split_stats(pair_data["stats"]),
    }


__all__ = ["train_ctde_joint_mlp_reranker"]
