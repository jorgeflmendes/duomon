from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import ensure_parent_dir
from .ctde_data import (
    _binary_auc,
    _collect_ctde_value_examples,
    _public_split_stats,
    _write_ctde_split_manifest,
)
from .ctde_eval import evaluate_ctde_joint_reranker_records
from .ctde_features import (
    CTDE_FEATURE_NAMES,
    _ctde_transform_features,
    _ctde_transformed_feature_names,
)
from .ctde_models import CTDEJointReranker


def _train_ctde_value_regression(
    dataset_path: str,
    output_path: str,
    outcomes: Dict[str, bool],
    results_dir: str,
    outcomes_path: str,
    allowed_benchmarks: Optional[set[str]],
    epochs: int,
    learning_rate: float,
    loss_weight: float,
    max_candidates_per_record: int,
    hidden_sizes: Sequence[int],
    batch_size: int,
    weight_decay: float,
    transform: str,
    activation: str,
    seed: int,
    max_pairs: int,
    device: str,
    validation_fraction: float,
    split_path: str,
    allow_empty_overwrite: bool,
    dropout: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lr_scheduler_patience: int,
    lr_scheduler_factor: float,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except Exception as exc:
        raise RuntimeError("PyTorch is required to train the CTDE MLP offline.") from exc

    data = _collect_ctde_value_examples(
        dataset_path=dataset_path,
        outcomes=outcomes,
        max_candidates_per_record=max_candidates_per_record,
        benchmarks=allowed_benchmarks,
        validation_fraction=validation_fraction,
        split_seed=seed,
    )
    inputs = data["inputs"]
    labels = data["labels"]
    sample_weights = data["sample_weights"]
    n_examples = len(inputs)
    if n_examples <= 0:
        _write_ctde_split_manifest(split_path, data["stats"], dataset_path, outcomes_path)
        result = {
            "examples": 0,
            "updates": 0,
            "output_path": output_path,
            "eval": {},
            "value_stats": _public_split_stats(data["stats"]),
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
                "objective": "value_regression",
                "benchmarks": sorted(allowed_benchmarks) if allowed_benchmarks else [],
                "value_stats": _public_split_stats(data["stats"]),
                "eval": {},
            },
        }
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        result["output_written"] = True
        return result

    rng = np.random.default_rng(int(seed))
    if max_pairs and int(max_pairs) > 0 and n_examples > int(max_pairs):
        keep = rng.choice(n_examples, size=int(max_pairs), replace=False)
        inputs = [inputs[int(i)] for i in keep]
        labels = [labels[int(i)] for i in keep]
        sample_weights = [sample_weights[int(i)] for i in keep]
        n_examples = len(inputs)

    raw = np.stack([_ctde_transform_features(row, transform) for row in inputs]).astype(np.float32)
    input_mean = raw.mean(axis=0).astype(np.float32)
    input_std = raw.std(axis=0).astype(np.float32)
    input_std[input_std < 1.0e-4] = 1.0
    x = ((raw - input_mean) / input_std).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    w = np.array(sample_weights, dtype=np.float32)

    pos_count = float(max(1.0, y.sum()))
    neg_count = float(max(1.0, len(y) - y.sum()))
    pos_weight_value = neg_count / pos_count
    w = np.where(y > 0.5, w * pos_weight_value * float(loss_weight), w)
    w = np.maximum(w, 0.05)

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

    hidden = tuple(int(size) for size in hidden_sizes if int(size) > 0)
    activation = str(activation or "tanh").strip().lower()
    if activation not in {"tanh", "relu"}:
        activation = "tanh"
    activation_module = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
    modules: List[Any] = []
    previous_size = int(x.shape[1])
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

    x_tensor = torch.from_numpy(x).to(torch_device)
    y_tensor = torch.from_numpy(y).to(torch_device)
    w_tensor = torch.from_numpy(w).to(torch_device)
    batch_size = max(16, int(batch_size))
    updates = 0
    last_loss = 0.0
    val_inputs = data.get("validation_inputs", [])
    if val_inputs:
        val_raw = np.stack([_ctde_transform_features(row, transform) for row in val_inputs]).astype(
            np.float32
        )
        val_x = ((val_raw - input_mean) / input_std).astype(np.float32)
        val_y = np.array(data.get("validation_labels", []), dtype=np.float32)
        val_w = np.maximum(
            np.array(data.get("validation_sample_weights", []), dtype=np.float32), 0.05
        )
        val_x_tensor = torch.from_numpy(val_x).to(torch_device)
        val_y_tensor = torch.from_numpy(val_y).to(torch_device)
        val_w_tensor = torch.from_numpy(val_w).to(torch_device)
    else:
        val_x_tensor = None
        val_y_tensor = None
        val_w_tensor = None
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
            if val_x_tensor is not None and val_y_tensor is not None and val_w_tensor is not None:
                logits = model(val_x_tensor).squeeze(-1)
                loss_terms = (
                    functional.binary_cross_entropy_with_logits(
                        logits, val_y_tensor, reduction="none"
                    )
                    * val_w_tensor
                )
                return float(loss_terms.mean().detach().cpu().item())
        return float(last_loss)

    for epoch in range(max(1, int(epochs))):
        model.train()
        permutation = rng.permutation(n_examples)
        for start in range(0, n_examples, batch_size):
            idx = torch.as_tensor(
                permutation[start : start + batch_size],
                dtype=torch.long,
                device=torch_device,
            )
            logits = model(x_tensor[idx]).squeeze(-1)
            target = y_tensor[idx]
            sample_w = w_tensor[idx]
            loss_terms = (
                functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                * sample_w
            )
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
        "objective": "value_regression",
        "loss_weight": float(loss_weight),
        "max_candidates_per_record": int(max_candidates_per_record),
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
        "examples": int(n_examples),
        "last_loss": float(last_loss),
        "value_stats": _public_split_stats(data["stats"]),
        "pos_weight_used": float(pos_weight_value),
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
        "examples": int(n_examples),
        "updates": int(updates),
        "metadata": metadata,
    }
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    eval_stats = evaluate_ctde_joint_reranker_records(
        data["eval_records"], outcomes, CTDEJointReranker(output_path)
    )

    with torch.no_grad():
        all_logits = model(x_tensor).squeeze(-1).cpu().numpy()
    auc = _binary_auc(all_logits, y) if len(y) > 0 else 0.0
    eval_stats["train_auc"] = float(auc)
    payload["metadata"]["eval"] = eval_stats
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _write_ctde_split_manifest(split_path, data["stats"], dataset_path, outcomes_path)
    return {
        "examples": int(n_examples),
        "updates": int(updates),
        "objective": "value_regression",
        "output_path": output_path,
        "output_written": True,
        "eval": eval_stats,
        "value_stats": _public_split_stats(data["stats"]),
    }


__all__ = ["_train_ctde_value_regression"]
