from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import AgentConfig, logger
from .ctde_features import CTDE_FEATURE_NAMES, _ctde_transform_features


class CTDEJointReranker:

    FEATURE_NAMES = CTDE_FEATURE_NAMES

    def __init__(self, path: str):
        self.path = path
        self.model_type = "linear"
        self.transform = "raw"
        self.weights = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
        self.layers: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.activation = "tanh"
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.examples = 0
        self.updates = 0
        self.metadata: Dict[str, Any] = {}
        self.objective: str = "outcome_margin"
        self.available = False
        self.load()

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(self.FEATURE_NAMES):
            return 0.0
        if self.model_type == "mlp":
            return self._predict_mlp(features)
        if len(features) != len(self.weights):
            return 0.0
        return float(np.dot(self.weights, np.array(features, dtype=np.float32)))

    def _predict_mlp(self, features: Sequence[float]) -> float:
        if not self.layers or len(self.layers) != len(self.biases):
            return 0.0
        x = _ctde_transform_features(features, self.transform)
        if (
            self.input_mean is not None
            and self.input_std is not None
            and len(x) == len(self.input_mean)
        ):
            x = (x - self.input_mean) / np.maximum(self.input_std, 1.0e-6)
        for idx, (weights, bias) in enumerate(zip(self.layers, self.biases)):
            if weights.ndim != 2 or weights.shape[1] != len(x) or weights.shape[0] != len(bias):
                return 0.0
            x = weights @ x + bias
            if idx < len(self.layers) - 1:
                if self.activation == "relu":
                    x = np.maximum(x, 0.0)
                else:
                    x = np.tanh(x)
        if len(x) != 1:
            return 0.0
        return float(x[0])

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("feature_names") != self.FEATURE_NAMES:
                return
            self.model_type = str(data.get("model_type", "linear") or "linear").strip().lower()
            if self.model_type == "mlp":
                self.transform = str(data.get("transform", "raw") or "raw").strip().lower()
                raw_layers = data.get("layers", [])
                raw_biases = data.get("biases", [])
                layers = [
                    np.array(layer, dtype=np.float32)
                    for layer in raw_layers
                    if isinstance(layer, list)
                ]
                biases = [
                    np.array(bias, dtype=np.float32)
                    for bias in raw_biases
                    if isinstance(bias, list)
                ]
                if not layers or len(layers) != len(biases):
                    self.model_type = "linear"
                    return
                input_size = len(
                    _ctde_transform_features(
                        np.zeros(len(self.FEATURE_NAMES), dtype=np.float32),
                        self.transform,
                    )
                )
                if layers[0].ndim != 2 or layers[0].shape[1] != input_size:
                    self.model_type = "linear"
                    return
                self.layers = layers
                self.biases = biases
                self.activation = str(data.get("activation", "tanh") or "tanh").strip().lower()
                raw_mean = data.get("input_mean")
                raw_std = data.get("input_std")
                if (
                    isinstance(raw_mean, list)
                    and isinstance(raw_std, list)
                    and len(raw_mean) == input_size
                    and len(raw_std) == input_size
                ):
                    self.input_mean = np.array(raw_mean, dtype=np.float32)
                    self.input_std = np.array(raw_std, dtype=np.float32)
                self.examples = int(data.get("examples", 0) or 0)
                self.updates = int(data.get("updates", 0) or 0)
                self.metadata = dict(data.get("metadata", {}) or {})
                self.objective = (
                    str(self.metadata.get("objective", "outcome_margin") or "outcome_margin")
                    .strip()
                    .lower()
                )
                self.available = True
                logger.info(
                    f"[model] action=loaded type=ctde_joint_reranker_mlp path={self.path} objective={self.objective}"
                )
                return
            self.model_type = "linear"
            raw_weights = data.get("weights", [])
            if len(raw_weights) != len(self.weights):
                return
            self.weights = np.array(raw_weights, dtype=np.float32)
            self.examples = int(data.get("examples", 0) or 0)
            self.updates = int(data.get("updates", 0) or 0)
            self.metadata = dict(data.get("metadata", {}) or {})
            self.available = True
            logger.info(f"[model] action=loaded type=ctde_joint_reranker_linear path={self.path}")
        except Exception as exc:
            logger.info(f"Failed to load CTDE joint reranker: {exc}")


def make_ctde_joint_reranker(config: AgentConfig):
    if not getattr(config, "use_ctde_joint_reranker", False):
        return None
    if float(getattr(config, "ctde_joint_reranker_weight", 0.0) or 0.0) <= 0.0:
        return None

    path = getattr(config, "ctde_joint_reranker_path", "")
    if path:
        reranker = CTDEJointReranker(path)
        if getattr(reranker, "available", False):
            logger.info(f"[model] action=factory type=ctde_single path={path}")
            return reranker
        logger.info(f"[model] action=missing type=ctde_single path={path}")
    return None


__all__ = ["CTDEJointReranker", "make_ctde_joint_reranker"]
