# src/models/embedding_baseline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EmbeddingLRConfig:
    """
    Linear baseline on embeddings.
    We avoid heavy tuning to keep comparison fair and academic.
    """
    C: float = 1.0
    max_iter: int = 2000
    class_weight: Optional[str] = "balanced"  # set None if your labels are balanced
    random_state: int = 42
    n_jobs: int = -1


def build_embedding_lr(cfg: EmbeddingLRConfig = EmbeddingLRConfig()) -> Pipeline:
    """
    Logistic Regression on embeddings.
    Note: StandardScaler can help if embeddings are not normalized.
    If you normalize embeddings in extraction, scaler has less effect but is harmless.
    """
    lr = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),  # safe for sparse-like; embeddings are dense but ok
            ("clf", lr),
        ]
    )


def concat_embeddings_and_numeric(emb: np.ndarray, numeric: np.ndarray) -> np.ndarray:
    """
    Concatenate dense embeddings with numeric features.
    """
    if emb.ndim != 2 or numeric.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got emb.shape={emb.shape}, numeric.shape={numeric.shape}")
    if emb.shape[0] != numeric.shape[0]:
        raise ValueError(f"Row mismatch: emb.shape[0]={emb.shape[0]} vs numeric.shape[0]={numeric.shape[0]}")
    return np.hstack([emb.astype(np.float32, copy=False), numeric.astype(np.float32, copy=False)])
