# src/utils/eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    accuracy: float
    f1: float
    precision: float
    recall: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }


def evaluate_split(name: str, model, X, y, verbose: bool = True) -> SplitMetrics:
    """
    Evaluate on a split and optionally print key classification metrics.
    `model` must implement predict(X).
    """
    y_true = np.asarray(y).astype(int)
    y_pred = model.predict(X)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    p = float(precision_score(y_true, y_pred))
    r = float(recall_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"\n{name} metrics:")
        print(f"  accuracy = {acc:.4f}")
        print(f"  f1       = {f1:.4f}")
        print(f"  precision= {p:.4f}")
        print(f"  recall   = {r:.4f}")
        print("  confusion matrix:")
        print(cm)

    return SplitMetrics(split=name, accuracy=acc, f1=f1, precision=p, recall=r)

def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Convenience helper to present results as a DataFrame."""
    return pd.DataFrame(rows)
