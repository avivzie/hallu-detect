# src/utils/eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "split": self.split,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }
        if self.roc_auc is not None:
            result["roc_auc"] = self.roc_auc
        return result


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


def evaluate_split_with_roc(
    name: str,
    model,
    X,
    y,
    scores: Optional[np.ndarray] = None,
    verbose: bool = True
) -> SplitMetrics:
    """
    Evaluate on a split with optional ROC-AUC computation.

    Extends evaluate_split() by adding ROC-AUC when scores are available.

    Args:
        name: Split name for display
        model: Trained model (must implement predict)
        X: Input features
        y: True labels
        scores: Optional pre-computed scores (e.g., from predict_proba).
                If None and model has predict_proba, will compute automatically.
                If not available, roc_auc will be None.
        verbose: If True, print metrics

    Returns:
        SplitMetrics with optional roc_auc field

    Example:
        # With pre-computed scores
        scores = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_split_with_roc("val", model, X_val, y_val, scores=scores)

        # Auto-compute from model
        metrics = evaluate_split_with_roc("val", model, X_val, y_val)
    """
    y_true = np.asarray(y).astype(int)
    y_pred = model.predict(X)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    p = float(precision_score(y_true, y_pred))
    r = float(recall_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    # Compute ROC-AUC if scores available
    roc_auc = None
    if scores is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, scores))
        except Exception:
            pass  # If computation fails, leave as None
    elif hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            roc_auc = float(roc_auc_score(y_true, proba[:, 1]))
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X)
            roc_auc = float(roc_auc_score(y_true, decision))
        except Exception:
            pass

    if verbose:
        print(f"\n{name} metrics:")
        print(f"  accuracy = {acc:.4f}")
        print(f"  f1       = {f1:.4f}")
        print(f"  precision= {p:.4f}")
        print(f"  recall   = {r:.4f}")
        if roc_auc is not None:
            print(f"  roc_auc  = {roc_auc:.4f}")
        print("  confusion matrix:")
        print(cm)

    return SplitMetrics(split=name, accuracy=acc, f1=f1, precision=p, recall=r, roc_auc=roc_auc)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path | str] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues"
) -> None:
    """
    Generate confusion matrix heatmap using matplotlib.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: If provided, save figure to this path
        title: Plot title
        cmap: Colormap name

    Example:
        plot_confusion_matrix(y_val, y_pred, save_path="reports/cm_val.png")
    """
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Add labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[0, 1],
           yticklabels=[0, 1],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[Path | str] = None,
    title: str = "Precision-Recall Curve"
) -> None:
    """
    Generate precision-recall curve using matplotlib.

    Args:
        y_true: True binary labels
        y_scores: Target scores (e.g., predict_proba[:, 1])
        save_path: If provided, save figure to this path
        title: Plot title

    Example:
        scores = model.predict_proba(X_val)[:, 1]
        plot_pr_curve(y_val, scores, save_path="reports/pr_val.png")
    """
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label='PR curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
