# src/utils/experiment.py
"""
Experiment utilities for reproducible research workflows.

Provides helpers for:
- Setting random seeds
- Creating standardized output directories
- Saving metrics, predictions, and reports
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def make_run_dir(
    reports_dir: Path | str,
    run_name: str,
    timestamp: bool = True
) -> Path:
    """
    Create experiment output directory with optional timestamping.

    Args:
        reports_dir: Base reports directory (e.g., Path("reports"))
        run_name: Experiment name (e.g., "nb02_baseline")
        timestamp: If True, append timestamp to run_name for unique runs.
                   If False, use stable path (overwrites previous runs).

    Returns:
        Path to created directory

    Example:
        # Timestamped (for new experiments):
        run_dir = make_run_dir(Path("reports"), "nb02_baseline")
        # -> reports/nb02_baseline_20260130_120000/

        # Stable path (for consolidated results):
        run_dir = make_run_dir(Path("reports"), "nb02_baseline", timestamp=False)
        # -> reports/nb02_baseline/
    """
    reports_dir = Path(reports_dir)

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = reports_dir / f"{run_name}_{ts}"
    else:
        run_dir = reports_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics_csv(
    path: Path | str,
    metrics_dicts: List[Dict[str, Any]]
) -> None:
    """
    Save metrics dictionaries to CSV.

    Args:
        path: Output CSV path
        metrics_dicts: List of metric dictionaries (e.g., from SplitMetrics.as_dict())

    Example:
        metrics = [
            {"split": "train", "accuracy": 0.95, "f1": 0.94},
            {"split": "val", "accuracy": 0.90, "f1": 0.89},
            {"split": "test", "accuracy": 0.91, "f1": 0.90},
        ]
        save_metrics_csv(Path("reports/metrics.csv"), metrics)
    """
    path = Path(path)
    df = pd.DataFrame(metrics_dicts)
    df.to_csv(path, index=False)
    print(f"Saved metrics to: {path}")


def save_markdown_table(
    path: Path | str,
    df: pd.DataFrame,
    float_format: str = ".4f"
) -> None:
    """
    Save DataFrame as markdown table.

    Args:
        path: Output markdown path
        df: DataFrame to save
        float_format: Format string for floating point numbers

    Example:
        save_markdown_table(Path("reports/results.md"), results_df)
    """
    path = Path(path)

    # Format numeric columns
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if pd.api.types.is_numeric_dtype(df_formatted[col]):
            if pd.api.types.is_float_dtype(df_formatted[col]):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:{float_format}}" if pd.notna(x) else ""
                )

    md_table = df_formatted.to_markdown(index=False)
    path.write_text(md_table, encoding="utf-8")
    print(f"Saved markdown table to: {path}")


def get_model_scores(model, X) -> np.ndarray:
    """
    Extract prediction scores from a model.

    Tries predict_proba first (returns probability for positive class),
    then decision_function, then falls back to binary predictions.

    Args:
        model: Trained sklearn-compatible model
        X: Input features

    Returns:
        1D array of scores (higher = more confident in positive class)
    """
    # Try predict_proba (most common for classifiers)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Return probability for positive class (column 1)
        return proba[:, 1]

    # Try decision_function (e.g., LinearSVC, SGDClassifier)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalize to [0, 1] range for consistency
        scores = np.asarray(scores).ravel()
        if len(scores) > 1:
            scores_min, scores_max = scores.min(), scores.max()
            if scores_max > scores_min:
                scores = (scores - scores_min) / (scores_max - scores_min)
        return scores

    # Fallback: binary predictions (0 or 1, no confidence)
    return model.predict(X).astype(float)


def save_predictions_csv(
    path: Path | str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    split: str,
    id_cols: Optional[List[str]] = None
) -> None:
    """
    Save predictions with metadata for error analysis.

    Args:
        path: Output CSV path
        df: Source dataframe with metadata (id, task, group_id, etc.)
        y_true: True labels
        y_pred: Predicted labels
        scores: Model confidence scores (from get_model_scores or predict_proba)
        split: Split name (e.g., "train", "val", "test")
        id_cols: Columns to include from df (default: ["id", "task", "group_id"])

    Example:
        scores = get_model_scores(model, X_val)
        y_pred = (scores >= 0.5).astype(int)

        save_predictions_csv(
            path=run_dir / "predictions_val.csv",
            df=val_df,
            y_true=y_val,
            y_pred=y_pred,
            scores=scores,
            split="val"
        )
    """
    path = Path(path)

    if id_cols is None:
        id_cols = ["id", "task", "group_id"]

    # Build predictions dataframe
    pred_data = {
        "split": split,
        "y_true": y_true,
        "y_pred": y_pred,
        "score": scores,
    }

    # Add metadata columns from source df
    for col in id_cols:
        if col in df.columns:
            pred_data[col] = df[col].values

    pred_df = pd.DataFrame(pred_data)

    # Add error indicators
    pred_df["correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df["fp"] = ((pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1)).astype(int)
    pred_df["fn"] = ((pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0)).astype(int)

    # Sort by score descending (most confident predictions first)
    pred_df = pred_df.sort_values("score", ascending=False).reset_index(drop=True)

    pred_df.to_csv(path, index=False)
    print(f"Saved predictions to: {path}")
    print(f"  Total: {len(pred_df)}, Correct: {pred_df['correct'].sum()}, "
          f"FP: {pred_df['fp'].sum()}, FN: {pred_df['fn'].sum()}")
