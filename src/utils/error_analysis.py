# src/utils/error_analysis.py
"""
Error analysis utilities for hallucination detection experiments.

Provides functions to:
- Identify false positives and false negatives
- Extract high-confidence errors
- Generate error analysis reports
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def add_error_flags(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Add error indicator columns to a predictions dataframe.

    Adds three boolean columns:
    - correct: True if y_true == y_pred
    - fp: True if false positive (predicted 1, true 0)
    - fn: True if false negative (predicted 0, true 1)

    Args:
        df_pred: Predictions dataframe with y_true and y_pred columns

    Returns:
        Copy of df_pred with error flag columns added

    Example:
        df_pred = pd.DataFrame({
            'y_true': [0, 1, 1, 0],
            'y_pred': [0, 0, 1, 1]
        })
        df_with_flags = add_error_flags(df_pred)
        # df_with_flags has columns: y_true, y_pred, correct, fp, fn
    """
    if "y_true" not in df_pred.columns or "y_pred" not in df_pred.columns:
        raise ValueError("df_pred must contain 'y_true' and 'y_pred' columns")

    df_out = df_pred.copy()

    df_out["correct"] = (df_out["y_true"] == df_out["y_pred"]).astype(int)
    df_out["fp"] = ((df_out["y_true"] == 0) & (df_out["y_pred"] == 1)).astype(int)
    df_out["fn"] = ((df_out["y_true"] == 1) & (df_out["y_pred"] == 0)).astype(int)

    return df_out


def compute_confidence(score: pd.Series) -> pd.Series:
    """
    Compute confidence from model scores.

    Two strategies:
    1. If score appears to be probability (all in [0,1]), confidence = max(score, 1-score)
    2. Otherwise (decision scores), confidence = abs(score)

    Args:
        score: Series of model scores

    Returns:
        Series of confidence values (higher = more confident)

    Example:
        # Probabilities
        scores = pd.Series([0.9, 0.2, 0.7, 0.1])
        conf = compute_confidence(scores)
        # Returns: [0.9, 0.8, 0.7, 0.9]

        # Decision scores
        scores = pd.Series([2.5, -1.3, 0.8, -2.1])
        conf = compute_confidence(scores)
        # Returns: [2.5, 1.3, 0.8, 2.1]
    """
    score_array = np.asarray(score)

    # Check if scores look like probabilities (all in [0, 1])
    is_probability = np.all((score_array >= 0) & (score_array <= 1))

    if is_probability:
        # For probabilities: confidence = max(p, 1-p)
        return pd.Series(np.maximum(score_array, 1 - score_array), index=score.index)
    else:
        # For decision scores: confidence = abs(score)
        return pd.Series(np.abs(score_array), index=score.index)


def extract_top_errors(
    df_pred: pd.DataFrame,
    kind: Literal["fp", "fn"],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Extract top-K errors by confidence.

    Args:
        df_pred: Predictions dataframe with fp/fn flags and score column
        kind: Error type to extract ("fp" or "fn")
        top_k: Number of top errors to return

    Returns:
        DataFrame with top-K errors, sorted by confidence descending

    Raises:
        ValueError: If kind not in {"fp", "fn"} or required columns missing

    Example:
        # Load predictions
        df_pred = pd.read_csv("predictions_val.csv")

        # Get top 20 false positives
        top_fp = extract_top_errors(df_pred, kind="fp", top_k=20)

        # Get top 10 false negatives
        top_fn = extract_top_errors(df_pred, kind="fn", top_k=10)
    """
    if kind not in {"fp", "fn"}:
        raise ValueError(f"kind must be 'fp' or 'fn', got: {kind}")

    required_cols = {"y_true", "y_pred", "score", kind}
    missing = required_cols - set(df_pred.columns)
    if missing:
        raise ValueError(f"df_pred missing required columns: {missing}")

    # Filter to error type
    df_errors = df_pred[df_pred[kind] == 1].copy()

    if len(df_errors) == 0:
        # No errors of this type
        return pd.DataFrame()

    # Compute confidence
    df_errors["confidence"] = compute_confidence(df_errors["score"])

    # Sort by confidence descending and take top-K
    df_top = df_errors.sort_values("confidence", ascending=False).head(top_k)

    return df_top.reset_index(drop=True)


def generate_error_report(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    output_path: Path | str,
    top_n: int = 10
) -> None:
    """
    Generate markdown error analysis report.

    Creates a report with:
    - Error counts summary
    - Top-N false positives table
    - Top-N false negatives table
    - Placeholder sections for thematic analysis

    Args:
        fp_df: False positives dataframe (from extract_top_errors)
        fn_df: False negatives dataframe (from extract_top_errors)
        output_path: Path to write markdown report
        top_n: Number of examples to include in tables

    Example:
        top_fp = extract_top_errors(df_pred, "fp", top_k=20)
        top_fn = extract_top_errors(df_pred, "fn", top_k=20)
        generate_error_report(top_fp, top_fn, "reports/error_analysis.md")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics
    n_fp = len(fp_df)
    n_fn = len(fn_df)
    n_errors = n_fp + n_fn

    # Build report
    lines = [
        "# Error Analysis Report",
        "",
        "## Summary Statistics",
        "",
        f"- **Total errors**: {n_errors}",
        f"- **False positives (FP)**: {n_fp} (predicted hallucination, true non-hallucination)",
        f"- **False negatives (FN)**: {n_fn} (predicted non-hallucination, true hallucination)",
        "",
    ]

    # False Positives section
    lines.extend([
        f"## Top {top_n} False Positives (by confidence)",
        "",
        "Model incorrectly predicted hallucination with high confidence.",
        "",
    ])

    if len(fp_df) > 0:
        # Select columns for display (if they exist)
        display_cols = []
        for col in ["id", "task", "score", "confidence", "y_true", "y_pred"]:
            if col in fp_df.columns:
                display_cols.append(col)

        fp_display = fp_df.head(top_n)[display_cols]

        # Format as markdown table
        lines.append(fp_display.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("*No false positives found.*")
        lines.append("")

    # False Negatives section
    lines.extend([
        f"## Top {top_n} False Negatives (by confidence)",
        "",
        "Model incorrectly predicted non-hallucination with high confidence.",
        "",
    ])

    if len(fn_df) > 0:
        display_cols = []
        for col in ["id", "task", "score", "confidence", "y_true", "y_pred"]:
            if col in fn_df.columns:
                display_cols.append(col)

        fn_display = fn_df.head(top_n)[display_cols]

        lines.append(fn_display.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("*No false negatives found.*")
        lines.append("")

    # Thematic analysis placeholders
    lines.extend([
        "## Error Themes (Manual Analysis)",
        "",
        "### Numbers and Facts",
        "- *[Examine whether errors involve numerical values, dates, or factual claims]*",
        "",
        "### Named Entities",
        "- *[Check if errors involve person names, locations, organizations]*",
        "",
        "### Causality and Reasoning",
        "- *[Look for logical inconsistencies or causal reasoning errors]*",
        "",
        "### Fluent Nonsense",
        "- *[Identify grammatically correct but semantically meaningless responses]*",
        "",
        "### Context Misalignment",
        "- *[Find cases where response doesn't match prompt context]*",
        "",
        "### Other Patterns",
        "- *[Note any other recurring patterns in errors]*",
        "",
    ])

    # Write report
    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
