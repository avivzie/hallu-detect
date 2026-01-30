# src/utils/ablation.py
"""
Feature ablation utilities for hallucination detection experiments.

Provides leave-one-out and add-one-at-a-time ablation analysis
to measure feature importance via performance delta.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.eval import evaluate_split


def run_single_feature_ablation(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    numeric_cols: List[str],
    text_col: str,
    model_builder: Callable[[List[str], str], any],
    seed: int = 42
) -> pd.DataFrame:
    """
    Leave-one-out numeric feature ablation.

    For each numeric feature f:
    1. Train baseline WITHOUT f
    2. Evaluate on validation set
    3. Compute ΔF1 = F1_without_f - F1_full

    Negative ΔF1 means removing the feature hurts performance (feature is useful).
    Positive ΔF1 means removing the feature helps (feature is harmful/redundant).

    Args:
        train_df: Training dataframe with text and numeric columns
        y_train: Training labels
        val_df: Validation dataframe
        y_val: Validation labels
        numeric_cols: List of numeric feature column names
        text_col: Name of text column (e.g., "response")
        model_builder: Function that takes (numeric_cols, text_col) and returns
                       a fitted-ready sklearn pipeline
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
        - feature_name: Name of ablated feature
        - f1_full: F1 with all features
        - f1_without: F1 without this feature
        - delta_f1: f1_without - f1_full (negative = feature helps)

    Example:
        def build_model(numeric_cols, text_col):
            # Build TF-IDF + numeric pipeline
            return pipeline

        results = run_single_feature_ablation(
            train_df, y_train, val_df, y_val,
            numeric_cols=["resp_n_chars", "resp_n_words"],
            text_col="response",
            model_builder=build_model
        )
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 1. Train full model (with all features)
    print("Training full model (all features)...")
    full_model = model_builder(numeric_cols, text_col)
    full_model.fit(train_df, y_train)

    full_metrics = evaluate_split("Val (full)", full_model, val_df, y_val, verbose=False)
    f1_full = full_metrics.f1

    print(f"  Full model F1: {f1_full:.4f}")

    # 2. Ablate each feature
    results = []

    print(f"\nRunning leave-one-out ablation over {len(numeric_cols)} features...")
    for feature in tqdm(numeric_cols, desc="Ablating features"):
        # Create feature list without current feature
        ablated_cols = [c for c in numeric_cols if c != feature]

        # Train model without this feature
        ablated_model = model_builder(ablated_cols, text_col)
        ablated_model.fit(train_df, y_train)

        # Evaluate
        ablated_metrics = evaluate_split(
            f"Val (without {feature})",
            ablated_model,
            val_df,
            y_val,
            verbose=False
        )
        f1_without = ablated_metrics.f1

        # Compute delta
        delta_f1 = f1_without - f1_full

        results.append({
            "feature_name": feature,
            "f1_full": f1_full,
            "f1_without": f1_without,
            "delta_f1": delta_f1
        })

    results_df = pd.DataFrame(results)

    # Sort by absolute impact (most impactful first)
    results_df = results_df.sort_values("delta_f1", key=abs, ascending=False)
    results_df = results_df.reset_index(drop=True)

    return results_df


def run_add_one_feature_ablation(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    numeric_cols: List[str],
    text_col: str,
    model_builder: Callable[[List[str], str], any],
    seed: int = 42
) -> pd.DataFrame:
    """
    Add-one-at-a-time numeric feature ablation.

    Start with text-only baseline, then add features one by one
    to measure marginal F1 gain.

    Args:
        train_df: Training dataframe with text and numeric columns
        y_train: Training labels
        val_df: Validation dataframe
        y_val: Validation labels
        numeric_cols: List of numeric feature column names (order matters)
        text_col: Name of text column (e.g., "response")
        model_builder: Function that takes (numeric_cols, text_col) and returns
                       a fitted-ready sklearn pipeline
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
        - feature_name: Name of added feature
        - f1_before: F1 before adding this feature
        - f1_after: F1 after adding this feature
        - delta_f1: f1_after - f1_before (positive = feature helps)

    Example:
        results = run_add_one_feature_ablation(
            train_df, y_train, val_df, y_val,
            numeric_cols=["resp_n_chars", "resp_n_words"],
            text_col="response",
            model_builder=build_model
        )
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 1. Train text-only baseline
    print("Training text-only baseline...")
    textonly_model = model_builder([], text_col)
    textonly_model.fit(train_df, y_train)

    textonly_metrics = evaluate_split("Val (text-only)", textonly_model, val_df, y_val, verbose=False)
    f1_textonly = textonly_metrics.f1

    print(f"  Text-only F1: {f1_textonly:.4f}")

    # 2. Add features one by one
    results = []
    current_cols = []
    current_f1 = f1_textonly

    print(f"\nAdding features one by one ({len(numeric_cols)} features)...")
    for feature in tqdm(numeric_cols, desc="Adding features"):
        # Add current feature to the set
        current_cols.append(feature)

        # Train model with current feature set
        model = model_builder(current_cols, text_col)
        model.fit(train_df, y_train)

        # Evaluate
        metrics = evaluate_split(
            f"Val (with {len(current_cols)} features)",
            model,
            val_df,
            y_val,
            verbose=False
        )
        new_f1 = metrics.f1

        # Compute delta
        delta_f1 = new_f1 - current_f1

        results.append({
            "feature_name": feature,
            "f1_before": current_f1,
            "f1_after": new_f1,
            "delta_f1": delta_f1
        })

        # Update current F1 for next iteration
        current_f1 = new_f1

    results_df = pd.DataFrame(results)
    return results_df
