# src/data/load_splits.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED_COLS = ["prompt", "response", "label", "task"]


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path


def default_split_paths(root: Path) -> SplitPaths:
    """Default locations for processed splits relative to repo root."""
    data_dir = root / "data_processed"
    return SplitPaths(
        train=data_dir / "train.csv",
        val=data_dir / "val.csv",
        test=data_dir / "test.csv",
    )


def _validate_df(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    if df["label"].isna().any():
        raise ValueError(f"{name} contains NaN labels")

    # Ensure binary int labels
    unique = set(df["label"].dropna().unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} label values are not binary 0/1: {sorted(unique)}")


def load_splits(
    root: str | Path = "..",
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    test_path: str | Path | None = None,
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits from CSV.

    Args:
        root: repo root (use '..' when called from notebooks/)
        train_path/val_path/test_path: optional overrides
        validate: run basic schema/label checks

    Returns:
        (train_df, val_df, test_df)
    """
    root = Path(root).resolve()

    if train_path and val_path and test_path:
        paths = SplitPaths(Path(train_path), Path(val_path), Path(test_path))
    else:
        defaults = default_split_paths(root)
        paths = SplitPaths(
            train=Path(train_path) if train_path else defaults.train,
            val=Path(val_path) if val_path else defaults.val,
            test=Path(test_path) if test_path else defaults.test,
        )

    train_df = pd.read_csv(paths.train)
    val_df = pd.read_csv(paths.val)
    test_df = pd.read_csv(paths.test)

    if validate:
        _validate_df(train_df, "train")
        _validate_df(val_df, "val")
        _validate_df(test_df, "test")

    return train_df, val_df, test_df
