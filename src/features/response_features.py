# src/features/response_features.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseFeatureConfig:
    """Configuration for response-based engineered features."""
    uncertainty_words: List[str]


DEFAULT_CONFIG = ResponseFeatureConfig(
    uncertainty_words=[
        "maybe", "perhaps", "probably", "possibly", "likely", "unlikely",
        "i think", "i believe", "i guess", "it seems", "it appears",
        "not sure", "unclear", "cannot confirm", "can't confirm",
        "might", "may", "could",
    ]
)

# Pre-compiled regex patterns (module-level for speed)
RE_NUM = re.compile(r"\d+")
RE_MULTI_EXCL = re.compile(r"!{2,}")
RE_MULTI_Q = re.compile(r"\?{2,}")
RE_ELLIPSIS = re.compile(r"\.{3,}")
RE_PUNCT = re.compile(r"[^\w\s]")  # rough punctuation detector


def safe_text(x) -> str:
    """Return a safe string (handles NaN/None)."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def count_uncertainty_phrases(text: str, cfg: ResponseFeatureConfig = DEFAULT_CONFIG) -> int:
    """Count occurrences of uncertainty phrases using simple substring matching."""
    t = text.lower()
    return sum(t.count(p) for p in cfg.uncertainty_words)


def basic_numeric_features(text: str, cfg: ResponseFeatureConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """
    Extract simple numeric features from a response text.
    Returns a dict: feature_name -> value.
    """
    t = safe_text(text).strip()

    n_chars = len(t)
    n_words = 0 if not t else len(t.split())

    # Punctuation signals
    n_punct = len(RE_PUNCT.findall(t))
    has_multi_excl = 1.0 if RE_MULTI_EXCL.search(t) else 0.0
    has_multi_q = 1.0 if RE_MULTI_Q.search(t) else 0.0
    has_ellipsis = 1.0 if RE_ELLIPSIS.search(t) else 0.0

    # Numbers
    n_numbers = float(len(RE_NUM.findall(t)))

    # Uncertainty / hedging
    n_uncertainty = float(count_uncertainty_phrases(t, cfg=cfg))

    # Ratios (avoid divide-by-zero)
    punct_per_word = float(n_punct) / max(n_words, 1)
    numbers_per_word = float(n_numbers) / max(n_words, 1)

    return {
        "resp_n_chars": float(n_chars),
        "resp_n_words": float(n_words),
        "resp_n_punct": float(n_punct),
        "resp_has_multi_excl": has_multi_excl,
        "resp_has_multi_q": has_multi_q,
        "resp_has_ellipsis": has_ellipsis,
        "resp_n_numbers": n_numbers,
        "resp_n_uncertainty": n_uncertainty,
        "resp_punct_per_word": punct_per_word,
        "resp_numbers_per_word": numbers_per_word,
    }


def add_numeric_feature_columns(
    df: pd.DataFrame,
    response_col: str = "response",
    cfg: ResponseFeatureConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """
    Add numeric feature columns to a copy of df (does not mutate original).
    """
    out = df.copy()
    feats = out[response_col].apply(lambda x: basic_numeric_features(x, cfg=cfg))
    feats_df = pd.DataFrame(list(feats))
    out = pd.concat([out.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    return out


def get_numeric_feature_cols(df: pd.DataFrame, prefix: str = "resp_") -> List[str]:
    """Return engineered numeric feature columns by prefix."""
    return [c for c in df.columns if c.startswith(prefix)]
