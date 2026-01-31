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


# ============================================================================
# Sentiment Features (vaderSentiment)
# ============================================================================

def sentiment_features(text: str) -> Dict[str, float]:
    """
    Extract sentiment features using VADER sentiment analyzer.

    Returns:
        Dictionary with sentiment scores:
        - resp_sent_compound: compound score [-1, 1]
        - resp_sent_positive: positive score [0, 1]
        - resp_sent_negative: negative score [0, 1]
        - resp_sent_neutral: neutral score [0, 1]
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError(
            "vaderSentiment not installed. "
            "Install with: pip install vaderSentiment"
        )

    # Initialize analyzer (note: creating analyzer each time is fine for small datasets)
    analyzer = SentimentIntensityAnalyzer()

    t = safe_text(text).strip()
    if not t:
        # Return neutral scores for empty text
        return {
            "resp_sent_compound": 0.0,
            "resp_sent_positive": 0.0,
            "resp_sent_negative": 0.0,
            "resp_sent_neutral": 1.0,
        }

    # Get sentiment scores
    scores = analyzer.polarity_scores(t)

    return {
        "resp_sent_compound": float(scores["compound"]),
        "resp_sent_positive": float(scores["pos"]),
        "resp_sent_negative": float(scores["neg"]),
        "resp_sent_neutral": float(scores["neu"]),
    }


def add_sentiment_features(
    df: pd.DataFrame,
    response_col: str = "response"
) -> pd.DataFrame:
    """
    Add sentiment feature columns to a copy of df using VADER.

    Args:
        df: Input dataframe
        response_col: Name of column containing text to analyze

    Returns:
        Dataframe with 4 additional sentiment columns:
        - resp_sent_compound: overall sentiment [-1, 1]
        - resp_sent_positive: positive sentiment [0, 1]
        - resp_sent_negative: negative sentiment [0, 1]
        - resp_sent_neutral: neutral sentiment [0, 1]
    """
    out = df.copy()
    sent_feats = out[response_col].apply(sentiment_features)
    sent_df = pd.DataFrame(list(sent_feats))
    out = pd.concat([out.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
    return out


def get_sentiment_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return sentiment feature columns."""
    return [c for c in df.columns if c.startswith("resp_sent_")]
