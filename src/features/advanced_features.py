# src/features/advanced_features.py
"""
Advanced features for hallucination detection:
- Sentiment analysis
- Confidence/uncertainty metrics
- Embedding-based uncertainty (pseudo-semantic entropy)
"""

import numpy as np
import pandas as pd
from typing import List, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Run: pip install vader sentiment")


def add_sentiment_features(df: pd.DataFrame, text_col: str = "response") -> pd.DataFrame:
    """
    Add sentiment analysis features using VADER.

    Hypothesis: Hallucinated responses may exhibit different sentiment patterns
    (e.g., more neutral/cautious, or overly positive/confident).

    Args:
        df: DataFrame with text column
        text_col: Column containing text to analyze

    Returns:
        DataFrame with added sentiment features
    """
    df = df.copy()

    if not VADER_AVAILABLE:
        # Fallback: simple proxy features
        texts = df[text_col].fillna("").astype(str)
        df[f"{text_col}_sent_proxy_pos"] = texts.str.lower().str.count(r'\b(good|great|excellent|perfect|correct|yes)\b')
        df[f"{text_col}_sent_proxy_neg"] = texts.str.lower().str.count(r'\b(bad|wrong|incorrect|no|not|never)\b')
        return df

    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].fillna("").astype(str)

    sentiments = texts.apply(lambda x: analyzer.polarity_scores(x))

    df[f"{text_col}_sent_neg"] = sentiments.apply(lambda x: x['neg'])
    df[f"{text_col}_sent_neu"] = sentiments.apply(lambda x: x['neu'])
    df[f"{text_col}_sent_pos"] = sentiments.apply(lambda x: x['pos'])
    df[f"{text_col}_sent_compound"] = sentiments.apply(lambda x: x['compound'])

    return df


def add_confidence_proxy_features(df: pd.DataFrame, text_col: str = "response") -> pd.DataFrame:
    """
    Add confidence/uncertainty proxy features.

    These approximate model confidence using linguistic markers
    (since we don't have access to model's internal confidence scores).

    Args:
        df: DataFrame with text column
        text_col: Column containing text to analyze

    Returns:
        DataFrame with added confidence proxy features
    """
    df = df.copy()
    texts = df[text_col].fillna("").astype(str)

    # Hedging language (uncertainty indicators)
    hedging_patterns = [
        r'\b(might|may|could|possibly|perhaps|maybe|probably|likely|unlikely|seems?|appears?)\b',
        r'\b(not sure|uncertain|unclear|unknown|unsure)\b',
        r'\b(I think|I believe|I guess|in my opinion)\b',
    ]

    df[f"{text_col}_conf_hedging"] = sum(
        texts.str.lower().str.count(pattern) for pattern in hedging_patterns
    )

    # Definitive language (confidence indicators)
    definitive_patterns = [
        r'\b(certainly|definitely|absolutely|clearly|obviously|undoubtedly)\b',
        r'\b(always|never|must|will|cannot|impossible)\b',
        r'\b(fact|proof|proven|evidence|demonstrate)\b',
    ]

    df[f"{text_col}_conf_definitive"] = sum(
        texts.str.lower().str.count(pattern) for pattern in definitive_patterns
    )

    # Confidence ratio (definitive / (hedging + definitive + 1))
    df[f"{text_col}_conf_ratio"] = df[f"{text_col}_conf_definitive"] / (
        df[f"{text_col}_conf_hedging"] + df[f"{text_col}_conf_definitive"] + 1
    )

    # Caveat language (disclaimers)
    caveat_patterns = [
        r'\b(however|although|though|but|yet|nevertheless)\b',
        r'\b(typically|generally|usually|often|sometimes)\b',
        r'\b(depending|varies|may vary)\b',
    ]

    df[f"{text_col}_conf_caveats"] = sum(
        texts.str.lower().str.count(pattern) for pattern in caveat_patterns
    )

    return df


def add_embedding_uncertainty_features(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prefix: str = "resp"
) -> pd.DataFrame:
    """
    Add embedding-based uncertainty features (pseudo-semantic entropy).

    This approximates semantic entropy by measuring embedding variance,
    without requiring multiple model samples.

    Args:
        df: DataFrame to add features to
        embeddings: Embedding matrix (n_samples, embedding_dim)
        prefix: Prefix for feature names

    Returns:
        DataFrame with added uncertainty features
    """
    df = df.copy()

    # L2 norm (embedding magnitude - confidence proxy)
    df[f"{prefix}_emb_norm"] = np.linalg.norm(embeddings, axis=1)

    # Distance to embedding centroid (outlier detection)
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    df[f"{prefix}_emb_dist_to_centroid"] = distances

    # Embedding entropy (approximation using component variance)
    # High variance → high uncertainty
    embedding_var = embeddings.var(axis=1)
    df[f"{prefix}_emb_variance"] = embedding_var

    return df


def get_advanced_feature_cols(df: pd.DataFrame, prefixes: List[str] = None) -> List[str]:
    """
    Get list of advanced feature column names.

    Args:
        df: DataFrame
        prefixes: Optional list of prefixes to filter (e.g., ['resp_sent', 'resp_conf'])

    Returns:
        List of feature column names
    """
    if prefixes is None:
        prefixes = ['_sent_', '_conf_', '_emb_']

    return [col for col in df.columns if any(prefix in col for prefix in prefixes)]


# Feature hypotheses for documentation
ADVANCED_FEATURE_HYPOTHESES = {
    "sentiment": {
        "sent_compound": "Hallucinations may be more neutral (avoid commitment) or overly positive (overconfidence)",
        "sent_neg": "Negative sentiment may indicate hedging or uncertainty → lower hallucination risk",
        "sent_pos": "Overly positive sentiment may indicate overconfidence → higher hallucination risk",
    },
    "confidence": {
        "conf_hedging": "High hedging language → model is uncertain → paradoxically may be MORE accurate (knows limits)",
        "conf_definitive": "High definitive language → model is confident → may be overconfident hallucination",
        "conf_ratio": "High ratio (definitive/hedging) → overconfidence → higher hallucination risk",
        "conf_caveats": "Caveats/qualifiers → model acknowledging limitations → lower hallucination risk",
    },
    "embedding": {
        "emb_norm": "Low embedding norm → weak signal → higher uncertainty → higher hallucination risk",
        "emb_dist_to_centroid": "Far from centroid → outlier → unusual response → higher hallucination risk",
        "emb_variance": "High embedding variance → high internal uncertainty → higher hallucination risk",
    }
}
