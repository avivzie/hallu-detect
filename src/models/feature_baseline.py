# src/models/feature_baseline.py
from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_tfidf_vectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    lowercase=True,
) -> TfidfVectorizer:
    """Factory for a standard TF-IDF vectorizer used across notebooks."""
    return TfidfVectorizer(
        lowercase=lowercase,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )


def build_tfidf_only_logreg(
    text_col: str = "response",
    class_weight: str | None = "balanced",
    max_iter: int = 2000,
    **tfidf_kwargs,
) -> Pipeline:
    """
    TF-IDF(text_col) -> LogisticRegression
    """
    tfidf = make_tfidf_vectorizer(**tfidf_kwargs)

    preprocess = ColumnTransformer(
        transformers=[("tfidf", tfidf, text_col)],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])


def build_tfidf_numeric_logreg(
    numeric_cols: List[str],
    text_col: str = "response",
    class_weight: str | None = "balanced",
    max_iter: int = 2000,
    sparse_threshold: float = 0.3,
    **tfidf_kwargs,
) -> Pipeline:
    """
    TF-IDF(text_col) + scaled numeric features -> LogisticRegression
    """
    tfidf = make_tfidf_vectorizer(**tfidf_kwargs)

    preprocess = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, text_col),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
