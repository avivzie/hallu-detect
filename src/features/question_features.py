# src/features/question_features.py
"""
Question-only feature extraction for hallucination risk prediction.

These features are computed from the prompt/question alone (before seeing the response),
enabling pre-generation risk assessment.
"""

import re
import pandas as pd
import numpy as np
from typing import List


def add_question_features(df: pd.DataFrame, prompt_col: str = "prompt") -> pd.DataFrame:
    """
    Add question-level features for hallucination risk prediction.

    Features predict: "Is this question likely to elicit a hallucination?"

    Args:
        df: DataFrame with prompt column
        prompt_col: Name of column containing questions/prompts

    Returns:
        DataFrame with added question feature columns (q_*)
    """
    df = df.copy()

    # Fill missing prompts
    prompts = df[prompt_col].fillna("").astype(str)

    # Basic length features
    df["q_n_chars"] = prompts.str.len()
    df["q_n_words"] = prompts.str.split().str.len()
    df["q_n_sentences"] = prompts.apply(lambda x: len(re.split(r'[.!?]+', x)) - 1)

    # Complexity indicators
    df["q_avg_word_length"] = prompts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )

    # Question type indicators (high risk for factual questions)
    df["q_is_who"] = prompts.str.lower().str.contains(r'\bwho\b', regex=True).astype(int)
    df["q_is_what"] = prompts.str.lower().str.contains(r'\bwhat\b', regex=True).astype(int)
    df["q_is_when"] = prompts.str.lower().str.contains(r'\bwhen\b', regex=True).astype(int)
    df["q_is_where"] = prompts.str.lower().str.contains(r'\bwhere\b', regex=True).astype(int)
    df["q_is_how"] = prompts.str.lower().str.contains(r'\bhow\b', regex=True).astype(int)
    df["q_is_why"] = prompts.str.lower().str.contains(r'\bwhy\b', regex=True).astype(int)

    # Factual content indicators (higher hallucination risk)
    df["q_has_numbers"] = prompts.str.contains(r'\d', regex=True).astype(int)
    df["q_n_numbers"] = prompts.apply(lambda x: len(re.findall(r'\d+', x)))
    df["q_has_year"] = prompts.str.contains(r'\b(19|20)\d{2}\b', regex=True).astype(int)
    df["q_has_percentage"] = prompts.str.contains(r'\d+\s*%', regex=True).astype(int)

    # Named entity proxies (proper nouns - higher risk)
    df["q_n_capitalized"] = prompts.apply(
        lambda x: sum(1 for w in x.split() if w and w[0].isupper())
    )
    df["q_pct_capitalized"] = df["q_n_capitalized"] / df["q_n_words"].replace(0, 1)

    # Specificity indicators
    df["q_has_quote"] = prompts.str.contains(r'["\']', regex=True).astype(int)
    df["q_n_commas"] = prompts.str.count(',')
    df["q_has_parentheses"] = prompts.str.contains(r'[()]', regex=True).astype(int)

    # Ambiguity indicators (lower risk - vague questions)
    df["q_has_maybe"] = prompts.str.lower().str.contains(r'\b(maybe|perhaps|possibly)\b', regex=True).astype(int)
    df["q_has_approximately"] = prompts.str.lower().str.contains(
        r'\b(about|around|approximately|roughly)\b', regex=True
    ).astype(int)

    # Domain-specific indicators
    # Medical terms (high risk - factual domain)
    medical_terms = r'\b(disease|patient|treatment|diagnosis|symptom|medical|doctor|hospital|drug|medicine)\b'
    df["q_has_medical"] = prompts.str.lower().str.contains(medical_terms, regex=True).astype(int)

    # Technical/mathematical terms (high risk - precise domain)
    tech_terms = r'\b(algorithm|equation|theorem|proof|calculate|formula|function|matrix)\b'
    df["q_has_technical"] = prompts.str.lower().str.contains(tech_terms, regex=True).astype(int)

    # Historical/factual terms (high risk)
    historical_terms = r'\b(born|died|invented|discovered|founded|happened|occurred|century)\b'
    df["q_has_historical"] = prompts.str.lower().str.contains(historical_terms, regex=True).astype(int)

    # Opinion/subjective questions (lower risk - less verifiable)
    opinion_terms = r'\b(think|feel|believe|opinion|prefer|like|favorite|best|worst)\b'
    df["q_has_opinion"] = prompts.str.lower().str.contains(opinion_terms, regex=True).astype(int)

    # Comparison questions (moderate risk)
    df["q_has_comparison"] = prompts.str.lower().str.contains(
        r'\b(better|worse|more|less|than|versus|vs|compared)\b', regex=True
    ).astype(int)

    # Multiple questions in one (higher complexity → higher risk)
    df["q_has_multiple_questions"] = (prompts.str.count(r'\?') > 1).astype(int)

    return df


def get_question_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return list of question feature column names."""
    return [col for col in df.columns if col.startswith("q_")]


# Hypothesis mapping for documentation
FEATURE_HYPOTHESES = {
    # High risk features (predict hallucination)
    "q_has_numbers": "Questions with numbers require precise recall → higher hallucination risk",
    "q_has_year": "Year-specific questions require exact memory → higher risk",
    "q_is_who": "Who questions demand specific names → higher risk",
    "q_is_when": "When questions require temporal precision → higher risk",
    "q_has_medical": "Medical questions require factual accuracy → higher risk",
    "q_has_technical": "Technical questions require domain knowledge → higher risk",
    "q_has_historical": "Historical questions require factual recall → higher risk",
    "q_n_capitalized": "More proper nouns → more specific entities → higher risk",
    "q_has_quote": "Questions with quotes imply verification needed → higher risk",

    # Low risk features (predict non-hallucination)
    "q_has_opinion": "Opinion questions are subjective → lower risk (less verifiable)",
    "q_has_maybe": "Hedge words indicate flexible answer expected → lower risk",
    "q_is_why": "Why questions allow explanatory answers → moderate-low risk",
    "q_is_how": "How questions allow procedural answers → moderate-low risk",
}
