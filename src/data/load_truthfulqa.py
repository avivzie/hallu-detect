"""
TruthfulQA Dataset Loader

Purpose:
--------
Load and preprocess the TruthfulQA dataset for cross-dataset validation
of hallucination detection models trained on HaluEval.

TruthfulQA Dataset:
-------------------
- Source: https://huggingface.co/datasets/truthful_qa
- Task: Multiple-choice QA designed to test whether models generate truthful answers
- Contains questions where humans might give false answers due to misconceptions
- ~800 questions with labeled correct/incorrect answers

Output Schema (matching HaluEval format):
-----------------------------------------
id       : unique row identifier (string)
group_id : question identifier (same for correct/incorrect pairs)
task     : "truthfulqa" (constant)
prompt   : the question (string)
response : the answer (correct or incorrect) (string)
label    : 1 = incorrect/hallucination, 0 = correct/truthful (int)
context  : empty string (TruthfulQA doesn't provide context)

Label Mapping:
--------------
- best_answer (correct answer) -> label=0 (non-hallucination)
- incorrect_answers -> label=1 (hallucination)
- We expand each question into multiple rows (1 correct + N incorrect)

Cross-Dataset Validation Strategy:
-----------------------------------
1. Load TruthfulQA (no train/val split - evaluation only)
2. Use HaluEval-trained model for inference
3. Evaluate on TruthfulQA to test generalization
4. Compare: HaluEval test set vs. TruthfulQA performance
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def _row(
    row_id: str,
    task: str,
    prompt: str,
    response: str,
    label: int,
    context: str = "",
    group_id: str = "",
) -> dict:
    """Create a single row in the unified schema (matches HaluEval format)."""
    return {
        "id": row_id,
        "group_id": group_id,
        "task": task,
        "prompt": prompt or "",
        "response": response or "",
        "label": int(label),
        "context": context or "",
    }


def load_truthfulqa_generation(max_incorrect_per_question: int = 3) -> pd.DataFrame:
    """
    Load TruthfulQA 'generation' configuration.

    This configuration contains:
    - question: The question text
    - best_answer: The correct answer
    - correct_answers: List of acceptable correct answers
    - incorrect_answers: List of common incorrect answers

    Args:
        max_incorrect_per_question: Max number of incorrect answers to include per question
                                   (TruthfulQA has many incorrect answers; we sample a few)

    Returns:
        DataFrame with unified schema (id, group_id, task, prompt, response, label, context)
    """
    print("Loading TruthfulQA (generation) from Hugging Face...")

    # Load dataset
    # Note: TruthfulQA has 'validation' split (~800 questions), no train/test split
    try:
        ds = load_dataset("truthful_qa", "generation", split="validation")
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")
        print("Attempting alternative loading method...")
        # Fallback: try without split specification
        ds = load_dataset("truthful_qa", "generation")["validation"]

    print(f"Loaded {len(ds)} questions from TruthfulQA")

    # Print example keys for debugging
    if len(ds) > 0:
        print("\n[Debug] Example question keys:", list(ds[0].keys()))

    rows: List[Dict] = []

    for i, ex in enumerate(tqdm(ds, desc="Processing TruthfulQA")):
        question = ex.get("question", "")

        # Best answer (ground truth) - label=0
        best_answer = ex.get("best_answer", "")
        if best_answer:
            group_id = f"truthfulqa_{i}"
            rows.append(_row(
                row_id=f"truthfulqa_{i}_correct",
                task="truthfulqa",
                prompt=question,
                response=best_answer,
                label=0,  # Correct answer
                context="",
                group_id=group_id
            ))

        # Incorrect answers (hallucinations) - label=1
        incorrect_answers = ex.get("incorrect_answers", [])

        # Sample a subset of incorrect answers to avoid imbalance
        # (some questions have 10+ incorrect answers)
        num_incorrect = min(len(incorrect_answers), max_incorrect_per_question)

        for j, incorrect_ans in enumerate(incorrect_answers[:num_incorrect]):
            rows.append(_row(
                row_id=f"truthfulqa_{i}_incorrect_{j}",
                task="truthfulqa",
                prompt=question,
                response=incorrect_ans,
                label=1,  # Incorrect/hallucination
                context="",
                group_id=group_id
            ))

    df = pd.DataFrame(rows)

    print(f"\n[TruthfulQA Stats]")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique questions: {df['group_id'].nunique()}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"  Label mean (hallucination rate): {df['label'].mean():.3f}")

    return df


def load_truthfulqa_multiple_choice() -> pd.DataFrame:
    """
    Load TruthfulQA 'multiple_choice' configuration.

    This configuration contains:
    - question: The question text
    - mc1_targets: Single correct answer (multiple choice with 1 correct)
    - mc2_targets: Multiple correct answers (multiple choice with multiple correct)

    Returns:
        DataFrame with unified schema

    Note: This is more complex to parse than 'generation' format.
          Recommended to use load_truthfulqa_generation() instead for simplicity.
    """
    print("Loading TruthfulQA (multiple_choice) from Hugging Face...")
    print("Note: Using 'generation' format is recommended for simpler processing.")

    try:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    except Exception:
        ds = load_dataset("truthful_qa", "multiple_choice")["validation"]

    print(f"Loaded {len(ds)} questions")

    # TODO: Implement if needed
    # For now, recommend using generation format
    raise NotImplementedError(
        "Multiple choice format parsing not implemented. "
        "Use load_truthfulqa_generation() instead."
    )


def preprocess_truthfulqa(
    df: pd.DataFrame,
    min_response_length: int = 5
) -> pd.DataFrame:
    """
    Clean and validate TruthfulQA data.

    Args:
        df: Raw TruthfulQA DataFrame
        min_response_length: Minimum character length for responses

    Returns:
        Cleaned DataFrame
    """
    print("\n[Preprocessing TruthfulQA]")

    initial_len = len(df)

    # Convert to strings
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df["context"] = df["context"].fillna("").astype(str)

    # Remove empty or too-short responses
    df = df[
        (df["prompt"].str.len() > 0) &
        (df["response"].str.len() >= min_response_length)
    ].reset_index(drop=True)

    removed = initial_len - len(df)
    print(f"  Removed {removed} rows with empty/short responses ({initial_len} -> {len(df)})")

    # Verify labels are binary
    unique_labels = df["label"].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Non-binary labels found: {unique_labels}")

    print(f"  Final shape: {df.shape}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    return df


def save_truthfulqa(
    df: pd.DataFrame,
    output_path: Path | str,
    sample_size: int = 50
) -> None:
    """
    Save TruthfulQA dataset to CSV.

    Args:
        df: Processed TruthfulQA DataFrame
        output_path: Output CSV path (e.g., data_processed/truthfulqa.csv)
        sample_size: Number of rows to save as preview sample
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    df.to_csv(output_path, index=False)
    print(f"\n[Saved] TruthfulQA dataset: {output_path}")

    # Save sample for quick preview
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    sample_path = output_path.parent / "truthfulqa_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"[Saved] TruthfulQA sample: {sample_path}")


def main() -> None:
    """CLI entry point for building TruthfulQA dataset."""
    parser = argparse.ArgumentParser(
        description="Load and preprocess TruthfulQA dataset for cross-validation"
    )
    parser.add_argument(
        "--output",
        default="data_processed/truthfulqa.csv",
        help="Output path for processed dataset"
    )
    parser.add_argument(
        "--max-incorrect",
        type=int,
        default=3,
        help="Max incorrect answers per question (default: 3)"
    )
    parser.add_argument(
        "--min-response-length",
        type=int,
        default=5,
        help="Minimum response length in characters (default: 5)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows for preview sample (default: 50)"
    )

    args = parser.parse_args()

    # Load TruthfulQA
    df = load_truthfulqa_generation(max_incorrect_per_question=args.max_incorrect)

    # Preprocess
    df = preprocess_truthfulqa(df, min_response_length=args.min_response_length)

    # Save
    save_truthfulqa(df, args.output, sample_size=args.sample_size)

    print("\n✅ TruthfulQA dataset ready for cross-validation!")
    print(f"   Load with: pd.read_csv('{args.output}')")


if __name__ == "__main__":
    main()
