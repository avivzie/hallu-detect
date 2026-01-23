"""
HaluEval Dataset Builder (Sprint 1)

Goal
----
Convert the HaluEval dataset (multiple subsets) into ONE unified supervised dataset
for hallucination detection / prediction.

Why we do this
--------------
HaluEval is provided as multiple subsets (qa/dialogue/summarization/general) and with
slightly different schemas. For modeling, we want a single clean table with a stable schema,
so that downstream feature engineering + model training is simple and reproducible.

Output (files)
--------------
data_processed/clean.csv      - the full unified dataset
data_processed/train.csv      - stratified split
data_processed/val.csv        - stratified split
data_processed/test.csv       - stratified split
data_processed/sample.csv     - small sample for quick preview / sharing / notebook visuals

Unified schema (columns)
------------------------
id       : unique row identifier (string)
task     : which HaluEval subset this came from (qa/dialogue/summarization/general)
prompt   : the user input / question / dialogue history / document (string)
response : the model output (string)
label    : 1 = hallucination, 0 = non-hallucination (int)
context  : optional supporting knowledge if available (string)

Label mapping logic
-------------------
- For qa/dialogue/summarization subsets:
  Each raw record contains TWO candidate outputs:
    - "right_*"         -> label=0 (non-hallucination)
    - "hallucinated_*"  -> label=1 (hallucination)
  We expand each raw record into two rows.

- For general subset:
  HaluEval includes a yes/no hallucination label for a single response.
  We map yes -> 1, no -> 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# -----------------------------
# Helper functions
# -----------------------------

def _row(row_id: str, task: str, prompt: str, response: str, label: int, context: str = "") -> dict:
    """
    Create a single row in the unified schema.

    WHY: Keeping row creation centralized prevents schema drift across tasks.
    """
    return {
        "id": row_id,
        "task": task,
        "prompt": prompt or "",
        "response": response or "",
        "label": int(label),
        "context": context or "",
    }


def _build_pair_rows(
    ds,
    *,
    task: str,
    prompt_col: str,
    right_col: str,
    hall_col: str,
    context_col: str | None,
) -> list[dict]:
    """
    Convert a dataset where each example contains BOTH:
      - a ground-truth (right_*) output
      - a hallucinated_* output

    Returns two rows per raw example.
    """
    rows: list[dict] = []
    for i, ex in enumerate(tqdm(ds, desc=f"Loading {task}")):
        prompt = ex.get(prompt_col, "")
        context = ex.get(context_col, "") if context_col else ""

        right = ex.get(right_col, "")
        hall = ex.get(hall_col, "")

        # WHY: We want a supervised dataset with explicit labels for each output candidate.
        rows.append(_row(f"{task}_{i}_gt", task, prompt, right, 0, context))
        rows.append(_row(f"{task}_{i}_hall", task, prompt, hall, 1, context))

    return rows


def _print_preview(df: pd.DataFrame, title: str, n: int = 3) -> None:
    """
    Print a tiny preview of the dataset for sanity.

    WHY: Supervisors usually want to see "what the data looks like" before/after processing.
    """
    print(f"\n--- {title} ---")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(n).to_string(index=False))


# -----------------------------
# Main builder
# -----------------------------

def build_halueval(out_csv: Path, seed: int = 42, sample_size: int = 200) -> None:
    """
    Load HaluEval subsets, unify schema, run basic cleaning,
    create stratified splits, and save outputs to disk.
    """
    rows: list[dict] = []

    # 1) Load and convert pair-style subsets
    # NOTE: column names match the HF dataset viewer you showed (dialogue_history/right_response/... etc).
    rows += _build_pair_rows(
        load_dataset("pminervini/HaluEval", "dialogue", split="data"),
        task="dialogue",
        prompt_col="dialogue_history",
        right_col="right_response",
        hall_col="hallucinated_response",
        context_col="knowledge",
    )

    rows += _build_pair_rows(
        load_dataset("pminervini/HaluEval", "qa", split="data"),
        task="qa",
        prompt_col="question",
        right_col="right_answer",
        hall_col="hallucinated_answer",
        context_col="knowledge",
    )

    rows += _build_pair_rows(
        load_dataset("pminervini/HaluEval", "summarization", split="data"),
        task="summarization",
        prompt_col="document",
        right_col="right_summary",
        hall_col="hallucinated_summary",
        context_col=None,
    )

    # 2) Load and convert the general subset (single output + label)
    gen = load_dataset("pminervini/HaluEval", "general", split="data")

    # Print keys for debugging once (helpful if HF schema changes)
    # WHY: Hugging Face conversions sometimes have minor field name differences.
    if len(gen) > 0:
        print("\n[Debug] general subset example keys:", list(gen[0].keys()))

    for i, ex in enumerate(tqdm(gen, desc="Loading general")):
        # Flexible fallbacks: different exports can name fields differently.
        prompt = ex.get("user_query") or ex.get("question") or ex.get("prompt") or ""
        response = ex.get("chatgpt_response") or ex.get("response") or ex.get("answer") or ""

        # Expected: hallucination_label often appears as "Yes"/"No"
        raw = (ex.get("hallucination_label") or ex.get("label") or "").strip().lower()
        label = 1 if raw in {"yes", "y", "hallucination", "hallucinated", "true", "1"} else 0

        rows.append(_row(f"general_{i}", "general", prompt, response, label, ""))

    # 3) Build DataFrame (pre-clean)
    df = pd.DataFrame(rows)
    _print_preview(df, "Before cleaning (raw unified dataframe)", n=3)

    # 4) Cleaning / normalization
    # WHY: We want stable string columns and to remove empty examples (bad for vectorizers).
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df["context"] = df["context"].fillna("").astype(str)

    # Remove rows where prompt/response are empty after cleanup
    before = len(df)
    df = df[(df["prompt"].str.len() > 0) & (df["response"].str.len() > 0)].reset_index(drop=True)
    after = len(df)
    print(f"\n[Cleaning] Removed empty rows: {before - after} (from {before} -> {after})")

    _print_preview(df, "After cleaning", n=3)

    # 5) Basic descriptive stats (useful for writeup + sanity)
    print("\n[Stats] Label mean (approx. hallucination rate):", round(float(df["label"].mean()), 4))
    print("[Stats] Label counts:", df["label"].value_counts().to_dict())
    print("[Stats] Task counts:", df["task"].value_counts().to_dict())

    # 6) Create splits (stratified by label)
    # WHY: Stratification ensures label ratio stays similar across train/val/test.
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"]
    )

    print("\n[Splits]")
    print("Train/Val/Test shapes:", train_df.shape, val_df.shape, test_df.shape)

    # 7) Save outputs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    train_df.to_csv(out_csv.parent / "train.csv", index=False)
    val_df.to_csv(out_csv.parent / "val.csv", index=False)
    test_df.to_csv(out_csv.parent / "test.csv", index=False)

    # Save a small sample for quick previews / notebook visuals
    # WHY: Lets you show data examples without loading the full file every time.
    sample_df = df.sample(min(sample_size, len(df)), random_state=seed)
    sample_df.to_csv(out_csv.parent / "sample.csv", index=False)

    print("\n[Saved]")
    print("clean.csv:", out_csv)
    print("train/val/test:", out_csv.parent / "train.csv", out_csv.parent / "val.csv", out_csv.parent / "test.csv")
    print("sample.csv:", out_csv.parent / "sample.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_processed/clean.csv", help="Output path for clean.csv")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits & sampling")
    ap.add_argument("--sample_size", type=int, default=200, help="How many rows to save into sample.csv")
    args = ap.parse_args()

    build_halueval(Path(args.out), seed=args.seed, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
