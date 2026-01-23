import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _row(row_id: str, task: str, prompt: str, response: str, label: int, context: str = "") -> dict:
    return {
        "id": row_id,
        "task": task,
        "prompt": prompt or "",
        "response": response or "",
        "label": int(label),
        "context": context or "",
    }


def _build_pair_rows(ds, task: str, prompt_col: str, right_col: str, hall_col: str, context_col: str | None):
    rows = []
    for i, ex in enumerate(tqdm(ds, desc=f"Loading {task}")):
        prompt = ex.get(prompt_col, "")
        context = ex.get(context_col, "") if context_col else ""

        right = ex.get(right_col, "")
        hall = ex.get(hall_col, "")

        rows.append(_row(f"{task}_{i}_gt", task, prompt, right, 0, context))
        rows.append(_row(f"{task}_{i}_hall", task, prompt, hall, 1, context))
    return rows


def build_halueval(out_csv: Path, seed: int = 42):
    rows = []

    # dialogue: knowledge, dialogue_history, right_response, hallucinated_response
    rows += _build_pair_rows(
        load_dataset("pminervini/HaluEval", "dialogue", split="data"),
        task="dialogue",
        prompt_col="dialogue_history",
        right_col="right_response",
        hall_col="hallucinated_response",
        context_col="knowledge",
    )

    # qa: knowledge, question, right_answer, hallucinated_answer
    rows += _build_pair_rows(
        load_dataset("pminervini/HaluEval", "qa", split="data"),
        task="qa",
        prompt_col="question",
        right_col="right_answer",
        hall_col="hallucinated_answer",
        context_col="knowledge",
    )

    # summarization: document, right_summary, hallucinated_summary
    rows += _build_pair_rows(
        load_dataset("pminervini/HaluEval", "summarization", split="data"),
        task="summarization",
        prompt_col="document",
        right_col="right_summary",
        hall_col="hallucinated_summary",
        context_col=None,
    )

    # general: prompt/response + hallucination_label (Yes/No)
    gen = load_dataset("pminervini/HaluEval", "general", split="data")
    for i, ex in enumerate(tqdm(gen, desc="Loading general")):
        prompt = ex.get("user_query") or ex.get("question") or ex.get("prompt") or ""
        response = ex.get("chatgpt_response") or ex.get("response") or ex.get("answer") or ""
        raw = (ex.get("hallucination_label") or ex.get("label") or "").strip().lower()
        label = 1 if raw in {"yes", "y", "hallucination", "hallucinated", "true", "1"} else 0
        rows.append(_row(f"general_{i}", "general", prompt, response, label, ""))

    df = pd.DataFrame(rows)

    # cleanup
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df["context"] = df["context"].fillna("").astype(str)
    df = df[(df["prompt"].str.len() > 0) & (df["response"].str.len() > 0)].reset_index(drop=True)

    # splits (stratified)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    train_df.to_csv(out_csv.parent / "train.csv", index=False)
    val_df.to_csv(out_csv.parent / "val.csv", index=False)
    test_df.to_csv(out_csv.parent / "test.csv", index=False)

    print("Saved:", out_csv)
    print("All:", df.shape, "label mean:", round(float(df["label"].mean()), 4))
    print("Train/Val/Test:", train_df.shape, val_df.shape, test_df.shape)
    print("Label counts:", df["label"].value_counts().to_dict())
    print("Tasks:", df["task"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_processed/clean.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_halueval(Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()