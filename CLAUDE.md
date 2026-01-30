# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MSc research project for hallucination detection/prediction in LLM outputs using the HaluEval dataset. The project implements and compares multiple baseline approaches for binary classification (hallucination vs. non-hallucination).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Dataset Architecture

The HaluEval dataset is processed into a unified format using `src/data/build_dataset.py`:

**Unified schema:**
- `id`: unique row identifier
- `group_id`: identifier used to keep paired examples together during splitting
- `task`: source subset (qa/dialogue/summarization/general)
- `prompt`: user input/question/dialogue history/document
- `response`: model output to classify
- `label`: 1 = hallucination, 0 = non-hallucination
- `context`: optional supporting knowledge

**Important split strategy:**
The dataset uses **group-aware splitting** via `GroupShuffleSplit` because qa/dialogue/summarization subsets expand each example into two rows (ground-truth + hallucinated) that share the same prompt. The `group_id` field ensures paired rows stay together in the same split to prevent data leakage.

**Loading splits:**
```python
from src.data.load_splits import load_splits
train_df, val_df, test_df = load_splits(root="..")  # from notebooks/
```

The `load_splits()` function validates that all splits contain required columns (`prompt`, `response`, `label`, `task`) and ensures labels are binary (0/1).

## Code Organization

### `src/data/`
- `build_dataset.py`: Converts HaluEval subsets into unified CSV format with group-aware splits
- `load_splits.py`: Loads and validates train/val/test splits

### `src/features/`
- `response_features.py`: Extracts engineered features from response text (uncertainty phrases, punctuation patterns, text length statistics)
- `embeddings.py`: Sentence-Transformer embedding extraction with caching (caches to `artifacts/embeddings/` by default)

### `src/models/`
- `feature_baseline.py`: TF-IDF + LogisticRegression pipelines (text-only and text+numeric features)
- `embedding_baseline.py`: LogisticRegression on sentence embeddings

### `src/utils/`
- `eval.py`: Binary classification evaluation utilities (`evaluate_split()` returns accuracy, F1, precision, recall)
- `experiment.py`: Experiment reproducibility utilities (random seeds, output directories, metrics saving, predictions export)

## Model Patterns

**TF-IDF baseline:**
```python
from src.models.feature_baseline import build_tfidf_only_logreg
model = build_tfidf_only_logreg(text_col="response")
model.fit(train_df, y_train)
```

**TF-IDF + engineered features:**
```python
from src.features.response_features import add_numeric_feature_columns, get_numeric_feature_cols
from src.models.feature_baseline import build_tfidf_numeric_logreg

train_df = add_numeric_feature_columns(train_df, response_col="response")
numeric_cols = get_numeric_feature_cols(train_df)
model = build_tfidf_numeric_logreg(numeric_cols, text_col="response")
model.fit(train_df, y_train)
```

**Embedding baseline:**
```python
from src.features.embeddings import EmbeddingConfig, embed_dataframe
from src.models.embedding_baseline import build_embedding_lr

cfg = EmbeddingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    text_col="response",
    normalize=True
)
X_train = embed_dataframe(train_df, cfg, cache_tag="train", use_cache=True)
model = build_embedding_lr()
model.fit(X_train, y_train)
```

## Notebook Workflow

Notebooks are in `notebooks/` and use the following pattern:
1. Add project root to path: `ROOT = Path("..").resolve(); sys.path.insert(0, str(ROOT))`
2. Load splits: `train_df, val_df, test_df = load_splits(root="..")`
3. Import from `src/` modules
4. Train models and evaluate on val/test
5. Save results to `reports/` as CSV or markdown tables

**Current notebooks:**
- `01_dataset_preview.ipynb`: EDA and dataset statistics
- `02_baseline_tfidf.ipynb`: TF-IDF + LogisticRegression baseline
- `03_feature_based_models.ipynb`: TF-IDF + engineered response features
- `04_embedding_based_models.ipynb`: Sentence-Transformer embeddings
- `05_tfidf_optimization.ipynb`: Hyperparameter tuning with group-aware CV
- `06_finetuned_transformer.ipynb`: Fine-tuned transformer models
- `07_generalization_and_input_ablation.ipynb`: Cross-task generalization and ablation studies

## Key Dependencies

- `scikit-learn`: ML pipelines and evaluation
- `sentence-transformers`: Embedding models
- `transformers`: Hugging Face models
- `datasets`: Loading HaluEval from Hugging Face
- `pandas`, `numpy`: Data manipulation
- `jupyterlab`: Notebook development
- `tqdm`: Progress bars

## Running Notebooks

Start JupyterLab:
```bash
jupyter lab
```

Notebooks expect to be run from the `notebooks/` directory and reference the project root as `..`.

## Caching

- Embeddings are cached to `artifacts/embeddings/<model_name>/<hash>.npz` based on SHA256 hash of input texts + model config
- Cache keys include model name, normalization settings, batch size, and optional `cache_tag` parameter
- To bypass cache: pass `use_cache=False` to `embed_texts()` or `embed_dataframe()`

## Model Evaluation Pattern

```python
from src.utils.eval import evaluate_split, metrics_table

train_metrics = evaluate_split("Train", model, X_train, y_train)
val_metrics = evaluate_split("Val", model, X_val, y_val)
test_metrics = evaluate_split("Test", model, X_test, y_test)

results = metrics_table([
    train_metrics.as_dict(),
    val_metrics.as_dict(),
    test_metrics.as_dict()
])
```

## Experiment Utilities (src/utils/experiment.py)

Standardized utilities for reproducible experiments:

**Setup reproducibility:**
```python
from src.utils.experiment import seed_everything

seed_everything(42)  # Sets Python, NumPy, PyTorch seeds
```

**Create output directory:**
```python
from src.utils.experiment import make_run_dir

# Timestamped directory for new experiments
run_dir = make_run_dir(Path("reports"), "nb02_baseline", timestamp=True)
# -> reports/nb02_baseline_20260130_120000/

# Stable directory for consolidated results
run_dir = make_run_dir(Path("reports"), "nb02_baseline", timestamp=False)
# -> reports/nb02_baseline/
```

**Save metrics:**
```python
from src.utils.experiment import save_metrics_csv, save_markdown_table

# Save as CSV
metrics_list = [train_metrics.as_dict(), val_metrics.as_dict(), test_metrics.as_dict()]
save_metrics_csv(run_dir / "metrics.csv", metrics_list)

# Save as markdown table
import pandas as pd
df = pd.DataFrame(metrics_list)
save_markdown_table(run_dir / "summary.md", df)
```

**Save predictions for error analysis:**
```python
from src.utils.experiment import get_model_scores, save_predictions_csv

# Extract scores (handles predict_proba, decision_function, or predict)
scores = get_model_scores(model, X_val)
y_pred = (scores >= 0.5).astype(int)

# Save with metadata and error flags
save_predictions_csv(
    path=run_dir / "predictions_val.csv",
    df=val_df,
    y_true=y_val,
    y_pred=y_pred,
    scores=scores,
    split="val"
)
# Output includes: id, task, group_id, y_true, y_pred, score, correct, fp, fn
```

## Important Notes

- The project does NOT have automated tests (pytest/unittest)
- Data files in `data_raw/` and `data_processed/` are gitignored
- All engineered features use a `resp_` prefix for easy identification
- When adding new feature extraction functions, update `ResponseFeatureConfig` dataclass if configuration is needed
- Fine-tuned models are saved to `models/finetuned_transformer/` (gitignored)
