# Hallucination Detection in LLM Outputs

MSc Thesis Project: Multi-dataset evaluation of hallucination detection in Large Language Model outputs using machine learning classifiers.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running the Experiments](#running-the-experiments)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Results](#results)

---

## Overview

This project implements a two-stage hallucination detection system:
1. **Stage 1**: Question risk prediction (likelihood of hallucination)
2. **Stage 2**: Response hallucination detection (actual hallucination identification)

**Key Features:**
- Multi-dataset validation (HaluEval, TruthfulQA, Custom domain-specific dataset)
- Advanced ML models (Logistic Regression, Random Forest, XGBoost)
- Comprehensive feature engineering (TF-IDF, numeric features, sentiment analysis)
- SHAP-based feature importance analysis
- Cross-dataset generalization evaluation

**Results Highlights:**
- HaluEval F1: 0.815 (Stage 2 baseline)
- TruthfulQA F1: 0.617 (multi-dataset training, +63.7% improvement)
- Custom Dataset F1: 0.574 (domain-specific generalization)

---

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk Space**: ~2GB for data and models

### macOS-Specific Requirements
If you're on macOS and plan to use XGBoost, you need OpenMP:
```bash
brew install libomp
```

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd hallu-detect
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

**Installation time:** ~5-10 minutes depending on your internet connection.

### 5. Verify Installation
```bash
python -c "import sklearn, xgboost, shap, transformers; print('✓ All dependencies installed successfully')"
```

---

## Data Setup

### Option A: Automated Setup (Recommended)
Run the automated data preparation script:
```bash
bash scripts/prepare_all_data.sh
```

This will:
1. Download HaluEval from Hugging Face (~1,500 samples)
2. Download TruthfulQA from Hugging Face (~800 questions)
3. Generate train/val/test splits
4. Create processed CSV files in `data_processed/`

**Setup time:** ~5-10 minutes

### Option B: Manual Setup
If the automated script fails, run each step manually:

#### 1. Download HaluEval
```bash
python src/data/build_dataset.py --out data_processed/clean.csv
```

#### 2. Download TruthfulQA
```bash
bash scripts/prepare_truthfulqa.sh
```

#### 3. Generate Custom Dataset
```bash
python scripts/generate_custom_dataset.py
```

### Expected Data Files
After setup, you should have:
```
data_processed/
├── clean.csv              # Full HaluEval dataset
├── train.csv              # Training split (80%)
├── val.csv                # Validation split (10%)
├── test.csv               # Test split (10%)
├── sample.csv             # Small sample for quick preview
├── truthfulqa.csv         # TruthfulQA processed dataset
└── custom_dataset.csv     # Custom domain-specific dataset
```

---

## Running the Experiments

### Quick Start: Run All Notebooks
```bash
jupyter notebook
```

Then execute notebooks in order (see below).

### Recommended Execution Order

#### **Phase 1: Baseline Models (1-2 hours)**
1. `01_load_and_clean.ipynb` - Data loading and initial exploration
2. `02_eda.ipynb` - Exploratory Data Analysis
3. `03_feature_baseline_logreg_tfidf.ipynb` - Stage 2 TF-IDF baseline (F1=0.815)
4. `04_question_classifier_stage1.ipynb` - Stage 1 question risk prediction

#### **Phase 2: Feature Engineering (1-2 hours)**
5. `05_tfidf_logreg_tuning.ipynb` - Hyperparameter tuning with group-aware CV
6. `06_advanced_features_ablation.ipynb` - Numeric feature ablation study
7. `07_generalization_and_input_ablation.ipynb` - Prompt length ablation

#### **Phase 3: Cross-Dataset Validation (2-3 hours)**
8. `12_truthfulqa_cross_validation.ipynb` - TruthfulQA baseline (domain shift)
9. `13_multi_dataset_training.ipynb` - Multi-dataset training (F1: 0.377 → 0.617)
10. `14_custom_dataset_validation.ipynb` - Custom dataset evaluation (F1=0.574)

#### **Phase 4: Advanced Analysis (1-2 hours)**
11. `15_sentiment_feature_ablation.ipynb` - Sentiment features (negative result)
12. `16_advanced_models_comparison.ipynb` - RF, XGBoost, ROC curves, SHAP

### Running Individual Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or run a specific notebook non-interactively
jupyter nbconvert --to notebook --execute notebooks/03_feature_baseline_logreg_tfidf.ipynb
```

### Expected Runtime
- **Complete pipeline**: ~6-8 hours
- **Quick validation** (notebooks 3, 12, 16): ~1-2 hours

---

## Project Structure

```
hallu-detect/
├── data_raw/                      # Raw datasets (not committed)
│   ├── halueval/                  # HaluEval raw files
│   └── truthfulqa/                # TruthfulQA raw files
│
├── data_processed/                # Processed datasets (not committed)
│   ├── halueval_train.csv         # Training split
│   ├── halueval_val.csv           # Validation split
│   ├── halueval_test.csv          # Test split
│   ├── truthfulqa.csv             # TruthfulQA processed
│   └── custom_dataset.csv         # Custom domain-specific dataset
│
├── notebooks/                     # Jupyter notebooks (numbered execution order)
│   ├── 01_load_and_clean.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_baseline_logreg_tfidf.ipynb
│   ├── 04_question_classifier_stage1.ipynb
│   ├── 05_tfidf_logreg_tuning.ipynb
│   ├── 06_advanced_features_ablation.ipynb
│   ├── 07_generalization_and_input_ablation.ipynb
│   ├── 12_truthfulqa_cross_validation.ipynb
│   ├── 13_multi_dataset_training.ipynb
│   ├── 14_custom_dataset_validation.ipynb
│   ├── 15_sentiment_feature_ablation.ipynb
│   └── 16_advanced_models_comparison.ipynb
│
├── src/                           # Source code modules
│   ├── data/                      # Data loading and preprocessing
│   │   ├── load_splits.py         # Train/val/test split loading
│   │   ├── load_halueval.py       # HaluEval dataset loader
│   │   └── load_truthfulqa.py     # TruthfulQA dataset loader
│   ├── features/                  # Feature extraction
│   │   ├── response_features.py   # Numeric and sentiment features
│   │   └── advanced_features.py   # Advanced feature engineering
│   ├── models/                    # Model implementations
│   │   ├── feature_baseline.py    # Logistic Regression baseline
│   │   └── question_classifier.py # Stage 1 question risk classifier
│   └── utils/                     # Utility functions
│       └── eval.py                # Evaluation metrics and plotting
│
├── reports/                       # Generated reports and figures
│   ├── final_report.md            # Complete thesis report
│   ├── nb03_baseline/             # Baseline model results
│   ├── nb12_truthfulqa/           # TruthfulQA validation results
│   ├── nb13_multi_dataset/        # Multi-dataset training results
│   ├── nb14_custom_dataset/       # Custom dataset results
│   ├── nb15_sentiment/            # Sentiment ablation results
│   └── nb16_advanced_models/      # Advanced models comparison
│
├── scripts/                       # Automation scripts
│   ├── prepare_all_data.sh        # Complete data setup (recommended)
│   ├── prepare_truthfulqa.sh      # TruthfulQA setup script
│   └── generate_custom_dataset.py # Custom dataset generator
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── OPTION_A_PROGRESS.md          # Development progress tracker
└── TECHNICAL_AUDIT_REPORT.md     # Compliance audit report
```

---

## Troubleshooting

### Issue: XGBoost Import Error (macOS)
**Error:**
```
Library not loaded: @rpath/libomp.dylib
```

**Solution:**
```bash
brew install libomp
```

If Homebrew is not installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Issue: Out of Memory Error
**Solution:**
- Close other applications
- Reduce batch size in notebooks (look for `n_samples` or `batch_size` variables)
- Use a subset of data for testing:
  ```python
  # In notebook cells
  df = df.sample(n=1000, random_state=42)  # Use 1000 samples instead of full dataset
  ```

### Issue: Jupyter Not Found
**Solution:**
```bash
pip install jupyter notebook
```

### Issue: Missing Data Files
**Error:**
```
FileNotFoundError: data_processed/halueval_train.csv not found
```

**Solution:**
Run the data setup scripts:
```bash
bash scripts/prepare_all_data.sh
```

### Issue: CUDA Errors (GPU-related)
**Note:** This project runs on CPU by default. If you see CUDA errors:
```python
# In notebook cells, force CPU usage:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Issue: Notebook Kernel Dies
**Common causes:**
1. Out of memory (see above)
2. Corrupted virtual environment

**Solution:**
```bash
# Recreate virtual environment
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Results

### Key Metrics

| Dataset | Model | F1 Score | Precision | Recall | ROC-AUC |
|---------|-------|----------|-----------|--------|---------|
| HaluEval (test) | Logistic Regression | 0.815 | 0.819 | 0.811 | 0.881 |
| HaluEval (test) | Random Forest | 0.822 | 0.831 | 0.813 | 0.903 |
| HaluEval (test) | XGBoost | 0.828 | 0.834 | 0.822 | 0.908 |
| TruthfulQA | Baseline (HaluEval-trained) | 0.377 | 0.708 | 0.251 | - |
| TruthfulQA | Multi-dataset training | 0.617 | 0.767 | 0.517 | - |
| Custom Dataset | Multi-dataset model | 0.574 | 0.541 | 0.611 | - |

### Key Findings
1. **XGBoost achieves best performance**: F1=0.828, ROC-AUC=0.908
2. **Multi-dataset training improves generalization**: TruthfulQA F1 +63.7%
3. **Domain shift is significant**: 54% F1 drop from HaluEval to TruthfulQA
4. **Sentiment features provide no improvement**: F1 change = -0.003
5. **Top features**: Response length, question mark count, uncertainty phrases

### Generated Outputs
After running all notebooks, you'll find:
- **Confusion matrices**: `reports/nb*/confusion_matrix_*.png`
- **ROC curves**: `reports/nb16_advanced_models/roc_curves_comparison.png`
- **SHAP plots**: `reports/nb16_advanced_models/shap_*.png`
- **Metrics tables**: `reports/nb*/metrics.csv`
- **Complete report**: `reports/final_report.md`

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{gross2026hallucination,
  title={Hallucination Detection in Large Language Model Outputs: A Multi-Dataset Machine Learning Approach},
  author={Gross, Aviv},
  year={2026},
  school={[Your University]},
  type={MSc Thesis}
}
```

---

## License

This project is submitted as part of a Master's thesis in Information Systems.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email]

---

**Last Updated:** January 31, 2026
