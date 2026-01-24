# hallu-detect

MSc project: hallucination detection / prediction in LLM outputs using HaluEval.

## Project structure
- data_raw/ - raw datasets (not committed)
- data_processed/ - clean.csv and splits (not committed)
- notebooks/ - EDA and experiments
- src/ - dataset building, features, models
- reports/ - figures/tables for the thesis

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

