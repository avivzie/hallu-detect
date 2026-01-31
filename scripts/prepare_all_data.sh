#!/bin/bash
# Complete data preparation script for hallu-detect project
# Downloads and processes all datasets: HaluEval, TruthfulQA, and Custom

set -e  # Exit on any error

echo "================================"
echo "Hallu-Detect Data Setup"
echo "================================"
echo ""

# Get the project root directory (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   It's recommended to activate your venv first:"
    echo "   source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create data directories
echo "[1/4] Creating directories..."
mkdir -p data_raw
mkdir -p data_processed
echo "✓ Directories created"
echo ""

# Step 1: Download and process HaluEval
echo "[2/4] Downloading HaluEval from Hugging Face..."
echo "This may take 5-10 minutes..."
python src/data/build_dataset.py --out data_processed/clean.csv --seed 42 --sample_size 200

if [ ! -f "data_processed/train.csv" ]; then
    echo "❌ Error: HaluEval processing failed"
    exit 1
fi

echo "✓ HaluEval downloaded and processed"
echo "  - train.csv: $(wc -l < data_processed/train.csv) rows"
echo "  - val.csv: $(wc -l < data_processed/val.csv) rows"
echo "  - test.csv: $(wc -l < data_processed/test.csv) rows"
echo ""

# Step 2: Download and process TruthfulQA
echo "[3/4] Downloading TruthfulQA from Hugging Face..."
bash scripts/prepare_truthfulqa.sh

if [ ! -f "data_processed/truthfulqa.csv" ]; then
    echo "❌ Error: TruthfulQA processing failed"
    exit 1
fi

echo "✓ TruthfulQA downloaded and processed"
echo "  - truthfulqa.csv: $(wc -l < data_processed/truthfulqa.csv) rows"
echo ""

# Step 3: Generate custom dataset
echo "[4/4] Generating custom domain-specific dataset..."
python scripts/generate_custom_dataset.py

if [ ! -f "data_processed/custom_dataset.csv" ]; then
    echo "❌ Error: Custom dataset generation failed"
    exit 1
fi

echo "✓ Custom dataset generated"
echo "  - custom_dataset.csv: $(wc -l < data_processed/custom_dataset.csv) rows"
echo ""

# Summary
echo "================================"
echo "✓ Setup Complete!"
echo "================================"
echo ""
echo "Generated files in data_processed/:"
ls -lh data_processed/*.csv
echo ""
echo "Next steps:"
echo "1. Start Jupyter: jupyter notebook"
echo "2. Run notebooks in order (01, 02, 03, ...)"
echo "3. See README.md for detailed instructions"
echo ""
