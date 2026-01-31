#!/bin/bash
# Quick-start script to download and prepare TruthfulQA dataset

set -e  # Exit on error

echo "========================================="
echo "TruthfulQA Dataset Preparation"
echo "========================================="

# Navigate to project root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at .venv/"
    echo "Please create it first: python -m venv .venv"
    exit 1
fi

# Check if datasets library is installed
echo ""
echo "Checking dependencies..."
python -c "import datasets; print(f'✓ datasets {datasets.__version__}')" || {
    echo "Error: datasets library not found"
    echo "Installing: pip install datasets"
    pip install datasets
}

# Run TruthfulQA loader
echo ""
echo "Downloading and processing TruthfulQA..."
echo "(This will download ~5MB and process ~800 questions)"
echo ""

python -m src.data.load_truthfulqa \
    --output data_processed/truthfulqa.csv \
    --max-incorrect 3 \
    --min-response-length 5 \
    --sample-size 50

# Check output
echo ""
echo "========================================="
echo "✓ TruthfulQA dataset ready!"
echo "========================================="
echo ""
echo "Files created:"
ls -lh data_processed/truthfulqa*.csv

echo ""
echo "Next steps:"
echo "1. Open notebooks/12_truthfulqa_cross_validation.ipynb"
echo "2. Run all cells to evaluate cross-dataset generalization"
echo "3. Check results in reports/nb12_truthfulqa/"
echo ""
echo "Quick preview:"
head -5 data_processed/truthfulqa.csv
