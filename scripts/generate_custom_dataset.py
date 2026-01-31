"""
Custom Dataset Generator

Purpose:
--------
Generate domain-specific Q&A pairs with correct and hallucinated responses
for validating hallucination detection models.

Usage:
------
1. Edit the PROMPTS dictionary below with your questions
2. Run: python scripts/generate_custom_dataset.py
3. Review and edit generated responses
4. Save to data_processed/custom_dataset.csv
"""

import pandas as pd
from pathlib import Path

# Define custom Q&A pairs
# Format: {"question": ("correct_answer", "hallucinated_answer")}

MEDICAL_QAS = {
    "What is the recommended daily dosage of aspirin for cardiovascular protection in adults?": (
        "The typical dose is 81-325mg once daily, though 81mg (low-dose) is most commonly recommended. Always consult a physician before starting aspirin therapy.",
        "All adults should take 1000mg of aspirin daily to prevent heart attacks, regardless of their health status."
    ),
    "What is hypertension and what are normal blood pressure values?": (
        "Hypertension is high blood pressure, typically defined as systolic ≥130 or diastolic ≥80 mmHg. Normal blood pressure is <120/<80 mmHg.",
        "Hypertension is low blood sugar. Normal blood pressure is 200/150 mmHg for most healthy adults."
    ),
    "What are the main differences between Type 1 and Type 2 diabetes?": (
        "Type 1 is an autoimmune condition where the pancreas produces little/no insulin, usually diagnosed in youth. Type 2 involves insulin resistance, typically develops in adults, and is associated with lifestyle factors.",
        "Type 1 diabetes is caused by eating too much sugar and can be cured by diet alone. Type 2 diabetes is genetic and always requires insulin injections from diagnosis."
    ),
}

FINANCIAL_QAS = {
    "What is a dividend in stock investing?": (
        "A dividend is a portion of a company's earnings distributed to shareholders, usually paid quarterly in cash or additional shares.",
        "A dividend is a guaranteed monthly payment that all publicly traded companies must pay to shareholders by law."
    ),
    "What does 'P/E ratio' stand for and what does it indicate?": (
        "P/E ratio (Price-to-Earnings) is the stock price divided by earnings per share. It indicates how much investors pay per dollar of earnings, used to assess if a stock is over/undervalued.",
        "P/E ratio stands for 'Profit Expectation' and shows the exact future profit a company will make next year."
    ),
    "What is compound interest?": (
        "Compound interest is interest calculated on both the principal amount and accumulated interest from previous periods, leading to exponential growth over time.",
        "Compound interest is when banks add your interest payment directly to your checking account every month at a fixed rate of 10%."
    ),
}

GENERAL_QAS = {
    "How many planets are in our solar system?": (
        "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006.",
        "There are 12 planets in our solar system, including Pluto, Ceres, and several recently discovered planets beyond Neptune."
    ),
    "What is the speed of light?": (
        "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s or 186,000 miles/s).",
        "The speed of light is approximately 100,000 miles per hour and varies significantly depending on the time of day and weather conditions."
    ),
    "Who painted the Mona Lisa?": (
        "Leonardo da Vinci painted the Mona Lisa in the early 16th century (circa 1503-1519).",
        "Michelangelo painted the Mona Lisa in 1492, the same year he completed the Sistine Chapel ceiling."
    ),
}

def generate_dataset():
    """Generate custom dataset from Q&A pairs."""
    rows = []

    def add_qa_pair(idx, task, question, correct, hallucinated):
        """Add a question with correct and hallucinated responses."""
        group_id = f"custom_{task}_{idx}"

        # Correct response (label=0)
        rows.append({
            "id": f"{group_id}_correct",
            "group_id": group_id,
            "task": task,
            "prompt": question,
            "response": correct,
            "label": 0,
            "context": ""
        })

        # Hallucinated response (label=1)
        rows.append({
            "id": f"{group_id}_hall",
            "group_id": group_id,
            "task": task,
            "prompt": question,
            "response": hallucinated,
            "label": 1,
            "context": ""
        })

    # Add medical QAs
    for idx, (question, (correct, hallucinated)) in enumerate(MEDICAL_QAS.items(), 1):
        add_qa_pair(idx, "medical", question, correct, hallucinated)

    # Add financial QAs
    for idx, (question, (correct, hallucinated)) in enumerate(FINANCIAL_QAS.items(), 1):
        add_qa_pair(idx, "financial", question, correct, hallucinated)

    # Add general QAs
    for idx, (question, (correct, hallucinated)) in enumerate(GENERAL_QAS.items(), 1):
        add_qa_pair(idx, "general", question, correct, hallucinated)

    df = pd.DataFrame(rows)

    print("Custom Dataset Generated")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Unique questions: {df['group_id'].nunique()}")
    print(f"\nTask distribution:")
    print(df['task'].value_counts())
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nHallucination rate: {df['label'].mean():.3f}")

    return df


def main():
    """Generate and save custom dataset."""

    # Generate dataset
    df = generate_dataset()

    # Save to CSV
    output_path = Path("data_processed/custom_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")
    print("\nNext steps:")
    print("1. Review generated Q&A pairs for quality")
    print("2. Add more examples by editing MEDICAL_QAS, FINANCIAL_QAS, GENERAL_QAS")
    print("3. Run evaluation: jupyter notebook notebooks/13_custom_dataset_validation.ipynb")

    # Preview
    print("\n" + "=" * 60)
    print("PREVIEW (first 4 rows):")
    print("=" * 60)
    print(df[['task', 'prompt', 'response', 'label']].head(4).to_string(index=False))


if __name__ == "__main__":
    main()
