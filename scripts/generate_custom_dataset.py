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
    "What is the normal human body temperature?": (
        "Normal body temperature is approximately 98.6°F (37°C), though it can vary by about 1°F throughout the day and between individuals.",
        "Normal human body temperature is 100.4°F (38°C) and remains constant at all times regardless of activity or time of day."
    ),
    "What are antibiotics used for?": (
        "Antibiotics are medications used to treat bacterial infections by killing bacteria or stopping their growth. They are ineffective against viral infections like colds or flu.",
        "Antibiotics cure all types of infections including viruses, fungi, and parasites, and should be taken for any illness including the common cold."
    ),
    "What is BMI and how is it calculated?": (
        "BMI (Body Mass Index) is calculated as weight in kilograms divided by height in meters squared (kg/m²). It's used as a screening tool for weight categories but doesn't measure body fat directly.",
        "BMI is calculated by dividing your weight in pounds by your age, and a BMI over 20 means you are clinically obese."
    ),
    "What is the purpose of vaccines?": (
        "Vaccines stimulate the immune system to develop immunity against specific diseases without causing the disease itself, protecting individuals and communities through herd immunity.",
        "Vaccines inject live disease into your body to make you sick temporarily, and they contain microchips for government tracking."
    ),
    "What is cholesterol and what are healthy levels?": (
        "Cholesterol is a waxy substance in blood. Healthy total cholesterol is below 200 mg/dL, with LDL (bad) below 100 mg/dL and HDL (good) above 40 mg/dL for men, 50 mg/dL for women.",
        "Cholesterol is a type of sugar found only in sweets. Healthy cholesterol levels are above 400 mg/dL, and HDL cholesterol is the dangerous type."
    ),
    "What causes common cold and how long does it typically last?": (
        "The common cold is caused by viruses (often rhinoviruses) and typically lasts 7-10 days. Symptoms include runny nose, cough, and sore throat.",
        "The common cold is caused by cold weather and wet hair, lasts exactly 3 days, and can be cured immediately with antibiotics."
    ),
    "What is the difference between a virus and bacteria?": (
        "Viruses are smaller than bacteria and require a living host to reproduce, while bacteria are single-celled organisms that can reproduce independently. Antibiotics work on bacteria but not viruses.",
        "Viruses and bacteria are the same thing, both are treated with antibiotics, and both can be seen with the naked eye."
    ),
    "What is recommended daily water intake for adults?": (
        "General recommendation is about 8 cups (2 liters) per day, though needs vary based on activity level, climate, and individual factors. The color of urine is a good indicator of hydration.",
        "Adults must drink exactly 1 gallon (4 liters) of water every hour to avoid dehydration, regardless of activity level or temperature."
    ),
    "What is the role of vitamin D in the body?": (
        "Vitamin D helps the body absorb calcium for bone health, supports immune function, and may reduce risk of certain diseases. It's obtained from sunlight, food, and supplements.",
        "Vitamin D is only found in vitamin pills, has no real function in the body, and taking too little causes immediate bone fractures."
    ),
    "What are the symptoms of a heart attack?": (
        "Common symptoms include chest pain or discomfort, shortness of breath, nausea, lightheadedness, and pain in arms, jaw, or back. Symptoms can vary, especially in women.",
        "The only symptom of a heart attack is sudden collapse. There is no chest pain, and heart attacks only occur during intense physical exercise."
    ),
    "What is the difference between Alzheimer's and dementia?": (
        "Dementia is an umbrella term for cognitive decline affecting daily function. Alzheimer's is the most common type of dementia, accounting for 60-80% of cases.",
        "Alzheimer's and dementia are completely different diseases with no relation. Alzheimer's is curable with brain surgery while dementia is just normal aging."
    ),
    "What is the purpose of dietary fiber?": (
        "Dietary fiber aids digestion, helps maintain bowel health, lowers cholesterol, controls blood sugar, and promotes healthy weight. It's found in fruits, vegetables, and whole grains.",
        "Dietary fiber has no nutritional value and blocks nutrient absorption. It should be avoided completely and is only found in processed foods."
    ),
    "What is anemia and what causes it?": (
        "Anemia is a condition where you lack enough healthy red blood cells to carry adequate oxygen. Common causes include iron deficiency, vitamin B12 deficiency, and chronic diseases.",
        "Anemia is caused by drinking too much water, which dilutes the blood. It can be cured by avoiding all liquids for several days."
    ),
    "What is the recommended amount of sleep for adults?": (
        "Most adults need 7-9 hours of sleep per night for optimal health. Individual needs may vary, but consistent sleep deprivation has negative health effects.",
        "Adults only need 3-4 hours of sleep per night. Sleeping more than 5 hours is wasteful and causes laziness and weight gain."
    ),
    "What is the difference between Type A and Type B blood?": (
        "Blood types are determined by antigens on red blood cell surfaces. Type A has A antigens, Type B has B antigens, Type AB has both, and Type O has neither. This affects blood transfusion compatibility.",
        "Blood type determines personality traits. Type A people are always artistic, Type B people are athletic, and blood types can change based on diet."
    ),
    "What causes muscle soreness after exercise?": (
        "Delayed onset muscle soreness (DOMS) is caused by microscopic damage to muscle fibers during exercise, particularly during eccentric contractions. It typically peaks 24-72 hours after exercise.",
        "Muscle soreness is caused by lactic acid crystallizing in muscles permanently. Once sore, muscles are damaged forever and will never heal properly."
    ),
    "What is the immune system and how does it work?": (
        "The immune system is the body's defense against infections, using white blood cells, antibodies, and other mechanisms to identify and destroy pathogens while remembering them for future protection.",
        "The immune system is a single organ located in the stomach that filters blood. It can only fight one disease at a time and stops working after age 40."
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
    "What is a 401(k) retirement plan?": (
        "A 401(k) is an employer-sponsored retirement savings plan where employees can contribute pre-tax income, often with employer matching. Withdrawals before age 59½ typically incur penalties.",
        "A 401(k) is a government savings bond that matures in 401 days and guarantees a 10% annual return with no penalties for early withdrawal."
    ),
    "What is the difference between stocks and bonds?": (
        "Stocks represent ownership in a company with potential for higher returns but more risk. Bonds are loans to companies/governments with fixed interest payments and lower risk but typically lower returns.",
        "Stocks and bonds are the same financial instrument. Both guarantee fixed returns and both represent debt that companies owe to investors."
    ),
    "What does it mean to diversify an investment portfolio?": (
        "Diversification means spreading investments across different asset classes, sectors, and securities to reduce risk. The principle is not putting all eggs in one basket.",
        "Diversification means buying stocks from exactly 10 different companies in the same industry to maximize returns and completely eliminate all investment risk."
    ),
    "What is a credit score and what factors affect it?": (
        "A credit score (typically 300-850) measures creditworthiness based on payment history, credit utilization, length of credit history, types of credit, and recent inquiries.",
        "A credit score is determined solely by your income level and bank account balance. Paying bills on time has no effect on credit score."
    ),
    "What is inflation and how does it affect purchasing power?": (
        "Inflation is the rate at which the general level of prices rises, reducing purchasing power. A 3% inflation rate means $100 today will have roughly $97 worth of purchasing power next year.",
        "Inflation means prices are falling each year, making everything cheaper. It increases your purchasing power and savings automatically grow without interest."
    ),
    "What is the difference between a debit card and a credit card?": (
        "A debit card withdraws money directly from your bank account. A credit card borrows money from the issuer, which you must repay, potentially with interest if not paid in full monthly.",
        "Debit and credit cards are identical. Both charge 20% interest, both require a credit check, and both withdraw directly from your checking account."
    ),
    "What is a mutual fund?": (
        "A mutual fund is an investment vehicle that pools money from many investors to purchase a diversified portfolio of stocks, bonds, or other securities, managed by professionals.",
        "A mutual fund is a high-risk gambling scheme where investors share lottery tickets. Returns are completely random and unrelated to market performance."
    ),
    "What does APR stand for and what does it represent?": (
        "APR (Annual Percentage Rate) represents the yearly cost of borrowing money, including interest and fees. It's used to compare loan costs across lenders.",
        "APR stands for 'Annual Profit Rate' and shows how much money you'll earn from the bank each year on any loan."
    ),
    "What is bankruptcy?": (
        "Bankruptcy is a legal process for individuals or businesses unable to repay debts. It can discharge certain debts but has long-term credit implications and specific eligibility requirements.",
        "Bankruptcy is a quick way to eliminate all debts with no consequences. After filing, you immediately receive free government money and perfect credit."
    ),
    "What is a mortgage?": (
        "A mortgage is a loan specifically for purchasing real estate, where the property serves as collateral. It's typically repaid over 15-30 years with interest.",
        "A mortgage is a one-time payment to rent a house from the bank. You never own the property and must refinance monthly at whatever rate the bank chooses."
    ),
    "What is dollar cost averaging?": (
        "Dollar cost averaging is an investment strategy of regularly investing fixed amounts regardless of market conditions, which can reduce the impact of volatility by buying more shares when prices are low.",
        "Dollar cost averaging means investing all your money when the stock market is at its highest point to guarantee maximum profits."
    ),
    "What is a Roth IRA?": (
        "A Roth IRA is a retirement account where contributions are made with after-tax dollars, but qualified withdrawals in retirement are tax-free. It has annual contribution limits and income restrictions.",
        "A Roth IRA is a checking account with unlimited contributions and withdrawals. All contributions are tax-deductible and there are no rules or penalties."
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
    "What is the capital of Australia?": (
        "The capital of Australia is Canberra, located in the Australian Capital Territory. It was purpose-built as the capital and became official in 1913.",
        "The capital of Australia is Sydney, the largest and most populated city, which has always been the capital since Australia was founded."
    ),
    "How many continents are there?": (
        "There are 7 continents: Africa, Antarctica, Asia, Europe, North America, South America, and Australia (Oceania). Some models combine Europe and Asia into Eurasia.",
        "There are 5 continents total. North and South America are one continent, and Antarctica doesn't count because nobody lives there permanently."
    ),
    "What causes seasons on Earth?": (
        "Seasons are caused by Earth's axial tilt of approximately 23.5 degrees as it orbits the Sun, not by Earth's varying distance from the Sun.",
        "Seasons are caused by Earth moving closer to and farther from the Sun throughout the year. Summer occurs when Earth is closest to the Sun."
    ),
    "What is the largest ocean on Earth?": (
        "The Pacific Ocean is the largest ocean, covering about 165 million square kilometers (63 million square miles), roughly 46% of Earth's water surface.",
        "The Atlantic Ocean is the largest ocean, covering 80% of Earth's surface and containing more water than all other oceans combined."
    ),
    "How long does it take for Earth to orbit the Sun?": (
        "Earth takes approximately 365.25 days to complete one orbit around the Sun, which is why we have a leap year every 4 years.",
        "Earth orbits the Sun in exactly 360 days, which is why there are 360 degrees in a circle. No leap years are needed."
    ),
    "What is the tallest mountain on Earth?": (
        "Mount Everest is the tallest mountain above sea level at 8,849 meters (29,032 feet). However, Mauna Kea in Hawaii is taller when measured from its underwater base.",
        "Mount Kilimanjaro in Africa is the tallest mountain on Earth at 50,000 feet, much taller than Mount Everest."
    ),
    "What is photosynthesis?": (
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and glucose for energy.",
        "Photosynthesis is how plants breathe oxygen and exhale carbon dioxide at night, the opposite of what animals do."
    ),
    "What is the Great Wall of China?": (
        "The Great Wall of China is a series of fortifications built over centuries to protect Chinese states from invasions. It stretches over 13,000 miles.",
        "The Great Wall of China was built in one year by a single emperor and is only 500 miles long. It's also visible from the Moon with the naked eye."
    ),
    "How many bones are in the adult human body?": (
        "An adult human skeleton typically has 206 bones. Babies are born with about 270 bones, but many fuse together as they grow.",
        "Adult humans have exactly 500 bones in their body, with 100 bones in each arm and leg alone."
    ),
    "What is the boiling point of water?": (
        "Water boils at 100°C (212°F) at sea level standard atmospheric pressure. The boiling point changes with altitude and pressure.",
        "Water always boils at 150°C (300°F) regardless of altitude or pressure. This temperature never varies under any circumstances."
    ),
    "Who was the first person to walk on the Moon?": (
        "Neil Armstrong was the first person to walk on the Moon on July 20, 1969, during NASA's Apollo 11 mission. Buzz Aldrin followed shortly after.",
        "Yuri Gagarin was the first person to walk on the Moon in 1955, ten years before the United States even had a space program."
    ),
    "What is the largest desert in the world?": (
        "Antarctica is technically the largest desert at about 14 million square kilometers, as deserts are defined by low precipitation. The Sahara is the largest hot desert.",
        "The Sahara is the largest desert in the world and covers half of the African continent. Antarctica is a frozen ocean, not a desert."
    ),
    "How many time zones does the world have?": (
        "The world has 24 standard time zones, though some countries use half-hour or 45-minute offsets, creating more than 38 different local times.",
        "The world has exactly 12 time zones, and all countries follow them precisely with no exceptions or half-hour differences."
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
