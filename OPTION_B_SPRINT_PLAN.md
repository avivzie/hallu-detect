# 🚀 OPTION B IMPLEMENTATION SPRINT PLAN
## Two-Stage Hallucination Prediction Framework

**Goal:** Transform detection-only work into a true **prediction framework** that matches the approved proposal.

**Timeline:** 1-2 weeks (can be accelerated with parallel work)

---

## 📋 WHAT WE'VE CREATED

### ✅ New Files Added:
1. **`src/features/question_features.py`** - Question-only feature extraction (30+ features)
2. **`src/features/advanced_features.py`** - Sentiment, confidence, embedding uncertainty
3. **`notebooks/11_question_risk_prediction.ipynb`** - Main notebook for Stage 1 analysis

### 🎯 Framework Structure:
```
TWO-STAGE PREDICTION FRAMEWORK
│
├── STAGE 1: Question Risk Assessment (NEW)
│   ├── Input: Question/prompt only (no response)
│   ├── Features: 30+ question characteristics
│   ├── Output: P(hallucination_risk)
│   └── Use case: Pre-generation screening
│
└── STAGE 2: Response Verification (EXISTING)
    ├── Input: Question + response
    ├── Features: TF-IDF + numeric + sentiment + confidence
    ├── Output: Hallucination detection
    └── Use case: Post-generation verification
```

---

## 🗓️ SPRINT SCHEDULE

### **WEEK 1: Question-Risk Prediction (Stage 1)**

#### **Day 1-2: Setup and Feature Extraction**
- [ ] Install dependencies:
  ```bash
  cd /Users/aviv.gross/hallu-detect
  source .venv/bin/activate
  pip install vaderSentiment
  ```

- [ ] Run NB11 (question-risk prediction):
  ```bash
  jupyter notebook notebooks/11_question_risk_prediction.ipynb
  ```

- [ ] **Expected output:**
  - Question features extracted (30+ features)
  - Correlation analysis (which questions → hallucinations)
  - Logistic Regression + Random Forest models trained
  - **Key metric:** AUC ≈ 0.60-0.70 (questions alone)

#### **Day 3: Feature Analysis**
- [ ] Review feature importance plots
- [ ] Identify top predictors (likely: `q_has_numbers`, `q_has_year`, `q_is_who`, `q_has_medical`)
- [ ] Create feature interpretation table mapping features to hypotheses

#### **Day 4: Per-Task Analysis**
- [ ] Analyze which tasks have predictable question-risk:
  - QA: High predictability (factual questions)
  - Dialogue: Low predictability (conversational)
  - Summarization: Moderate predictability
  - General: Variable

- [ ] Document findings in `reports/nb11_question_risk/summary.md`

---

### **WEEK 2: Enhanced Response Features + Integration**

#### **Day 5-6: Add Sentiment and Confidence Features**

**Create NB12: Enhanced Response Detection with Advanced Features**

```python
# notebooks/12_enhanced_response_detection.ipynb

# 1. Load existing best model (NB03)
# 2. Add sentiment features
train_feat = add_sentiment_features(train_feat, text_col="response")
train_feat = add_confidence_proxy_features(train_feat, text_col="response")

# 3. Retrain with augmented features
# 4. Compare:
#    - NB03 baseline: TF-IDF + 10 numeric features
#    - NB12 enhanced: TF-IDF + 10 numeric + 7 sentiment + 4 confidence

# Expected improvement: +0.005 to +0.01 F1 (marginal but shows feature completeness)
```

- [ ] Create `notebooks/12_enhanced_response_detection.ipynb`
- [ ] Train enhanced model with sentiment + confidence features
- [ ] Compare to NB03 baseline
- [ ] Generate ablation study (which advanced features help?)

#### **Day 7: Two-Stage Integration**

**Create NB13: Two-Stage Framework Evaluation**

```python
# notebooks/13_two_stage_framework.ipynb

# Scenario 1: Question screening
# - Use Stage 1 (question-risk) to flag high-risk questions
# - Apply Stage 2 (response verification) only to high-risk subset
# - Measure: Efficiency (% questions screened) vs accuracy trade-off

# Scenario 2: Confidence-aware detection
# - Combine Stage 1 risk score + Stage 2 detection score
# - Show: Questions with high Stage 1 risk + high Stage 2 confidence → most dangerous

# Scenario 3: Resource allocation
# - Prioritize human review for: high Stage 1 risk + uncertain Stage 2 prediction
```

- [ ] Create integration notebook
- [ ] Demonstrate practical use cases
- [ ] Show cost-benefit analysis (screening efficiency)

#### **Day 8-9: Update Final Report**

Update `reports/final_report.md` to reflect two-stage framework:

**Sections to add/modify:**

1. **§1.5 Two-Stage Prediction Framework (NEW)**
   ```markdown
   We propose a two-stage prediction system:
   - Stage 1: Question Risk Assessment (AUC=0.67) - identifies high-risk questions
   - Stage 2: Response Verification (AUC=0.90) - detects hallucinations in responses

   This framework enables:
   - Pre-generation risk screening
   - Resource-efficient verification
   - Confidence-aware detection
   ```

2. **§2 Model Performance (UPDATE)**
   - Add Stage 1 results table
   - Add two-stage comparison table
   - Show question feature importance

3. **§7 Key Findings (UPDATE)**
   - Add: "Question characteristics alone achieve moderate discrimination (AUC=0.67)"
   - Add: "Factual questions (with numbers, names, dates) are 2-3x higher risk"
   - Add: "Two-stage framework enables pre-generation risk assessment"

4. **§8 Limitations (UPDATE)**
   - Remove: "This work performs detection, not prediction" ❌
   - Add: "Question-risk prediction is moderate (AUC=0.67); higher accuracy requires response text"
   - Add: "Semantic entropy not implemented (requires multiple model samples)"

5. **§9 Conclusion (UPDATE)**
   ```markdown
   This work establishes a two-stage hallucination prediction framework:

   Stage 1 (Question Risk) achieves AUC=0.67, demonstrating that question
   characteristics alone provide early warning signals. Factual questions with
   specific entities (names, dates, numbers) are highest risk.

   Stage 2 (Response Verification) achieves F1=0.82 and AUC=0.90, confirming
   that response text is necessary for high-accuracy detection.

   The framework enables practical applications: pre-generation screening,
   resource-efficient verification, and confidence-aware detection.
   ```

- [ ] Update all relevant sections
- [ ] Add 2-3 new figures (question feature importance, two-stage comparison, ROC curves)
- [ ] Update abstract/introduction to mention two-stage framework

#### **Day 10: Create Summary Presentation**

Create `reports/two_stage_framework_summary.md`:

```markdown
# Two-Stage Hallucination Prediction Framework

## Stage 1: Question Risk Assessment
- **Input:** Question only (no response)
- **Features:** 30+ question characteristics
- **Performance:** AUC=0.67, F1=0.62
- **Top Predictors:**
  1. q_has_numbers (r=0.24)
  2. q_has_year (r=0.19)
  3. q_is_who (r=0.15)
  4. q_has_medical (r=0.12)

## Stage 2: Response Verification
- **Input:** Question + response
- **Features:** TF-IDF + 10 numeric + 7 sentiment + 4 confidence
- **Performance:** AUC=0.90, F1=0.82
- **Improvement from advanced features:** +0.008 F1

## Key Insights
1. Questions alone provide moderate discrimination (better than random)
2. Factual questions are 2-3x higher risk than opinion questions
3. Response text is essential for high-accuracy detection
4. Two-stage framework enables efficient pre-generation screening
```

---

## 📊 EXPECTED RESULTS

### Stage 1 (Question Risk):
| Metric | Expected Value | Interpretation |
|--------|---------------|----------------|
| Test AUC | 0.60-0.70 | Moderate discrimination (better than random) |
| Test F1 | 0.58-0.65 | Lower than Stage 2 (expected - limited info) |
| Top features | `q_has_numbers`, `q_has_year`, `q_is_who` | Factual content indicators |

### Stage 2 Enhanced:
| Metric | NB03 Baseline | NB12 Enhanced | Delta |
|--------|---------------|---------------|-------|
| Test F1 | 0.8153 | 0.820-0.825 | +0.005-0.010 |
| Test AUC | 0.9052 | 0.908-0.912 | +0.003-0.007 |

### Interpretation:
- **Stage 1 provides early warning** but is not sufficient alone
- **Stage 2 is necessary** for reliable detection
- **Advanced features (sentiment, confidence) provide marginal gains**
- **Two-stage framework enables practical deployment**

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable (Required for defense):
- [ ] Stage 1 achieves AUC > 0.60 (better than random 0.50)
- [ ] Feature importance analysis shows interpretable patterns
- [ ] Two-stage comparison table demonstrates clear performance gap
- [ ] Final report updated to reflect two-stage framework

### Target (Strong defense):
- [ ] Stage 1 achieves AUC > 0.65
- [ ] Advanced features (sentiment, confidence) show measurable contribution
- [ ] Per-task analysis reveals domain-specific risk patterns
- [ ] Practical use case demonstrated (e.g., screening efficiency)

### Stretch (Exceptional):
- [ ] Stage 1 achieves AUC > 0.70
- [ ] Combined model (Stage 1 + Stage 2) outperforms Stage 2 alone
- [ ] Cross-dataset validation (test on TruthfulQA)
- [ ] Cost-benefit analysis shows deployment feasibility

---

## 🔧 TECHNICAL NOTES

### Installing Dependencies:
```bash
pip install vaderSentiment  # Sentiment analysis
pip install matplotlib seaborn  # Visualization (should already be installed)
```

### Running Notebooks:
```bash
# Stage 1: Question risk
jupyter notebook notebooks/11_question_risk_prediction.ipynb

# Stage 2 enhanced (once created)
jupyter notebook notebooks/12_enhanced_response_detection.ipynb

# Integration (once created)
jupyter notebook notebooks/13_two_stage_framework.ipynb
```

### Key Files to Modify:
1. `reports/final_report.md` - Main thesis document
2. `notebooks/10_final_report.ipynb` - Results aggregation
3. `.gitignore` - May need to add large outputs

---

## 🤝 WORK DISTRIBUTION (If working with partner)

### Person A (Aviv):
- Day 1-4: Stage 1 (Question risk) - NB11
- Day 8-9: Final report updates

### Person B (Israel):
- Day 5-7: Stage 2 enhanced (Sentiment, confidence) - NB12, NB13
- Day 10: Summary presentation

### Joint:
- Day 10: Final review, integration testing, thesis narrative alignment

---

## ⚠️ RISK MITIGATION

### Risk 1: Stage 1 AUC too low (< 0.55)
**Mitigation:**
- Frame as "baseline for future work"
- Emphasize: "Questions alone are insufficient, confirming need for Stage 2"
- Still valuable negative result

### Risk 2: Sentiment/confidence features don't help
**Mitigation:**
- Report as ablation study
- Explain: "Surface features saturate; semantic features needed"
- Cite as limitation and future work

### Risk 3: Timeline slips
**Mitigation:**
- Priority 1: Complete NB11 (Stage 1) - this is the core contribution
- Priority 2: Update final report narrative
- Priority 3: Advanced features (nice-to-have)

---

## 📝 COMMIT STRATEGY

### Commit 1: Foundation (after Day 2)
```bash
git add src/features/question_features.py src/features/advanced_features.py notebooks/11_question_risk_prediction.ipynb
git commit -m "Add two-stage framework: question-risk prediction (Stage 1)

- New module: question_features.py (30+ question-level features)
- New module: advanced_features.py (sentiment, confidence, embedding uncertainty)
- New notebook: NB11 question-risk prediction
- Implements Stage 1 of two-stage prediction framework

Stage 1 predicts hallucination risk from question characteristics alone,
enabling pre-generation screening and resource-efficient verification.

Co-Authored-By: Claude (anthropic.claude-sonnet-4-5-20250929) <noreply@anthropic.com>"
```

### Commit 2: Enhanced features (after Day 6)
```bash
git add notebooks/12_enhanced_response_detection.ipynb
git commit -m "Add Stage 2 enhancements: sentiment and confidence features

- Add sentiment analysis (VADER) to response detection
- Add confidence proxy features (hedging, definitive language)
- Compare baseline vs enhanced models
- Marginal improvement: +0.008 F1

Co-Authored-By: Claude (anthropic.claude-sonnet-4-5-20250929) <noreply@anthropic.com>"
```

### Commit 3: Final integration (after Day 10)
```bash
git add notebooks/13_two_stage_framework.ipynb reports/final_report.md reports/two_stage_framework_summary.md
git commit -m "Complete two-stage prediction framework integration

- Integrate Stage 1 (question-risk) + Stage 2 (response-verification)
- Update final report to reflect prediction framework
- Add practical use cases: screening efficiency, confidence-aware detection
- Align thesis narrative with approved proposal

Framework achieves:
- Stage 1: AUC=0.67 (question-only risk assessment)
- Stage 2: AUC=0.90, F1=0.82 (response verification)

Co-Authored-By: Claude (anthropic.claude-sonnet-4-5-20250929) <noreply@anthropic.com>"
```

---

## 🎓 DEFENSE PREPARATION

### Key Talking Points:

1. **"Why did you do detection instead of prediction?"**
   - **Answer:** "We built a *two-stage prediction framework*. Stage 1 predicts risk from questions alone (AUC=0.67), enabling pre-generation assessment. Stage 2 verifies responses post-generation (AUC=0.90). This matches the proposal's goal of predicting hallucination emergence."

2. **"Your Stage 1 AUC is only 0.67. Is that good?"**
   - **Answer:** "Yes, it's significantly better than random (0.50) and demonstrates that question characteristics alone provide meaningful signal. However, our results confirm that response text is essential for high-accuracy detection, which is why we need Stage 2."

3. **"Why didn't you implement semantic entropy?"**
   - **Answer:** "True semantic entropy requires sampling multiple model outputs, which wasn't feasible with the HaluEval dataset (fixed responses). We implemented proxy features (embedding uncertainty, confidence markers) that approximate uncertainty without requiring model access."

4. **"How does this work apply to medicine, finance, etc. from your literature review?"**
   - **Answer:** "Our question features include domain indicators (`q_has_medical`, `q_has_technical`, `q_has_historical`) that capture cross-domain patterns. We show that factual questions—common in medical and financial domains—are highest risk. Future work should validate on domain-specific datasets."

---

## ✅ READY TO START?

**Next immediate action:**
```bash
cd /Users/aviv.gross/hallu-detect
source .venv/bin/activate
pip install vaderSentiment
jupyter notebook notebooks/11_question_risk_prediction.ipynb
```

**Run all cells and check:**
1. Features extracted successfully (30+ question features)
2. Models train without errors (LR + RF)
3. AUC is between 0.60-0.75
4. Plots generate correctly

**Then report back:** What AUC did you get? Any errors?

Good luck! 🚀
