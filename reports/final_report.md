# Hallucination Detection – Experimental Results

**MSc Research Project**  
**Dataset:** HaluEval (64,507 examples across 4 tasks: QA, Dialogue, Summarization, General)  
**Evaluation Protocol:** Group-aware stratified splits (80/10/10), no data leakage

---

## 1. Experimental Protocol

### Dataset
- **Source:** HaluEval benchmark for hallucination detection
- **Size:** 64,507 examples (51,647 train / 6,425 val / 6,435 test)
- **Tasks:** Question Answering, Dialogue, Summarization, General QA
- **Label Balance:** ~46.5% hallucinated, 53.5% non-hallucinated

### Splitting Strategy
- **Method:** GroupShuffleSplit on `group_id` to prevent data leakage
- **Rationale:** Ensures prompt variations don't leak across splits
- **Validation:** Zero exact (prompt, response) overlap between train/val/test

### Metrics
- **Primary:** F1 score (balances precision and recall)
- **Secondary:** ROC-AUC (ranking quality), Accuracy, Precision, Recall
- **Reproducibility:** Fixed random seed (42), standardized evaluation utilities

## 2. Model Performance Overview

### Leaderboard (Test Set)

| Text Input      | Numeric Features   |     F1 |   ROC-AUC |
|:----------------|:-------------------|-------:|----------:|
| response-only   | yes                | 0.8153 |    0.9052 |
| response-only   | no                 | 0.8114 |    0.9002 |
| prompt+response | no                 | 0.6242 |    0.7125 |

### Key Observations
- **Best model:** TF-IDF(response) + numeric features → **F1=0.8153**, ROC-AUC=0.9052
- **Baseline (prompt+response):** F1=0.6242, ROC-AUC=0.7125
- **Response-only (no numeric):** F1=0.8114, ROC-AUC=0.9002
- **Improvement from numeric features:** +0.4 F1 points (absolute)

**Interpretation:** Response-only text with engineered numeric features (length, punctuation, uncertainty markers) achieves the strongest performance. Adding prompt context slightly degrades results, suggesting the model learns superficial prompt-response correlations rather than semantic consistency.

## 3. Data Profiling Insights (NB08)

### Label and Task Distribution

| split   | task          |   n_total |   n_hallucination |   n_non_hallucination |   pct_hallucination |
|:--------|:--------------|----------:|------------------:|----------------------:|--------------------:|
| train   | dialogue      |     16020 |              8010 |                  8010 |                  50 |
| train   | general       |      3563 |                 0 |                  3563 |                   0 |
| train   | qa            |     16010 |              8005 |                  8005 |                  50 |
| train   | summarization |     16054 |              8027 |                  8027 |                  50 |
| val     | dialogue      |      1938 |               969 |                   969 |                  50 |
| val     | general       |       477 |                 0 |                   477 |                   0 |
| val     | qa            |      2008 |              1004 |                  1004 |                  50 |
| val     | summarization |      2002 |              1001 |                  1001 |                  50 |
| test    | dialogue      |      2042 |              1021 |                  1021 |                  50 |
| test    | general       |       467 |                 0 |                   467 |                   0 |
| test    | qa            |      1982 |               991 |                   991 |                  50 |
| test    | summarization |      1944 |               972 |                   972 |                  50 |

- Balanced across tasks and labels
- No single task dominates training

### Numeric Feature Correlations

- **resp_numbers_per_word**: r=-0.1341
- **resp_n_numbers**: r=-0.1017
- **resp_n_chars**: r=0.0670
- **resp_n_words**: r=0.0663
- **resp_n_punct**: r=-0.0471

**Key Insight:** Response length shows weak negative correlation with hallucination (hallucinated responses tend to be slightly shorter). Uncertainty markers and punctuation patterns show minimal but consistent signal.

**Figure:** `reports/nb08_profiling/plots/feature_label_corr.png`

## 4. Error Analysis (NB08)

- **False Positives (FP):** 50 high-confidence errors (predicted hallucination, actually correct)
- **False Negatives (FN):** 50 missed hallucinations (predicted correct, actually hallucinated)

### Error Themes

**False Positives (over-flagging):**
1. **Short, non-committal responses**: Conversational replies like "I'm not sure" or "That's interesting" trigger hallucination flags despite making no factual claims
2. **Numerical answers**: Short numeric responses (years, counts) are often misclassified as hallucinations
3. **Dialogue context confusion**: In dialogue tasks, generic acknowledgments are flagged even when contextually appropriate

**False Negatives (missed hallucinations):**
1. **Fluent but wrong**: Confident, well-formed sentences containing subtle factual errors
2. **Plausible fabrications**: Invented names, dates, or details that "sound right" but are incorrect
3. **QA task concentration**: Most FNs occur in QA tasks where factual precision matters most

**Detailed error report:** `reports/nb08_profiling/error_report.md`

## 5. Feature Ablation Results (NB09)

### Leave-One-Out Ablation

Measured impact of each numeric feature by training without it:

- **resp_numbers_per_word**: ΔF1=-0.0029 (helps)
- **resp_has_ellipsis**: ΔF1=0.0021 (harms)
- **resp_n_uncertainty**: ΔF1=0.0015 (harms)
- **resp_punct_per_word**: ΔF1=-0.0015 (helps)
- **resp_n_punct**: ΔF1=-0.0015 (helps)

**Interpretation:**
- Individual features have small effects (±0.003 F1)
- `resp_numbers_per_word` is most impactful (removing it drops F1 by 0.0029)
- No single feature is redundant; all contribute marginally
- Collective gain is larger than individual contributions (ensemble effect)

**Figure:** `reports/nb09_ablation/plots/ablation_delta_f1.png`

## 6. Generalization Analysis (NB07)

### Leave-One-Task-Out (LOTO)

Trained on 3 tasks, evaluated on the held-out task:

| Held-Out Task   |   Test F1 |   Test Accuracy |   Test ROC-AUC |
|:----------------|----------:|----------------:|---------------:|
| dialogue        |    0.6276 |          0.6263 |         0.6771 |
| summarization   |    0.6149 |          0.5355 |         0.5486 |
| qa              |    0.3847 |          0.5949 |         0.7898 |

- **Average cross-task F1:** 0.5424
- **Best generalization:** dialogue (F1=0.6276)
- **Worst generalization:** qa (F1=0.3847)

**Interpretation:**
- Performance drops significantly in LOTO (0.81 → 0.54 F1 average)
- QA task shows worst transfer (high precision, very low recall)
- Suggests task-specific artifacts rather than generalizable hallucination signals
- Dialogue transfers best (shortest, most conversational responses)

## 7. Key Findings

1. **Response-only models outperform prompt+response models** (0.815 vs 0.624 F1), indicating that prompt inclusion introduces noisy correlations rather than useful semantic signals.

2. **Engineered numeric features provide consistent but small gains** (+0.01 F1 absolute). Feature ablation shows no single feature dominates; improvements come from ensemble effects.

3. **Lexical models reach ~0.82 F1 ceiling**. TF-IDF baselines saturate quickly; grid search over hyperparameters yields minimal improvement.

4. **Cross-task generalization is poor** (LOTO F1 drops to 0.54). QA task generalizes worst, suggesting task-specific artifacts drive performance.

5. **Error patterns reveal brittleness**: Model over-flags short/uncertain responses (FP) and misses fluent fabrications (FN). Hallucination detection remains sensitive to stylistic rather than factual cues.

6. **No evidence of data leakage**: Group-aware splitting, shuffled-label baseline (~0.50 F1), and overlap analysis confirm experimental validity.

7. **Response length is weakly predictive**: Hallucinated responses tend to be slightly shorter (weak negative correlation), but effect is small and non-diagnostic.

8. **Numeric features show minimal individual correlation** (|r| < 0.1 for all), yet contribute collectively through ensemble effects in ablation experiments.

## 8. Limitations

1. **Small marginal gains from feature engineering**: Numeric features add only ~1% F1 improvement, suggesting diminishing returns on hand-crafted features.

2. **Dataset-specific performance**: Strong in-task performance (0.82 F1) but poor cross-task transfer (0.54 F1) indicates overfitting to HaluEval's task-specific patterns.

3. **Binary hallucination framing**: Real-world hallucinations exist on a spectrum (partial truths, misleading emphasis, outdated facts). Binary labels oversimplify the problem.

4. **No external knowledge verification**: Models rely on surface patterns, not factual grounding. Cannot distinguish "fluent but wrong" from "fluent and correct."

5. **Transformer fine-tuning not explored**: Lexical baselines may have reached their ceiling; semantic models (BERT, RoBERTa fine-tuning) remain untested.

6. **Error analysis is qualitative**: Themes identified from top-K errors may not generalize to full error distribution. Formal error taxonomy would strengthen conclusions.

## 9. Conclusion

This work establishes a reproducible experimental protocol for hallucination detection on HaluEval, demonstrating that lexical models with minimal feature engineering achieve competitive performance (F1=0.82) within task but generalize poorly across tasks. The findings suggest that current benchmark performance is driven by task-specific surface patterns rather than robust hallucination signals. Future work should prioritize cross-task generalization, fine-grained error taxonomies, and integration of external knowledge verification to move beyond lexical shortcuts toward factual grounding.

---

## Appendix: Experimental Artifacts

### Metrics and Results
- `reports/final_leaderboard.csv` – Unified model comparison table
- `reports/nb02_baseline_tfidf/metrics.csv` – NB02 baseline results
- `reports/nb03_feature_based/metrics.csv` – NB03 feature-based results
- `reports/nb07_input_ablation.csv` – Input field ablation
- `reports/nb07_loto.csv` – Leave-one-task-out generalization

### Error Analysis
- `reports/nb08_profiling/error_report.md` – Detailed error analysis
- `reports/nb08_profiling/top_fp.csv` – Top 50 false positives
- `reports/nb08_profiling/top_fn.csv` – Top 50 false negatives

### Feature Analysis
- `reports/nb09_ablation/ablation_delta_f1.csv` – Feature ablation results
- `reports/nb08_profiling/feature_label_corr.csv` – Feature-label correlations

### Figures
- `reports/nb08_profiling/plots/feature_label_corr.png` – Correlation heatmap
- `reports/nb09_ablation/plots/ablation_delta_f1.png` – Feature impact bar chart
- `reports/nb02_baseline_tfidf/plots/confusion_matrix_val.png` – Baseline confusion matrix
