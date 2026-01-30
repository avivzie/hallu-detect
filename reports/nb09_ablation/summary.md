# Feature Ablation Summary

## Baseline Performance
- Val F1: 0.8205
- Val ROC-AUC: 0.9144
- Test F1: 0.8212
- Test ROC-AUC: 0.9088

## Top 5 Feature Contributors

1. **resp_numbers_per_word**: ΔF1=-0.0029 (helps)
2. **resp_has_ellipsis**: ΔF1=0.0021 (harms)
3. **resp_n_uncertainty**: ΔF1=0.0015 (harms)
4. **resp_punct_per_word**: ΔF1=-0.0015 (helps)
5. **resp_n_punct**: ΔF1=-0.0015 (helps)

## Interpretation

### Triangulation with NB08 Findings
- *[Compare ablation results with NB08 correlations]*
- *[Check if high-impact features align with correlation strengths]*

### Error Theme Alignment
- *[Review NB08 top_fp/top_fn to see if errors relate to high-impact features]*
- *[E.g., do FP examples have unusual numbers/lengths/uncertainty patterns?]*

### Key Insights
- *[State which features actually matter for hallucination detection]*
- *[Note any surprising findings or mismatches with correlation]*
