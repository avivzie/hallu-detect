# Error Analysis Report

## Summary Statistics

- **Total errors**: 100
- **False positives (FP)**: 50 (predicted hallucination, true non-hallucination)
- **False negatives (FN)**: 50 (predicted non-hallucination, true hallucination)

## Top 10 False Positives (by confidence)

Model incorrectly predicted hallucination with high confidence.

| id                    | task          |    score |   confidence |   y_true |   y_pred |
|:----------------------|:--------------|---------:|-------------:|---------:|---------:|
| qa_111_gt             | qa            | 0.999874 |     0.999874 |        0 |        1 |
| general_752           | general       | 0.976649 |     0.976649 |        0 |        1 |
| dialogue_6611_gt      | dialogue      | 0.964834 |     0.964834 |        0 |        1 |
| dialogue_8486_gt      | dialogue      | 0.957511 |     0.957511 |        0 |        1 |
| dialogue_2729_gt      | dialogue      | 0.94042  |     0.94042  |        0 |        1 |
| summarization_3531_gt | summarization | 0.936488 |     0.936488 |        0 |        1 |
| dialogue_173_gt       | dialogue      | 0.934617 |     0.934617 |        0 |        1 |
| summarization_5244_gt | summarization | 0.934251 |     0.934251 |        0 |        1 |
| qa_533_gt             | qa            | 0.932848 |     0.932848 |        0 |        1 |
| dialogue_3719_gt      | dialogue      | 0.932369 |     0.932369 |        0 |        1 |

## Top 10 False Negatives (by confidence)

Model incorrectly predicted non-hallucination with high confidence.

| id                      | task          |     score |   confidence |   y_true |   y_pred |
|:------------------------|:--------------|----------:|-------------:|---------:|---------:|
| dialogue_2925_hall      | dialogue      | 0.0259748 |     0.974025 |        1 |        0 |
| qa_8385_hall            | qa            | 0.0383026 |     0.961697 |        1 |        0 |
| qa_6472_hall            | qa            | 0.0485799 |     0.95142  |        1 |        0 |
| dialogue_7817_hall      | dialogue      | 0.0551808 |     0.944819 |        1 |        0 |
| summarization_8822_hall | summarization | 0.0552493 |     0.944751 |        1 |        0 |
| summarization_1463_hall | summarization | 0.0553701 |     0.94463  |        1 |        0 |
| dialogue_3726_hall      | dialogue      | 0.0613309 |     0.938669 |        1 |        0 |
| summarization_6514_hall | summarization | 0.0643298 |     0.93567  |        1 |        0 |
| dialogue_2837_hall      | dialogue      | 0.0645942 |     0.935406 |        1 |        0 |
| dialogue_6062_hall      | dialogue      | 0.0718214 |     0.928179 |        1 |        0 |

## Error Themes (Manual Analysis)

### Numbers and Facts
- *[Examine whether errors involve numerical values, dates, or factual claims]*

### Named Entities
- *[Check if errors involve person names, locations, organizations]*

### Causality and Reasoning
- *[Look for logical inconsistencies or causal reasoning errors]*

### Fluent Nonsense
- *[Identify grammatically correct but semantically meaningless responses]*

### Context Misalignment
- *[Find cases where response doesn't match prompt context]*

### Other Patterns
- *[Note any other recurring patterns in errors]*
