# NB07 — Input Ablation + Leave-One-Task-Out (LOTO)

## Input ablation (Top-3 by Val F1)

| setting               |   val_f1 |   val_accuracy |   val_roc_auc |   test_f1 |   test_accuracy |   test_roc_auc |
|:----------------------|---------:|---------------:|--------------:|----------:|----------------:|---------------:|
| response_only         | 0.813376 |       0.828016 |      0.902781 |  0.811415 |        0.824398 |       0.900248 |
| response_plus_context | 0.743764 |       0.753774 |      0.843958 |  0.73587  |        0.746542 |       0.838257 |
| prompt_plus_response  | 0.669628 |       0.687471 |      0.759952 |  0.66241  |        0.679565 |       0.754226 |

## LOTO summary (response_only)

| heldout_task   |   test_f1 |   test_accuracy |   test_roc_auc |   test_precision |   test_recall |   test_size |
|:---------------|----------:|----------------:|---------------:|-----------------:|--------------:|------------:|
| dialogue       |  0.627623 |        0.626347 |       0.67715  |         0.625486 |      0.629775 |        2042 |
| summarization  |  0.614925 |        0.535494 |       0.548639 |         0.525127 |      0.74177  |        1944 |
| qa             |  0.384674 |        0.594854 |       0.789792 |         0.799363 |      0.25328  |        1982 |

## Key takeaways

- Response-only is the best input setting; adding prompt/context hurts performance.

- Cross-task generalization drops in LOTO, worst when QA is held out (high precision, low recall).

- This suggests task-specific artifacts; motivates hardened baseline / per-task analysis.
