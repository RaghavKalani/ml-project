# ML Model Report: Stroke Prediction

## Method
- The target is `stroke`; the identifier column was excluded.
- Missing BMI values are median-imputed inside the pipeline.
- Categorical variables are one-hot encoded inside the pipeline.
- The data uses an 80/20 stratified train/test split with `random_state=42`.
- Logistic Regression and Random Forest use balanced class weights because stroke is rare.

## Test-set Results
| model               |   precision |   recall |       f1 |   roc_auc |   pr_auc |
|:--------------------|------------:|---------:|---------:|----------:|---------:|
| logistic_regression |    0.138408 |     0.8  | 0.235988 |  0.843663 | 0.268266 |
| random_forest       |    0.174603 |     0.44 | 0.25     |  0.817119 | 0.192519 |

The selected baseline is **Logistic Regression**, based on the highest test PR-AUC (0.2683).

## Threshold Analysis
|   threshold |   precision |   recall |       f1 |
|------------:|------------:|---------:|---------:|
|         0.2 |   0.0828516 |     0.86 | 0.151142 |
|         0.3 |   0.0967742 |     0.84 | 0.173554 |
|         0.4 |   0.11898   |     0.84 | 0.208437 |
|         0.5 |   0.138408  |     0.8  | 0.235988 |
|         0.6 |   0.171053  |     0.78 | 0.280576 |

A threshold below 0.50 can improve recall, but it also increases false positives. The final threshold should be chosen according to the application's clinical or operational cost of missed cases versus unnecessary alerts.

## Validation Notes
- PR-AUC and recall are emphasized because the target is highly imbalanced.
- These results are an educational modeling baseline, not a clinical diagnostic system.
