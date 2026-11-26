# Run Summary - Bank Customer Churn Prediction

Run time: 52 s | train rows: 8,000 | test rows: 2,000 | seed: 42
Rows after cleaning: 10,000 of 10,000

## Experiments (selection uses cv_f1; test columns are for reporting only)

| run_id | model | description | cv_f1 | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | Logistic Regression | Logistic regression, all original features, default settings, threshold 0.5 | 0.315 | 0.808 | 0.589 | 0.187 | 0.284 | 0.775 |
| E2 | Logistic Regression | Same as E1 but class_weight='balanced' so missed churners cost more | 0.489 | 0.714 | 0.387 | 0.700 | 0.499 | 0.777 |
| E3a | Random Forest | Random forest (300 trees), balanced class weights | 0.604 | 0.840 | 0.611 | 0.582 | 0.596 | 0.851 |
| E3b | Gradient Boosting | Histogram gradient boosting, balanced class weights | 0.602 | 0.824 | 0.561 | 0.624 | 0.591 | 0.843 |
| E3c | MLP Neural Network (ANN) | MLP neural network (3 x 11 ReLU, same as original Keras ANN), early stopping | 0.561 | 0.866 | 0.745 | 0.516 | 0.610 | 0.863 |
| E4a | Random Forest | Best E3 model + 4 engineered features | 0.605 | 0.844 | 0.614 | 0.622 | 0.618 | 0.851 |
| E4b | Random Forest | Best E3 model without weak features (HasCrCard, Tenure, EstimatedSalary) | 0.599 | 0.826 | 0.566 | 0.612 | 0.588 | 0.836 |
| E4c | Random Forest | Best E3 model + engineered features - weak features | 0.602 | 0.831 | 0.580 | 0.614 | 0.597 | 0.846 |
| E5 | Random Forest | Grid-searched hyper-parameters + tuned decision threshold | 0.609 | 0.832 | 0.572 | 0.681 | 0.622 | 0.859 |

## Decisions

- E2 vs E1: balancing class weights changed CV recall from 0.214 to 0.686 and CV F1 from 0.315 to 0.489.
- E3: best model family by CV F1 is Random Forest (CV F1 0.604).
- E4: best feature set by CV F1 is run E4a (engineered=True, dropped=none, CV F1 0.605).
- E5: grid search tried 36 combinations x 5 folds; best CV F1 0.615 with {'class_weight': 'balanced_subsample', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 5}.
- E5: threshold tuned on out-of-fold training predictions to maximise recall with precision >= 0.55 -> 0.45 (OOF F1 0.609, recall 0.675, precision 0.555).
- E5 vs E4a: tuning changed CV F1 (threshold 0.5) from 0.605 to 0.615; at the tuned threshold the OOF recall is 0.675. (E5's cv_* columns are out-of-fold scores at the tuned threshold.)
- FINAL MODEL: Random Forest (run E5), threshold 0.45. Selected on cross-validated training performance only; test set used once for the final report.

## Final model vs baseline (held-out test set)

| Metric | Baseline E1 (t=0.50) | Final E5 (t=0.50) | Final E5 (tuned t=0.45) |
| --- | --- | --- | --- |
| accuracy | 0.808 | 0.848 | 0.832 |
| precision | 0.589 | 0.625 | 0.572 |
| recall | 0.187 | 0.634 | 0.681 |
| f1 | 0.284 | 0.629 | 0.622 |
| roc_auc | 0.775 | 0.859 | 0.859 |
| pr_auc | 0.479 | 0.700 | 0.700 |
| missed churners (FN) | 331 | 149 | 130 |
| false alarms (FP) | 53 | 155 | 207 |

## Success criteria

| criterion | metric | target | achieved | met |
| --- | --- | --- | --- | --- |
| Catch at least 65% of customers who actually churn (recall) | recall | 0.65 | 0.681 | True |
| At least half of the customers flagged really churn (precision) | precision | 0.5 | 0.572 | True |
| Rank churners above non-churners well (ROC-AUC) | roc_auc | 0.85 | 0.859 | True |

## Top features (permutation importance)

| feature | importance | std |
| --- | --- | --- |
| NumOfProducts | 0.128 | 0.0113 |
| Age | 0.1145 | 0.0052 |
| Balance | 0.0458 | 0.0086 |
| Geography | 0.0257 | 0.0028 |
| IsActiveMember | 0.0243 | 0.0024 |
| Gender | 0.0076 | 0.002 |

## Fairness (test set, final model)

| attribute | group | n | actual_churn_rate | predicted_churn_rate | recall | precision | false_positive_rate | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gender | Female | 930 | 0.251 | 0.301 | 0.704 | 0.586 | 0.166 | 0.639 |
| Gender | Male | 1070 | 0.163 | 0.191 | 0.649 | 0.554 | 0.102 | 0.598 |
| Geography | France | 1021 | 0.171 | 0.181 | 0.583 | 0.551 | 0.098 | 0.567 |
| Geography | Germany | 504 | 0.3 | 0.423 | 0.815 | 0.577 | 0.255 | 0.676 |
| Geography | Spain | 475 | 0.171 | 0.181 | 0.642 | 0.605 | 0.086 | 0.623 |

## Error analysis (mean feature values by outcome)

| outcome | count | CreditScore | Age | Tenure | Balance | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | is_germany | is_female |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FN (missed churner) | 130 | 652.29 | 39.18 | 4.98 | 87449.2 | 1.41 | 0.68 | 0.42 | 94360.54 | 0.22 | 0.53 |
| FP (false alarm) | 207 | 643.98 | 44.72 | 5.14 | 91207.94 | 1.26 | 0.71 | 0.46 | 105945.3 | 0.43 | 0.56 |
| TN (correct stay) | 1386 | 650.45 | 36.11 | 5.08 | 70866.42 | 1.58 | 0.68 | 0.57 | 101546.62 | 0.19 | 0.42 |
| TP (caught churner) | 277 | 648.51 | 47.83 | 4.5 | 91479.91 | 1.56 | 0.73 | 0.31 | 101497.27 | 0.44 | 0.59 |