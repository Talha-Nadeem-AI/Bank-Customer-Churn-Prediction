# Testing and Error Log

Requirement 9 asks for testing on unseen data, at least three edge cases or failure conditions, and a record of the errors found and how they were fixed.

## 1. Testing on unseen data
- 20% of the data (2,000 customers) is held out with a stratified split and is never used for training, model selection or threshold tuning.
- All model choices use 5-fold cross-validation on the training set. The test set is scored once for the final report (`outputs/results/run_summary.md`).
- `tests/test_models_and_predict.py::test_saved_final_model_scores_unseen_test_customers` loads the saved model and scores 10 real test-set customers.

## 2. Automated tests (`python -m pytest`), 47 tests
| Area | Edge case / failure condition | Expected behaviour |
|---|---|---|
| Loading | File does not exist | `FileNotFoundError` with a hint to download the data |
| Loading | Empty file / header-only file | `DataValidationError` |
| Loading | Wrong extension (.xlsx) | `DataValidationError` |
| Loading | Required column missing | `DataValidationError` naming the column |
| Validation | Age = -5, Geography = "Mars", Age = "forty" | All three reported as invalid |
| Cleaning | Duplicates, out-of-range age, missing target | Rows removed and counted in the report |
| Cleaning | "  france " (messy text) | Standardised to "France" |
| Cleaning | Only one class left | Clear error (cannot train) |
| Cleaning | Saving data that still has Surname/CustomerId | Refused (privacy) |
| Features | EstimatedSalary = 0 | No division by zero; finite ratio |
| Split | Train/test overlap | None; churn rate equal within 1 point |
| Prediction | Single customer | Works; returns probability and risk band |
| Prediction | Missing value (Balance = NaN) | Imputed by the pipeline; valid probability |
| Prediction | Unknown country "Italy" | `DataValidationError` |
| Prediction | NumOfProducts = 0 | `DataValidationError` |
| Prediction | Column missing | `DataValidationError` |
| Prediction | Extreme but valid values (Age 92, CreditScore 350) | Finite probability |
| Metrics | Mismatched lengths, probability > 1, empty input, threshold 0 | `ValueError` |
| Metrics | Hand-calculated confusion matrix | Matches exactly |
| Prediction | Risk band vs flag | "High" risk always equals "flagged as churn" |
| Demo app | Invalid customer (Italy, age 150) | Rejected with both problems listed |
| Demo app | Uploaded CSV containing `<script>` text | Shown as text, never run as HTML (all values escaped) |
| Demo app | CSV with no required columns | "File failed validation" panel listing the missing columns |
| Demo app | Overview page runs headlessly (Streamlit AppTest) | No exceptions |
| Model | Final model file missing | Helpful error telling the user to run `main.py` |

## 3. Errors found and fixed
| # | Where | Error | Fix |
|---|---|---|---|
| 1 | Original notebook | `pd.get_dummies(...).astype(int)` truncated decimals in Balance (63% of rows) and EstimatedSalary | Only the encoder creates integer columns; numeric columns keep their floats. Regression test `test_clean_keeps_decimal_balances` |
| 2 | Original notebook | `import matplotlib as plt` (wrong module), unused seaborn import | Use `matplotlib.pyplot`; seaborn removed (the brief requires Matplotlib) |
| 3 | Original notebook | Only accuracy reported; 0.857 looked good but "always predict stay" scores 0.80 on that split | Report confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC |
| 4 | Original notebook | Unstratified split; no TensorFlow seed, so results changed every run | Stratified split; one seed (`config.RANDOM_STATE`) for every step |
| 5 | Original notebook | Model built with Keras, but the brief requires scikit-learn models | Rebuilt the same ANN (3 x 11 ReLU) with `sklearn.neural_network.MLPClassifier` |
| 6 | New code, `evaluate.error_analysis` | `np.select` crashed: string choices with the default integer value (`TypeError: Choicelist and default value do not have a common dtype`) | Added `default="unknown"` |
| 7 | New code, threshold tuning | Tuning the threshold for F1 lowered recall (0.59 on test, below the 0.65 target) | Tune for recall subject to precision >= 0.55 on out-of-fold training predictions. The decision is based on the stakeholder's cost of missed churners, not on test scores |
| 8 | Test suite | `test_clean_drops_duplicates...` failed: the test modified the same rows it had duplicated, so they were no longer duplicates | Duplicated different rows (5 and 6) |
| 10 | Risk bands | A customer at probability 0.50 was flagged as churn (threshold 0.45) but shown as "Medium" risk, because the bands were fixed at 0.30/0.60 | "High" now starts exactly at the model threshold; test `test_high_risk_always_matches_flag` |
| 11 | Demo app | Streamlit 1.67 deprecated `use_container_width` | Switched to `width="stretch"`; requirements pin `streamlit>=1.50` |
| 12 | ChurnGuard UI | Sidebar labels hidden under the panel above: Streamlit gives each Markdown block a −16px bottom margin | CSS sets that margin to 0 in the sidebar |
| 13 | ChurnGuard UI | Card titles rendered at 36px: Streamlit's own Markdown heading styles overrode ours | Overrides scoped to `[data-testid="stMarkdownContainer"]` |
| 14 | ChurnGuard UI | Sliders, checkboxes and toggles used Streamlit's default red | `.streamlit/config.toml` sets the theme's primary colour to the app's teal |
| 15 | ChurnGuard UI | Python 3.9 rejects backslashes inside f-string expressions (used in two layout helpers) | Added a `cls` argument to `card()`; verified the whole project on Python 3.9 + Streamlit 1.50 |
| 16 | Demo app | The Deploy button was missing: the toolbar was hidden for a cleaner look | Toolbar restored (`toolbarMode = "auto"`) |
| 17 | Deployment | A deployed copy has neither the raw CSV (kept out of Git for privacy) nor a trained model, so the app could not start | Falls back to the anonymised `data/processed/churn_cleaned.csv`; a first-launch screen trains the model in one click. Tested on a copy with no raw data, model or results |
| 9 | EDA figure 3 | Matplotlib warning about categorical units for numeric-looking string labels | Plot by position with explicit tick labels |

> Team: add any further errors you find while working, with the date and the member who fixed them.
