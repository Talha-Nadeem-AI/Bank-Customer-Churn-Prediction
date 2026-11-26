# Bank Customer Churn Prediction

**Programming for AI: Major Team Project, CUSIT Fall 2026**

This project predicts which bank customers are likely to leave (churn), so that a retention team can contact them first. The pipeline is Python and scikit-learn: data audit, cleaning, EDA, 5 documented experiments (baseline → improved model), fairness and error analysis, and a working prediction demo.

> **Live demo:** [bank-customer-churn-prediction-vycaupp7x4mh5jfxozrhmt.streamlit.app](https://bank-customer-churn-prediction-vycaupp7x4mh5jfxozrhmt.streamlit.app): the trained model is included, so it opens straight on the dashboard.

> **New to the project?** Read **[GUIDE.md](GUIDE.md)**. It explains what churn is, how to install and run everything, and how to test each feature, with expected results and a review form.

> ⚠️ This is a classroom model. It is **not** ready for deployment and must not be used for high-stakes decisions about real customers.

## Results (held-out test set, 2,000 customers)

| | Baseline: Logistic Regression | **Final: tuned Random Forest (threshold 0.45)** |
|---|---|---|
| Accuracy | 0.808 | 0.832 |
| Precision | 0.589 | 0.572 |
| **Recall** | 0.187 | **0.681** |
| F1 | 0.284 | 0.622 |
| ROC-AUC | 0.775 | 0.859 |
| Churners missed | 331 of 407 | 130 of 407 |

All three success criteria are met (recall ≥ 0.65, precision ≥ 0.50, ROC-AUC ≥ 0.85). The full numbers are in `outputs/results/run_summary.md`, which is regenerated on every run.

## Project structure
```
churn-prediction/
├── main.py                     # runs the whole workflow end to end
├── app.py                      # ChurnGuard Streamlit app (6 pages)
├── .streamlit/config.toml      # app theme colours
├── requirements.txt
├── GUIDE.md                    # step-by-step guide for testers and reviewers
├── src/churn/
│   ├── config.py               # paths, seed, columns, validation rules, success criteria
│   ├── data_loader.py          # load CSV + input validation (custom DataValidationError)
│   ├── preprocessing.py        # cleaning, feature engineering, leak-free pipeline, split
│   ├── eda.py                  # dataset audit, data dictionary, 7 Matplotlib figures
│   ├── models.py               # LR, Random Forest, Gradient Boosting, MLP (ANN) + grids
│   ├── experiments.py          # experiments E1-E5 with cross-validation
│   ├── evaluate.py             # metrics, threshold tuning, charts, fairness, error analysis
│   ├── predict.py              # save/load model, score new customers (CLI demo)
│   └── ui/                     # app: theme.py, components.py, data.py, sidebar.py, pages.py
├── notebooks/churn_prediction.ipynb   # step-by-step walkthrough with outputs
├── tests/                      # 45 pytest tests (edge cases + failure conditions)
├── data/
│   ├── raw/                    # Churn_Modelling.csv (not committed, see below)
│   ├── processed/              # churn_cleaned.csv (identifiers removed)
│   └── sample/                 # new_customers.csv (10 unseen test customers), invalid_customers.csv (broken on purpose)
├── models/                     # final_model.joblib + metadata (generated)
├── outputs/
│   ├── figures/                # 15 PNG charts
│   └── results/                # audit, EDA findings, experiment table, fairness, summary
└── docs/testing_log.md         # tests run + errors found and fixed
```

## Setup and run
```bash
# 1. Create an environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Put the dataset at data/raw/Churn_Modelling.csv

# 3. Run everything (about 5 minutes; --quick takes about 1 minute)
python main.py

# 4. Run the tests
python -m pytest -v

# 5. Live demo app (opens in your browser at http://localhost:8501)
streamlit run app.py

# 5b. Command-line alternative: score a CSV with the saved model
python -m src.churn.predict --input data/sample/new_customers.csv

# 6. Optional: open the notebook
jupyter notebook notebooks/churn_prediction.ipynb
```
All randomness uses `RANDOM_STATE = 42`, so repeated runs on the same machine give identical results.

> **Library versions matter.** The numbers in this README come from Python 3.13 + scikit-learn 1.9. With other versions (e.g. Python 3.9 + scikit-learn 1.6) results differ slightly and the pipeline may even pick a different final model. Always run `python main.py` on your own machine before presenting, and quote the numbers from *your* `outputs/results/run_summary.md`. The app reads them automatically. The included `models/final_model.joblib` only loads with the same scikit-learn version, so retrain it first.

## ChurnGuard demo app (`streamlit run app.py`)
A 6-page Streamlit app with a custom design: evergreen sidebar, Bricolage Grotesque + Nunito fonts, muted teal and brass palette. Every number on every page is read from the dataset, the saved model and the results of your latest `python main.py` run.

| Sidebar group | Page | What it shows |
|---|---|---|
| Analyse | **Overview** | Key figures, churn by products / country / age band, test-set outcomes, items that need attention |
| Analyse | **Data insights** | Dataset audit, class balance, correlations, what drives the model, cleaning decisions, all EDA charts |
| Predict | **Predict a customer** | Input form → probability gauge, risk level, suggested action, what-if scenarios |
| Predict | **Batch scoring** | Upload a CSV (or use 10 sample customers) → ranked list, download, validation errors explained |
| Govern | **Model performance** | Baseline vs final confusion matrices, metric comparison, full experiment log, pipeline charts |
| Govern | **Fairness & use** | Performance by gender and country, responsible-use guidance, limitations |

The sidebar also shows the model status (recall ring), portfolio churn, risk-level legend and dataset details.
Risk levels: **High** = at or above the model threshold (flagged as churn), **Medium** = 0.30 up to the threshold, **Low** = below 0.30.
Fonts load from Google Fonts; offline, the app falls back to the system font.

## Deploying online (Streamlit Community Cloud)
1. Push the project to a GitHub repository (the `.gitignore` keeps out the raw CSV, because it contains surnames, and the trained model).
2. Run the app locally and click **Deploy** (top right) → Streamlit Community Cloud, or go to share.streamlit.io. Pick the repo, branch `main` and file `app.py`.
3. The trained model (`models/final_model.joblib`) is committed, so the app opens straight on the dashboard with the same results as the report. Only if the host installs a different scikit-learn version does the app show **Set up ChurnGuard**; then click **Train the model now** and it retrains on the anonymised `data/processed/churn_cleaned.csv` in about 2–3 minutes, using one CPU core so it stays within the free tier's memory limit.

## Dataset
- **Name:** Churn Modelling (bank customer churn), 10,000 customers × 14 columns. Target `Exited` (20.4% churn).
- **Source:** Kaggle. *TODO (team): add the exact URL, uploader, licence and the date you accessed it.*
- **Privacy:** `Surname` and `CustomerId` are removed during cleaning. The raw file is excluded from Git via `.gitignore`.
- The full audit and data dictionary are in `outputs/results/dataset_audit.md`.

## Method summary
1. **Cleaning:** drop identifiers, tidy text, remove missing-target/duplicate/invalid rows (none found in this dataset), keep decimal values.
2. **EDA:** 7 Matplotlib charts, each with a computed interpretation (`outputs/results/eda_findings.md`).
3. **Split:** stratified 80/20. Preprocessing (imputation, scaling, one-hot) sits inside a scikit-learn `Pipeline`, so it is fitted on training data only.
4. **Experiments** (choices are made with 5-fold CV on the training set, never on the test set):

| Run | Change | CV F1 |
|---|---|---|
| E1 | Baseline logistic regression | 0.315 |
| E2 | + balanced class weights | 0.489 |
| E3a/b/c | Random Forest / Gradient Boosting / MLP (ANN) | 0.604 / 0.602 / 0.561 |
| E4a/b/c | + engineered features / − weak features / both | 0.605 / 0.599 / 0.602 |
| E5 | Grid search (36 combinations) + threshold tuned for recall (precision ≥ 0.55) | 0.615* |

\*CV F1 at threshold 0.5.

5. **Metric choice:** a missed churner costs the bank a customer, while a false alarm costs only a phone call, so recall is prioritised and precision is kept above a floor. Accuracy alone is misleading: predicting "stays" for everyone already scores about 80%.

## Responsible use
- **Supports:** prioritising which customers the retention team contacts.
- **Must not be used for:** denying services, pricing, credit or any automated decision about a customer.
- **Known issues:** the model flags German customers and female customers as churners more often when they actually stay (see `outputs/results/fairness_report.csv`). Gender and Geography are used as inputs. The data has no date range or named bank, and covers only 3 countries.
- **Before real use:** validate on the bank's own recent data, review fairness with compliance, monitor drift, and A/B-test the retention actions.

## Team
| Member | Student ID | Role | Main responsibilities | Git part |
|---|---|---|---|---|
| Talha Nadeem | 16131 | Project Lead & Data Engineer | Problem and stakeholder research, proposal, coordination and meeting log; dataset comparison and audit, data dictionary, data loading, cleaning, EDA; repository owner | Part 1: setup, data loading, cleaning, EDA (4 commits) |
| Syed Abdullah Bin Masood | 16048 | ML Developer | Baseline and improved models, the 5 experiments, evaluation metrics, threshold tuning, fairness and error analysis, prediction CLI, notebook | Part 2: models, evaluation, experiments, prediction, results (5 commits) |
| Muhammad Moaz | 15966 | Git/QA, App & Documentation Lead | ChurnGuard app, testing and testing log, README and tester guide, report integration, slides and demo | Part 3: app, UI tests, documentation (4 commits) |

Full task and evidence breakdown: [`docs/contribution_matrix.md`](docs/contribution_matrix.md). Meeting record: [`docs/project_log.md`](docs/project_log.md).

AI-tool assistance: *TODO (team): disclose according to the course instructions.*
