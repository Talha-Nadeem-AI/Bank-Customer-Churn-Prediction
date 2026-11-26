# Team Contribution Matrix

**Project:** Bank Customer Churn Prediction (ChurnGuard) · Programming for AI, CUSIT, Fall 2026
**Team size:** 3 (the Git/QA role is combined with the app and documentation work, as the brief allows)

## Members and roles

| Member | Student ID | Role | Git part |
|---|---|---|---|
| Talha Nadeem | 16131 | Project Lead & Data Engineer | Part 1 (4 commits) |
| Syed Abdullah Bin Masood | 16048 | ML Developer | Part 2 (5 commits) |
| Muhammad Moaz | 15966 | Git/QA, App & Documentation Lead | Part 3 (4 commits) |

## Tasks, files and evidence

Status as of 26 November 2026. Update the last column as remaining work is finished.

### Talha Nadeem (16131) — Project Lead & Data Engineer

| Task | Files / evidence | Status |
|---|---|---|
| Stakeholder and problem definition, objectives, 2+ success criteria | `Project_Proposal.docx`; `src/churn/config.py` (`SUCCESS_CRITERIA`) | Done |
| Background research (5+ credible sources) | Proposal section 6; report section 3 (8 sources) | Done |
| Compare 2+ candidate datasets; choose and justify | `Dataset_Audit.docx`, section 2 | Done |
| Dataset audit and data dictionary | `outputs/results/dataset_audit.md`, `src/churn/eda.py` | Done |
| Validated data loading | `src/churn/data_loader.py`, `tests/test_data_loader.py` | Done |
| Cleaning, feature engineering, stratified split | `src/churn/preprocessing.py`, `tests/test_preprocessing.py` | Done |
| EDA: 7 Matplotlib charts with written findings | `outputs/figures/eda_*.png`, `outputs/results/eda_findings.md` | Done |
| Repository owner; add collaborators; meeting log | GitHub repo settings, `docs/project_log.md` | In progress: repo set up; collaborators and meeting log to complete |
| Commits | Git Part 1: project setup · data loading · cleaning · EDA | Committed; publish to GitHub |

### Syed Abdullah Bin Masood (16048) — ML Developer

| Task | Files / evidence | Status |
|---|---|---|
| Baseline and improved model pipelines (LR, RF, Gradient Boosting, MLP ANN) | `src/churn/models.py` | Done |
| Metrics, confusion matrices, threshold tuning, fairness, error analysis | `src/churn/evaluate.py` | Done |
| 5 documented experiments (9 runs) and the end-to-end pipeline | `src/churn/experiments.py`, `main.py`, `outputs/results/experiment_results.csv` | Done |
| Model saving and prediction CLI with edge-case tests | `src/churn/predict.py`, `tests/test_models_and_predict.py` | Done |
| Results, model charts and walkthrough notebook | `outputs/results/`, `outputs/figures/model_*.png`, `notebooks/churn_prediction.ipynb` | Done |
| Report sections: Methodology, Experiments, Results and discussion | `Technical_Report.docx`, sections 5, 7, 8 | Drafted; review before submission |
| Commits | Git Part 2: models · evaluation · experiments · prediction · results | Pending push |

### Muhammad Moaz (15966) — Git/QA, App & Documentation Lead

| Task | Files / evidence | Status |
|---|---|---|
| ChurnGuard Streamlit app (6 pages, theme, first-launch setup) | `app.py`, `src/churn/ui/`, `.streamlit/config.toml` | Done |
| Testing: edge cases, failure conditions, UI tests; errors found and fixed | `tests/test_ui.py`, `docs/testing_log.md` | Done |
| README, tester guide, requirements | `README.md`, `GUIDE.md`, `requirements.txt` | Done |
| Git workflow: issues/tasks, collaborator pushes, final history check | GitHub Issues and commit history | To do: open Issues; confirm 13 commits from 3 members |
| Report integration, slides (8–10) and demo | `Technical_Report.docx`, slides, Streamlit app | Report drafted; demo app ready; slides to do |
| Commits | Git Part 3: app theme · app pages · tests and testing log · documentation | Pending push |

## Report and presentation split

| Member | Report sections | Presentation (7–10 min, everyone speaks) |
|---|---|---|
| Talha Nadeem | Executive summary; Problem, stakeholder and objectives; Related work; Dataset, audit and privacy | Problem, stakeholder, dataset and data quality |
| Syed Abdullah Bin Masood | Methodology; Experiments; Results and discussion (error analysis, fairness) | Baseline vs improved model, metrics, experiments |
| Muhammad Moaz | System design and Git workflow; Testing; Conclusion and future work; References and appendices | Workflow, live demo, Git evidence, recommendation and limitations |

## Viva preparation

Every member must be able to explain the whole project, not only their own part. Suggested cross-checks: each member walks another through their files before the viva (Talha → Abdullah: data pipeline; Abdullah → Moaz: models and metrics; Moaz → Talha: app and tests).
