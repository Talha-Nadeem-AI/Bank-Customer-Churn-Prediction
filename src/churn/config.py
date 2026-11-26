"""Central configuration for the bank customer churn project.

Every path, column list, random seed and validation rule lives here so that
the rest of the code never hard-codes them. Change a value here and the whole
pipeline picks it up.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root, so the project runs on any machine)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Churn_Modelling.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_MODEL_META_PATH = MODELS_DIR / "final_model_metadata.json"

# --------------------------------------------------------------------------
# Reproducibility and evaluation settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42          # one seed used everywhere
TEST_SIZE = 0.20           # 80% train / 20% held-out test
CV_FOLDS = 5               # stratified k-fold cross-validation on training data
DEFAULT_THRESHOLD = 0.50   # probability cut-off before threshold tuning

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
TARGET = "Exited"

# Identifiers: no predictive meaning, and Surname/CustomerId identify people.
# They are removed during cleaning for privacy and to avoid spurious patterns.
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features that EDA shows have almost no relationship with churn
# (used in the feature-selection experiment).
WEAK_FEATURES = ["HasCrCard", "Tenure", "EstimatedSalary"]

# Sensitive attributes checked in the fairness analysis.
SENSITIVE_FEATURES = ["Gender", "Geography"]

# --------------------------------------------------------------------------
# Validation rules (used for input checks on training and prediction data)
# --------------------------------------------------------------------------
VALID_CATEGORIES = {
    "Geography": {"France", "Germany", "Spain"},
    "Gender": {"Male", "Female"},
}

# (min, max) inclusive; None means "no limit on that side".
VALUE_RANGES = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 50),
    "Balance": (0, None),
    "NumOfProducts": (1, 10),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0, None),
}

# Plain-language description of every column (used for the data dictionary).
DATA_DICTIONARY = {
    "RowNumber": "Row index in the original file (identifier, removed).",
    "CustomerId": "Bank customer ID (identifier, removed for privacy).",
    "Surname": "Customer surname (personal data, removed for privacy).",
    "CreditScore": "Customer credit score.",
    "Geography": "Country of the customer (France, Germany, Spain).",
    "Gender": "Customer gender as recorded by the bank (Male/Female).",
    "Age": "Customer age in years.",
    "Tenure": "Number of years the customer has been with the bank.",
    "Balance": "Account balance (currency not stated in the source).",
    "NumOfProducts": "Number of bank products the customer uses.",
    "HasCrCard": "1 if the customer has a credit card, else 0.",
    "IsActiveMember": "1 if the bank classes the customer as active, else 0.",
    "EstimatedSalary": "Estimated annual salary (currency not stated).",
    "Exited": "TARGET: 1 if the customer left the bank, 0 if they stayed.",
}

# Measurable success criteria (Requirement 1). Checked automatically on the test set.
SUCCESS_CRITERIA = [
    {"text": "Catch at least 65% of customers who actually churn (recall)", "metric": "recall", "target": 0.65},
    {"text": "At least half of the customers flagged really churn (precision)", "metric": "precision", "target": 0.50},
    {"text": "Rank churners above non-churners well (ROC-AUC)", "metric": "roc_auc", "target": 0.85},
]

# Threshold tuning rule: maximise recall (catch churners) while keeping precision on
# out-of-fold TRAINING predictions at or above this floor. The floor is set above the
# 0.50 success criterion to leave a safety margin for unseen data.
THRESHOLD_MIN_PRECISION = 0.55

# Risk bands shown to a non-technical stakeholder:
#   Low    : probability < LOW_RISK_CUTOFF
#   Medium : LOW_RISK_CUTOFF <= probability < model threshold   (watch list)
#   High   : probability >= model threshold                     (flagged as churn)
# Tying "High" to the model's decision threshold means "High" always equals "flagged".
LOW_RISK_CUTOFF = 0.30


# Project team (shown in the app sidebar and used in the documentation).
TEAM = [
    {"name": "Talha Nadeem", "id": "16131", "role": "Project Lead & Data Engineer", "git_part": 1},
    {"name": "Syed Abdullah Bin Masood", "id": "16048", "role": "ML Developer", "git_part": 2},
    {"name": "Muhammad Moaz", "id": "15966", "role": "Git/QA, App & Documentation Lead", "git_part": 3},
]


def default_data_path() -> Path:
    """Dataset to use: the raw Kaggle file if present, otherwise the anonymised cleaned copy.

    The raw file contains surnames, so it is kept out of Git. A deployed copy of the
    app (e.g. Streamlit Community Cloud) therefore trains on data/processed/churn_cleaned.csv,
    which has the same rows and columns minus the identifiers.
    """
    return RAW_DATA_PATH if RAW_DATA_PATH.exists() else PROCESSED_DATA_PATH


def ensure_output_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for folder in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR, PROCESSED_DATA_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)
