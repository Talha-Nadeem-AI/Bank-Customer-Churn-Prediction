"""Cleaning, feature engineering, train/test splitting and the preprocessing pipeline.

Design rule for avoiding data leakage: anything that *learns* from data
(imputation values, scaling means/standard deviations, one-hot categories) is
inside a scikit-learn ``Pipeline``. The pipeline is fitted on the training set
only, so no information from the test set reaches the model.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data_loader import DataValidationError, check_required_columns, find_invalid_rows

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the raw dataset and report every change made.

    Steps (each one is justified in the report):
      1. Drop identifier columns (RowNumber, CustomerId, Surname): no predictive
         value and Surname/CustomerId are personal data.
      2. Tidy text in categorical columns (strip spaces, consistent capitals).
      3. Drop rows with a missing target, since they cannot be used for training.
      4. Drop exact duplicate rows.
      5. Drop rows that break the value rules in ``config.VALUE_RANGES`` /
         ``config.VALID_CATEGORIES`` (impossible ages, unknown countries, ...).
      6. Store numeric columns with correct dtypes. Floats such as Balance keep
         their decimals (the original notebook cast them to int and lost them).

    Missing *feature* values are not filled here; the model pipeline imputes
    them using statistics from the training set only.

    Returns:
        (cleaned DataFrame, report dict of how many rows/columns each step removed)
    """
    check_required_columns(df, config.FEATURES + [config.TARGET])
    data = df.copy()
    report: Dict[str, int] = {"rows_in": len(data)}

    id_cols = [c for c in config.ID_COLUMNS if c in data.columns]
    data = data.drop(columns=id_cols)
    report["id_columns_dropped"] = len(id_cols)

    for col in config.CATEGORICAL_FEATURES:
        data[col] = data[col].where(data[col].isna(), data[col].astype(str).str.strip().str.title())

    before = len(data)
    data = data.dropna(subset=[config.TARGET])
    report["rows_missing_target"] = before - len(data)

    before = len(data)
    data = data.drop_duplicates()
    report["duplicate_rows"] = before - len(data)

    invalid = find_invalid_rows(data)
    bad_rows = invalid["row"].unique()
    data = data.drop(index=bad_rows)
    report["invalid_rows"] = len(bad_rows)

    for col in config.NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col])
    data[config.TARGET] = data[config.TARGET].astype(int)
    for col in config.CATEGORICAL_FEATURES:
        data[col] = data[col].astype(object)

    data = data.reset_index(drop=True)
    report["missing_feature_values"] = int(data[config.FEATURES].isna().sum().sum())
    report["rows_out"] = len(data)

    if data[config.TARGET].nunique() < 2:
        raise DataValidationError("After cleaning, the target has only one class; cannot train.")

    logger.info("Cleaning report: %s", report)
    return data, report


# --------------------------------------------------------------------------
# Feature engineering (a scikit-learn transformer so it lives inside the pipeline)
# --------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Optionally add engineered features and/or drop selected features.

    Engineered features (all computed row by row, so they cannot leak):
      * ``HasZeroBalance``   - 1 if Balance == 0 (36% of customers).
      * ``BalanceSalaryRatio`` - Balance / EstimatedSalary.
      * ``TenureByAge``      - Tenure / Age (share of life spent as a customer).
      * ``IsSingleProduct``  - 1 if the customer holds exactly one product.

    Args:
        add_features: Whether to add the engineered features.
        drop_features: Column names to remove (used for feature selection).
    """

    def __init__(self, add_features: bool = False, drop_features: Optional[Sequence[str]] = None):
        self.add_features = add_features
        self.drop_features = drop_features

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        """Nothing to learn; present for scikit-learn compatibility."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with features added/dropped."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame.")
        out = X.copy()
        if self.add_features:
            salary = out["EstimatedSalary"].clip(lower=1.0)  # avoid division by zero
            age = out["Age"].clip(lower=1.0)
            out["HasZeroBalance"] = (out["Balance"] == 0).astype(float)
            out["BalanceSalaryRatio"] = out["Balance"] / salary
            out["TenureByAge"] = out["Tenure"] / age
            out["IsSingleProduct"] = (out["NumOfProducts"] == 1).astype(float)
        if self.drop_features:
            out = out.drop(columns=[c for c in self.drop_features if c in out.columns])
        return out


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing step used by every model.

    * Numeric columns: median imputation then standard scaling.
    * Categorical columns: most-frequent imputation then one-hot encoding.
      ``handle_unknown='ignore'`` means an unseen category at prediction time
      becomes all zeros instead of crashing.

    Columns are picked by dtype, so the same preprocessor works whether or not
    engineered features were added.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
        ("cat", categorical_pipe, make_column_selector(dtype_exclude=np.number)),
    ])


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the target y."""
    check_required_columns(df, config.FEATURES + [config.TARGET])
    return df[config.FEATURES].copy(), df[config.TARGET].astype(int).copy()


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Stratifying keeps the ~20% churn rate identical in both sets, which matters
    because the classes are imbalanced. The test set is put aside and used only
    once, for the final evaluation.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if len(X) != len(y):
        raise ValueError("X and y have different numbers of rows.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_processed_data(df: pd.DataFrame, path=config.PROCESSED_DATA_PATH) -> None:
    """Save the cleaned, anonymised dataset (no identifier columns)."""
    leaked = [c for c in config.ID_COLUMNS if c in df.columns]
    if leaked:
        raise DataValidationError(f"Refusing to save data containing identifiers: {leaked}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s", path)
