"""Shared test fixtures. Tests use a slice of the real dataset (never invented data)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.churn import config  # noqa: E402
from src.churn.preprocessing import clean_data, split_features_target  # noqa: E402


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("Raw dataset not found in data/raw/")
    return pd.read_csv(config.RAW_DATA_PATH)


@pytest.fixture(scope="session")
def small_raw(raw_df) -> pd.DataFrame:
    """First 600 real rows - enough to train quickly in tests."""
    return raw_df.head(600).copy()


@pytest.fixture(scope="session")
def trained_pipeline(small_raw):
    """A small logistic-regression pipeline trained on real rows."""
    from src.churn.models import build_pipeline  # imported here so data-only tests run without the model code
    clean, _ = clean_data(small_raw)
    X, y = split_features_target(clean)
    return build_pipeline("logistic_regression", class_weight="balanced").fit(X, y)


@pytest.fixture
def one_customer(small_raw) -> pd.DataFrame:
    """A single real customer's feature row."""
    return small_raw[config.FEATURES].iloc[[0]].reset_index(drop=True)
