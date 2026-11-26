"""Loading and validating the churn dataset.

The functions here are the single entry point for reading data. They check
that the file exists, is a non-empty CSV, has the expected columns and that the
values make sense, and they raise clear errors when something is wrong instead
of letting a bad file fail later in an obscure way.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset does not meet the expected schema or value rules."""


def load_raw_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Read the raw churn CSV and check its structure.

    Args:
        path: Location of the CSV. Defaults to ``config.default_data_path()`` (raw file,
            or the anonymised cleaned copy when the raw file is not available).

    Returns:
        The raw data as a DataFrame.

    Raises:
        FileNotFoundError: The file does not exist.
        DataValidationError: The file is not a CSV, is empty, or is missing
            required columns.
    """
    path = Path(path) if path is not None else config.default_data_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download Churn_Modelling.csv and "
            f"place it in data/raw/ (see README)."
        )
    if path.suffix.lower() != ".csv":
        raise DataValidationError(f"Expected a .csv file, got '{path.suffix}'.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"The file '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Could not parse '{path}' as CSV: {exc}") from exc

    if df.empty:
        raise DataValidationError(f"The file '{path}' has a header but no rows.")

    check_required_columns(df, config.FEATURES + [config.TARGET])
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path.name)
    return df


def check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ``DataValidationError`` if any required column is missing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {missing}")


def find_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table describing every rule violation found in ``df``.

    Checks that numeric columns are numeric and inside ``config.VALUE_RANGES``,
    and that categorical columns only contain ``config.VALID_CATEGORIES``.
    Missing values are *not* reported here (they are handled by imputation).

    Returns:
        DataFrame with columns ``row``, ``column``, ``value``, ``problem``.
        Empty if the data is valid.
    """
    problems = []

    for col, (low, high) in config.VALUE_RANGES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        not_numeric = numeric.isna() & df[col].notna()
        for idx in df.index[not_numeric]:
            problems.append((idx, col, df.at[idx, col], "not a number"))
        if low is not None:
            for idx in df.index[numeric < low]:
                problems.append((idx, col, df.at[idx, col], f"below minimum {low}"))
        if high is not None:
            for idx in df.index[numeric > high]:
                problems.append((idx, col, df.at[idx, col], f"above maximum {high}"))

    for col, allowed in config.VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        values = df[col]
        bad = values.notna() & ~values.isin(allowed)
        for idx in df.index[bad]:
            problems.append((idx, col, df.at[idx, col], f"not one of {sorted(allowed)}"))

    return pd.DataFrame(problems, columns=["row", "column", "value", "problem"])


def validate_features(df: pd.DataFrame, allow_missing: bool = True) -> pd.DataFrame:
    """Strictly validate feature data before it is given to a model.

    Used on new customers at prediction time. Any rule violation raises an
    error so that the model never silently scores nonsense input.

    Args:
        df: Customer records containing at least ``config.FEATURES``.
        allow_missing: If False, any missing feature value raises an error.
            If True, missing values are allowed and later imputed (a warning
            is logged).

    Returns:
        A copy of ``df`` restricted to ``config.FEATURES`` with numeric
        columns converted to numbers.

    Raises:
        DataValidationError: Missing columns, empty input, invalid values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise DataValidationError("No customer records were provided.")
    check_required_columns(df, config.FEATURES)

    features = df[config.FEATURES].copy()
    for col in config.CATEGORICAL_FEATURES:
        features[col] = features[col].where(features[col].isna(), features[col].astype(str).str.strip())

    issues = find_invalid_rows(features)
    if not issues.empty:
        preview = issues.head(5).to_string(index=False)
        raise DataValidationError(
            f"{len(issues)} invalid value(s) found in the input:\n{preview}"
        )

    n_missing = int(features.isna().sum().sum())
    if n_missing:
        if not allow_missing:
            raise DataValidationError(f"Input contains {n_missing} missing value(s).")
        logger.warning("%d missing value(s) will be imputed by the model pipeline.", n_missing)

    for col in config.NUMERIC_FEATURES:
        features[col] = pd.to_numeric(features[col]).astype(float)
    # Keep categoricals as plain object dtype so scikit-learn treats them consistently.
    for col in config.CATEGORICAL_FEATURES:
        features[col] = features[col].astype(object).where(features[col].notna(), np.nan)
    return features
