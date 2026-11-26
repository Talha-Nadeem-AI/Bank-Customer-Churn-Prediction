"""Tests for cleaning, feature engineering and splitting."""
import numpy as np
import pandas as pd
import pytest

from src.churn import config
from src.churn.data_loader import DataValidationError
from src.churn.preprocessing import (
    FeatureEngineer,
    clean_data,
    save_processed_data,
    split_features_target,
    split_train_test,
)


def test_clean_removes_identifiers_for_privacy(small_raw):
    clean, report = clean_data(small_raw)
    for col in config.ID_COLUMNS:
        assert col not in clean.columns
    assert report["id_columns_dropped"] == 3


def test_clean_keeps_decimal_balances(small_raw):
    """Regression test for the original notebook bug (.astype(int) truncated Balance)."""
    clean, _ = clean_data(small_raw)
    assert np.allclose(clean["Balance"].to_numpy(), small_raw["Balance"].to_numpy())
    assert (clean["Balance"] % 1 != 0).any()


def test_clean_drops_duplicates_invalid_rows_and_missing_target(small_raw):
    dirty = pd.concat([small_raw.head(20), small_raw.iloc[5:7]], ignore_index=True)  # rows 5,6 duplicated
    dirty.loc[0, "Age"] = 150                     # out of range
    dirty.loc[1, config.TARGET] = np.nan          # missing target
    clean, report = clean_data(dirty)
    assert report["duplicate_rows"] == 2
    assert report["invalid_rows"] == 1
    assert report["rows_missing_target"] == 1
    assert len(clean) == 22 - 2 - 1 - 1


def test_clean_standardises_category_text(small_raw):
    messy = small_raw.head(5).copy()
    messy.loc[messy.index[0], "Geography"] = "  france "
    clean, report = clean_data(messy)
    assert report["invalid_rows"] == 0
    assert clean.loc[0, "Geography"] == "France"


def test_clean_fails_if_only_one_class(small_raw):
    single = small_raw[small_raw[config.TARGET] == 0].head(50)
    with pytest.raises(DataValidationError, match="one class"):
        clean_data(single)


def test_stratified_split_preserves_churn_rate(raw_df):
    clean, _ = clean_data(raw_df)
    X, y = split_features_target(clean)
    X_tr, X_te, y_tr, y_te = split_train_test(X, y)
    assert len(X_te) == int(len(X) * config.TEST_SIZE)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert not set(X_tr.index) & set(X_te.index)  # no overlap -> no leakage


def test_split_rejects_bad_test_size(small_raw):
    clean, _ = clean_data(small_raw)
    X, y = split_features_target(clean)
    with pytest.raises(ValueError):
        split_train_test(X, y, test_size=1.5)


def test_feature_engineer_adds_and_drops(small_raw):
    X = small_raw[config.FEATURES]
    out = FeatureEngineer(add_features=True, drop_features=["Tenure"]).fit_transform(X)
    assert {"HasZeroBalance", "BalanceSalaryRatio", "TenureByAge", "IsSingleProduct"} <= set(out.columns)
    assert "Tenure" not in out.columns
    assert np.isfinite(out["BalanceSalaryRatio"]).all()


def test_feature_engineer_handles_zero_salary(one_customer):
    row = one_customer.copy()
    row["EstimatedSalary"] = 0.0
    out = FeatureEngineer(add_features=True).transform(row)
    assert np.isfinite(out["BalanceSalaryRatio"]).all()


def test_save_refuses_identifiable_data(small_raw, tmp_path):
    with pytest.raises(DataValidationError, match="identifiers"):
        save_processed_data(small_raw, tmp_path / "x.csv")


def test_default_data_path_falls_back_to_cleaned_file(monkeypatch, tmp_path):
    """Deployed copies have no raw CSV (it holds surnames), so the anonymised file is used."""
    monkeypatch.setattr(config, "RAW_DATA_PATH", tmp_path / "missing.csv")
    assert config.default_data_path() == config.PROCESSED_DATA_PATH
