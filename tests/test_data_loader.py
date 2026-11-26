"""Tests for loading and validating data (failure conditions)."""
import pandas as pd
import pytest

from src.churn import config
from src.churn.data_loader import DataValidationError, find_invalid_rows, load_raw_data, validate_features


def test_load_real_dataset_has_expected_shape(raw_df):
    df = load_raw_data()
    assert df.shape == raw_df.shape
    assert config.TARGET in df.columns


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_raw_data(path)


def test_header_only_file_raises(tmp_path, small_raw):
    path = tmp_path / "header_only.csv"
    small_raw.head(0).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="no rows"):
        load_raw_data(path)


def test_wrong_extension_raises(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("x")
    with pytest.raises(DataValidationError, match=".csv"):
        load_raw_data(path)


def test_missing_column_raises(tmp_path, small_raw):
    path = tmp_path / "no_age.csv"
    small_raw.drop(columns=["Age"]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Age"):
        load_raw_data(path)


def test_find_invalid_rows_detects_bad_values(small_raw):
    bad = small_raw.head(3).copy()
    bad["Age"] = bad["Age"].astype(object)
    bad.loc[bad.index[0], "Age"] = -5            # impossible age
    bad.loc[bad.index[1], "Geography"] = "Mars"   # unknown country
    bad.loc[bad.index[2], "Age"] = "forty"        # text instead of a number
    problems = find_invalid_rows(bad)
    assert set(problems["problem"].str.split().str[0]) >= {"below", "not"}
    assert len(problems) == 3


def test_validate_features_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_features([1, 2, 3])


def test_validate_features_rejects_empty(one_customer):
    with pytest.raises(DataValidationError, match="No customer"):
        validate_features(one_customer.iloc[0:0])


def test_validate_features_missing_values_policy(one_customer):
    row = one_customer.copy()
    row.loc[0, "CreditScore"] = None
    assert validate_features(row, allow_missing=True)["CreditScore"].isna().all()
    with pytest.raises(DataValidationError, match="missing"):
        validate_features(row, allow_missing=False)


def test_validate_features_ignores_extra_columns(one_customer):
    row = one_customer.assign(Surname="ShouldBeIgnored")
    assert list(validate_features(row).columns) == config.FEATURES
