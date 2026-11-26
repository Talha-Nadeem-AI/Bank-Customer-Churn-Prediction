"""Tests for models, metrics and prediction on unseen/edge-case inputs."""
import numpy as np
import pandas as pd
import pytest

from src.churn import config
from src.churn.data_loader import DataValidationError
from src.churn.evaluate import compute_metrics, find_best_threshold
from src.churn.models import MODEL_LABELS, build_pipeline, make_estimator
from src.churn.predict import ModelLoadError, load_model, predict_churn, risk_band, save_model


@pytest.mark.parametrize("name", list(MODEL_LABELS))
def test_every_model_trains_and_predicts_probabilities(name, small_raw):
    from src.churn.preprocessing import clean_data, split_features_target
    X, y = split_features_target(clean_data(small_raw)[0])
    pipe = build_pipeline(name).fit(X, y)
    proba = pipe.predict_proba(X.head(5))[:, 1]
    assert proba.shape == (5,) and ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        make_estimator("xgboost_magic")


def test_metrics_match_hand_calculation():
    y_true = [1, 1, 0, 0, 1]
    y_prob = [0.9, 0.2, 0.8, 0.1, 0.7]
    m = compute_metrics(y_true, y_prob, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [0.5])          # length mismatch
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [1.4, 0.2])     # probability > 1
    with pytest.raises(ValueError):
        compute_metrics([], [])                 # empty
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [0.4, 0.2], threshold=0)


def test_threshold_respects_precision_floor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    t = find_best_threshold(y, p, metric="recall", min_precision=0.7)
    assert compute_metrics(y, p, t)["precision"] >= 0.7


def test_predict_single_customer(trained_pipeline, one_customer):
    out = predict_churn(one_customer, trained_pipeline, threshold=0.5)
    assert len(out) == 1
    assert out.loc[0, "risk_band"] in {"Low", "Medium", "High"}
    assert out.loc[0, "predicted_churn"] in (0, 1)


def test_predict_with_missing_value_is_imputed(trained_pipeline, one_customer):
    row = one_customer.copy()
    row.loc[0, "Balance"] = np.nan
    out = predict_churn(row, trained_pipeline, threshold=0.5)
    assert 0 <= out.loc[0, "churn_probability"] <= 1


def test_predict_rejects_unknown_country(trained_pipeline, one_customer):
    row = one_customer.copy()
    row.loc[0, "Geography"] = "Italy"
    with pytest.raises(DataValidationError, match="Geography"):
        predict_churn(row, trained_pipeline, threshold=0.5)


def test_predict_rejects_impossible_values(trained_pipeline, one_customer):
    row = one_customer.copy()
    row.loc[0, "NumOfProducts"] = 0
    with pytest.raises(DataValidationError, match="NumOfProducts"):
        predict_churn(row, trained_pipeline, threshold=0.5)


def test_predict_rejects_missing_column(trained_pipeline, one_customer):
    with pytest.raises(DataValidationError, match="Age"):
        predict_churn(one_customer.drop(columns=["Age"]), trained_pipeline, threshold=0.5)


def test_predict_handles_extreme_but_valid_values(trained_pipeline, one_customer):
    row = one_customer.copy()
    row.loc[0, ["Age", "Balance", "CreditScore"]] = [92, 250000.0, 350]
    out = predict_churn(row, trained_pipeline, threshold=0.5)
    assert np.isfinite(out.loc[0, "churn_probability"])


def test_risk_band_boundaries():
    assert risk_band(0.0, 0.45) == "Low"
    assert risk_band(0.30, 0.45) == "Medium"
    assert risk_band(0.449, 0.45) == "Medium"
    assert risk_band(0.45, 0.45) == "High"      # High starts exactly at the threshold
    assert risk_band(1.0, 0.45) == "High"
    with pytest.raises(ValueError):
        risk_band(1.5, 0.45)


def test_high_risk_always_matches_flag(trained_pipeline, small_raw):
    out = predict_churn(small_raw.head(100), trained_pipeline, threshold=0.45)
    assert ((out["risk_band"] == "High") == (out["predicted_churn"] == 1)).all()


def test_load_model_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python main.py"):
        load_model(tmp_path / "none.joblib", tmp_path / "none.json")


def test_saved_final_model_scores_unseen_test_customers():
    """Runs only after `python main.py` has produced the final model."""
    if not config.FINAL_MODEL_PATH.exists():
        pytest.skip("Final model not trained yet")
    sample = pd.read_csv(config.PROJECT_ROOT / "data" / "sample" / "new_customers.csv")
    out = predict_churn(sample)
    assert len(out) == len(sample)
    assert out["churn_probability"].between(0, 1).all()


def test_model_from_other_sklearn_version_gives_clear_error(trained_pipeline, tmp_path):
    model, meta = tmp_path / "m.joblib", tmp_path / "m.json"
    save_model(trained_pipeline, 0.45, {}, model, meta)
    import json
    data = json.loads(meta.read_text())
    data["sklearn_version"] = "0.0.1"
    meta.write_text(json.dumps(data))
    with pytest.raises(ModelLoadError, match="retrain"):
        load_model(model, meta)
