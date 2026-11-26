"""Cached data loaders for the app. Every number shown in the UI comes from here,
i.e. from the dataset and the files written by ``python main.py``.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .. import config
from ..data_loader import load_raw_data
from ..predict import load_model
from ..preprocessing import clean_data


@st.cache_resource(show_spinner=False)
def model_bundle():
    """(pipeline, threshold, metadata) of the saved final model. Raises FileNotFoundError if not trained."""
    return load_model()


@st.cache_data(show_spinner=False)
def dataset_stats() -> Dict[str, object]:
    """Audit and EDA figures computed from the raw dataset."""
    raw = load_raw_data()
    clean, _ = clean_data(raw)
    y = clean[config.TARGET]

    def rate(mask) -> float:
        return float(y[mask].mean()) if mask.any() else float("nan")

    age = clean["Age"]
    age_bands = [("Under 30", age < 30), ("30–39", age.between(30, 39)), ("40–49", age.between(40, 49)),
                 ("50–59", age.between(50, 59)), ("60+", age >= 60)]
    corr = clean[config.NUMERIC_FEATURES + [config.TARGET]].corr()[config.TARGET].drop(config.TARGET)
    return {
        "rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
        "missing": int(raw.isna().sum().sum()), "duplicates": int(raw.duplicated().sum()),
        "duplicate_ids": int(raw["CustomerId"].duplicated().sum()) if "CustomerId" in raw else 0,
        "churned": int(y.sum()), "stayed": int((y == 0).sum()), "churn_rate": float(y.mean()),
        "country_counts": clean["Geography"].value_counts().to_dict(),
        "country_rate": clean.groupby("Geography")[config.TARGET].mean().to_dict(),
        "gender_rate": clean.groupby("Gender")[config.TARGET].mean().to_dict(),
        "products": clean.groupby("NumOfProducts")[config.TARGET].agg(["mean", "count"]).reset_index().values.tolist(),
        "age_bands": [(name, rate(mask)) for name, mask in age_bands],
        "median_age_stayed": float(age[y == 0].median()), "median_age_churned": float(age[y == 1].median()),
        "active_rate": rate(clean["IsActiveMember"] == 1), "inactive_rate": rate(clean["IsActiveMember"] == 0),
        "zero_balance_share": float((clean["Balance"] == 0).mean()),
        "zero_balance_rate": rate(clean["Balance"] == 0), "has_balance_rate": rate(clean["Balance"] > 0),
        "correlations": corr.reindex(corr.abs().sort_values(ascending=False).index).to_dict(),
    }


def _read_csv(name: str) -> Optional[pd.DataFrame]:
    path = config.RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def experiments() -> Optional[pd.DataFrame]:
    return _read_csv("experiment_results.csv")


@st.cache_data(show_spinner=False)
def fairness() -> Optional[pd.DataFrame]:
    return _read_csv("fairness_report.csv")


@st.cache_data(show_spinner=False)
def importance() -> Optional[pd.DataFrame]:
    return _read_csv("feature_importance.csv")


@st.cache_data(show_spinner=False)
def final_metrics() -> Optional[dict]:
    path = config.RESULTS_DIR / "final_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def success_targets() -> Dict[str, float]:
    """Metric -> target from the project's success criteria."""
    return {c["metric"]: c["target"] for c in config.SUCCESS_CRITERIA}


def fmt_pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"
