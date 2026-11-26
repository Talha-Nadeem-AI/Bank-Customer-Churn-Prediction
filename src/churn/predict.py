"""Save, load and use the final model to score new customers.

Command-line demo:
    python -m src.churn.predict --input data/sample/new_customers.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from . import config
from .data_loader import DataValidationError, validate_features

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """The saved model exists but cannot be used (e.g. trained with another scikit-learn version)."""


def save_model(pipeline: Pipeline, threshold: float, metadata: Dict[str, Any],
               model_path: Path = config.FINAL_MODEL_PATH,
               meta_path: Path = config.FINAL_MODEL_META_PATH) -> None:
    """Save the fitted pipeline and a JSON file with its threshold and metrics."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    meta = {"threshold": threshold, "features": config.FEATURES, "sklearn_version": sklearn.__version__, **metadata}
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved model to %s", model_path)


def load_model(model_path: Union[str, Path] = config.FINAL_MODEL_PATH,
               meta_path: Union[str, Path] = config.FINAL_MODEL_META_PATH) -> Tuple[Pipeline, float, Dict]:
    """Load the saved pipeline, its decision threshold and metadata.

    Raises:
        FileNotFoundError: The model has not been trained yet.
        ModelLoadError: The model was trained with a different scikit-learn version or is unreadable.
    """
    model_path, meta_path = Path(model_path), Path(meta_path)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. Run 'python main.py' first to train it."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    saved = meta.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        raise ModelLoadError(
            f"The saved model was trained with scikit-learn {saved}, but this computer has {sklearn.__version__}. "
            f"Run 'python main.py' to retrain it here."
        )
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:  # corrupted file or incompatible library versions
        raise ModelLoadError(f"Could not load the saved model ({exc}). Run 'python main.py' to retrain it.") from exc
    return pipeline, float(meta["threshold"]), meta


def risk_band(probability: float, threshold: float = config.DEFAULT_THRESHOLD) -> str:
    """Translate a churn probability into Low / Medium / High for stakeholders.

    "High" starts exactly at the model's decision threshold, so every customer
    flagged as a likely churner is High risk and nobody else is.
    """
    if not 0 <= probability <= 1:
        raise ValueError(f"Probability {probability} is outside [0, 1].")
    if probability >= threshold:
        return "High"
    if probability >= min(config.LOW_RISK_CUTOFF, threshold):
        return "Medium"
    return "Low"


def predict_churn(customers: pd.DataFrame, pipeline: Optional[Pipeline] = None,
                  threshold: Optional[float] = None) -> pd.DataFrame:
    """Score customers and return probability, yes/no flag and risk band.

    Input is validated first (required columns, allowed categories, value
    ranges). Invalid input raises ``DataValidationError`` with details.

    Args:
        customers: One row per customer with at least ``config.FEATURES``.
        pipeline, threshold: Use a specific model; by default the saved final model.
    """
    if pipeline is None or threshold is None:
        loaded_pipe, loaded_t, _ = load_model()
        pipeline = pipeline or loaded_pipe
        threshold = threshold if threshold is not None else loaded_t

    features = validate_features(customers, allow_missing=True)
    proba = pipeline.predict_proba(features)[:, 1]
    out = customers.copy()
    out["churn_probability"] = proba.round(3)
    out["predicted_churn"] = (proba >= threshold).astype(int)
    out["risk_band"] = [risk_band(p, threshold) for p in proba]
    return out


def main(argv=None) -> int:
    """Command-line entry point: score a CSV of customers and print/save results."""
    parser = argparse.ArgumentParser(description="Predict churn risk for bank customers.")
    parser.add_argument("--input", required=True, help="CSV file with customer features.")
    parser.add_argument("--output", help="Optional CSV path to save predictions.")
    args = parser.parse_args(argv)

    try:
        customers = pd.read_csv(args.input)
        result = predict_churn(customers)
    except (FileNotFoundError, ModelLoadError, DataValidationError, pd.errors.EmptyDataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cols = [c for c in ["Geography", "Gender", "Age", "NumOfProducts", "IsActiveMember", "Balance"]
            if c in result.columns] + ["churn_probability", "predicted_churn", "risk_band"]
    print(result[cols].to_string(index=False))
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
