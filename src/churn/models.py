"""Model definitions. Every model is a full scikit-learn Pipeline:

    FeatureEngineer  ->  preprocessing (impute, scale, one-hot)  ->  classifier

Keeping preprocessing inside the pipeline means it is re-fitted on the
training part of every cross-validation fold, which prevents data leakage.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import FeatureEngineer, build_preprocessor

# Human-readable names used in tables and charts.
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp_ann": "MLP Neural Network (ANN)",
}

# Hyper-parameter grids searched in the tuning experiment.
# Keys use the "model__" prefix because the classifier is the "model" step of the pipeline.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [3, 5, None],
        "model__max_leaf_nodes": [15, 31],
        "model__l2_regularization": [0.0, 1.0],
        "model__class_weight": [None, "balanced"],
    },
    "mlp_ann": {
        "model__hidden_layer_sizes": [(11, 11, 11), (32, 16), (64, 32)],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__learning_rate_init": [1e-3, 1e-2],
    },
}

# Smaller grids for a fast run (python main.py --quick).
QUICK_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "logistic_regression": {"model__C": [0.1, 1.0], "model__class_weight": ["balanced"]},
    "random_forest": {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5],
                      "model__class_weight": ["balanced"]},
    "gradient_boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, None],
                          "model__class_weight": [None, "balanced"]},
    "mlp_ann": {"model__hidden_layer_sizes": [(11, 11, 11), (32, 16)], "model__alpha": [1e-4, 1e-2]},
}


def make_estimator(name: str, **params: Any) -> ClassifierMixin:
    """Create an untrained classifier by name with sensible, seeded defaults.

    Args:
        name: One of ``MODEL_LABELS`` keys.
        **params: Extra keyword arguments passed to the estimator (e.g.
            ``class_weight="balanced"``).

    Raises:
        ValueError: Unknown model name.
    """
    rs = config.RANDOM_STATE
    if name == "logistic_regression":
        base = dict(max_iter=2000, random_state=rs)
        return LogisticRegression(**{**base, **params})
    if name == "random_forest":
        base = dict(n_estimators=300, random_state=rs, n_jobs=config.N_JOBS)
        return RandomForestClassifier(**{**base, **params})
    if name == "gradient_boosting":
        base = dict(max_iter=300, random_state=rs)
        return HistGradientBoostingClassifier(**{**base, **params})
    if name == "mlp_ann":
        # Same architecture as the team's original Keras ANN: 3 hidden layers of 11 ReLU units.
        base = dict(hidden_layer_sizes=(11, 11, 11), activation="relu", solver="adam",
                    max_iter=500, early_stopping=True, validation_fraction=0.2,
                    n_iter_no_change=20, random_state=rs)
        return MLPClassifier(**{**base, **params})
    raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_LABELS)}.")


def build_pipeline(
    model_name: str,
    add_features: bool = False,
    drop_features: Optional[Sequence[str]] = None,
    **model_params: Any,
) -> Pipeline:
    """Assemble the full feature-engineering + preprocessing + model pipeline."""
    return Pipeline([
        ("features", FeatureEngineer(add_features=add_features,
                                     drop_features=list(drop_features) if drop_features else None)),
        ("preprocess", build_preprocessor()),
        ("model", make_estimator(model_name, **model_params)),
    ])
