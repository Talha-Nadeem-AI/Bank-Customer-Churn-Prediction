"""Documented experiments: baseline -> imbalance handling -> model comparison ->
feature changes -> hyper-parameter and threshold tuning.

Model selection rule (avoids "peeking" at the test set):
    * Every run is scored with 5-fold stratified cross-validation on the
      TRAINING set. Choices between runs use the cross-validated F1 score.
    * Each run is also scored on the held-out test set for the comparison
      table, but test scores are never used to pick a model or threshold.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .evaluate import compute_metrics, find_best_threshold
from .models import MODEL_LABELS, PARAM_GRIDS, QUICK_PARAM_GRIDS, build_pipeline

logger = logging.getLogger(__name__)

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


@dataclass
class RunSpec:
    """Description of one experimental run."""

    run_id: str
    experiment: str
    description: str
    model_name: str
    add_features: bool = False
    drop_features: Optional[List[str]] = None
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Everything produced by one run."""

    spec: RunSpec
    pipeline: Pipeline
    cv_scores: Dict[str, float]
    test_metrics: Dict[str, float]
    test_proba: np.ndarray
    seconds: float

    def as_row(self) -> Dict[str, Any]:
        """Flatten into one row of the results table."""
        row = {
            "run_id": self.spec.run_id,
            "experiment": self.spec.experiment,
            "model": MODEL_LABELS[self.spec.model_name],
            "description": self.spec.description,
            "engineered_features": self.spec.add_features,
            "dropped_features": ", ".join(self.spec.drop_features or []) or "-",
            "params": ", ".join(f"{k}={v}" for k, v in self.spec.model_params.items()) or "defaults",
        }
        row.update({f"cv_{k}": v for k, v in self.cv_scores.items()})
        row.update({f"test_{k}": v for k, v in self.test_metrics.items()})
        row["train_seconds"] = round(self.seconds, 1)
        return row


def make_cv() -> StratifiedKFold:
    """Stratified k-fold splitter shared by all runs (same folds for fair comparison)."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def run_single(spec: RunSpec, X_train, y_train, X_test, y_test,
               threshold: float = config.DEFAULT_THRESHOLD) -> RunResult:
    """Cross-validate a run on the training set, then fit and score it on the test set."""
    start = time.time()
    pipe = build_pipeline(spec.model_name, spec.add_features, spec.drop_features, **spec.model_params)
    cv = cross_validate(pipe, X_train, y_train, cv=make_cv(), scoring=CV_SCORING, n_jobs=config.N_JOBS)
    cv_scores = {m: float(np.mean(cv[f"test_{m}"])) for m in CV_SCORING}
    cv_scores["f1_std"] = float(np.std(cv["test_f1"]))
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    result = RunResult(spec, pipe, cv_scores, compute_metrics(y_test, proba, threshold), proba,
                       time.time() - start)
    logger.info("%-4s %-28s CV F1=%.3f  test F1=%.3f  recall=%.3f", spec.run_id,
                MODEL_LABELS[spec.model_name], cv_scores["f1"], result.test_metrics["f1"],
                result.test_metrics["recall"])
    return result


def best_by_cv(results: List[RunResult], metric: str = "f1") -> RunResult:
    """Return the run with the highest cross-validated score."""
    return max(results, key=lambda r: r.cv_scores[metric])


def run_all_experiments(X_train, y_train, X_test, y_test, quick: bool = False) -> Dict[str, Any]:
    """Run the five documented experiments and return every result.

    Returns a dict with:
        results      - list of RunResult (E1..E4 runs and the tuned E5 run)
        final        - RunResult of the selected final model
        threshold    - tuned decision threshold for the final model
        oof_proba    - out-of-fold training probabilities used to tune the threshold
        grid_search  - fitted GridSearchCV object
        decisions    - plain-language log of what was chosen and why
    """
    results: List[RunResult] = []
    decisions: List[str] = []

    # ---- Experiment 1: baseline ------------------------------------------------
    e1 = run_single(RunSpec("E1", "1. Baseline",
                            "Logistic regression, all original features, default settings, threshold 0.5",
                            "logistic_regression"), X_train, y_train, X_test, y_test)
    results.append(e1)

    # ---- Experiment 2: handle class imbalance -----------------------------------
    e2 = run_single(RunSpec("E2", "2. Class imbalance",
                            "Same as E1 but class_weight='balanced' so missed churners cost more",
                            "logistic_regression", model_params={"class_weight": "balanced"}),
                    X_train, y_train, X_test, y_test)
    results.append(e2)
    decisions.append(
        f"E2 vs E1: balancing class weights changed CV recall from {e1.cv_scores['recall']:.3f} to "
        f"{e2.cv_scores['recall']:.3f} and CV F1 from {e1.cv_scores['f1']:.3f} to {e2.cv_scores['f1']:.3f}."
    )

    # ---- Experiment 3: model comparison -----------------------------------------
    e3_specs = [
        RunSpec("E3a", "3. Model comparison", "Random forest (300 trees), balanced class weights",
                "random_forest", model_params={"class_weight": "balanced"}),
        RunSpec("E3b", "3. Model comparison", "Histogram gradient boosting, balanced class weights",
                "gradient_boosting", model_params={"class_weight": "balanced"}),
        RunSpec("E3c", "3. Model comparison",
                "MLP neural network (3 x 11 ReLU, same as original Keras ANN), early stopping",
                "mlp_ann"),
    ]
    e3 = [run_single(s, X_train, y_train, X_test, y_test) for s in e3_specs]
    results.extend(e3)
    best_family = best_by_cv([e2] + e3)
    decisions.append(
        f"E3: best model family by CV F1 is {MODEL_LABELS[best_family.spec.model_name]} "
        f"(CV F1 {best_family.cv_scores['f1']:.3f})."
    )

    # ---- Experiment 4: feature engineering and selection -------------------------
    fam = best_family.spec
    e4_specs = [
        RunSpec("E4a", "4. Feature changes", "Best E3 model + 4 engineered features",
                fam.model_name, add_features=True, model_params=dict(fam.model_params)),
        RunSpec("E4b", "4. Feature changes",
                f"Best E3 model without weak features ({', '.join(config.WEAK_FEATURES)})",
                fam.model_name, drop_features=list(config.WEAK_FEATURES), model_params=dict(fam.model_params)),
        RunSpec("E4c", "4. Feature changes", "Best E3 model + engineered features - weak features",
                fam.model_name, add_features=True, drop_features=list(config.WEAK_FEATURES),
                model_params=dict(fam.model_params)),
    ]
    e4 = [run_single(s, X_train, y_train, X_test, y_test) for s in e4_specs]
    results.extend(e4)
    best_features = best_by_cv([best_family] + e4)
    decisions.append(
        f"E4: best feature set by CV F1 is run {best_features.spec.run_id} "
        f"(engineered={best_features.spec.add_features}, dropped={best_features.spec.drop_features or 'none'}, "
        f"CV F1 {best_features.cv_scores['f1']:.3f})."
    )

    # ---- Experiment 5: hyper-parameter tuning + threshold tuning -----------------
    spec = best_features.spec
    grid = (QUICK_PARAM_GRIDS if quick else PARAM_GRIDS)[spec.model_name]
    base_pipe = build_pipeline(spec.model_name, spec.add_features, spec.drop_features)
    search = GridSearchCV(base_pipe, grid, scoring="f1", cv=make_cv(), n_jobs=config.N_JOBS, refit=True)
    start = time.time()
    search.fit(X_train, y_train)
    tuned_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    decisions.append(
        f"E5: grid search tried {len(search.cv_results_['params'])} combinations x {config.CV_FOLDS} folds; "
        f"best CV F1 {search.best_score_:.3f} with {tuned_params}."
    )

    # Tune the decision threshold on out-of-fold TRAINING predictions only.
    tuned_spec = RunSpec("E5", "5. Tuning", "Grid-searched hyper-parameters + tuned decision threshold",
                         spec.model_name, spec.add_features, spec.drop_features, tuned_params)
    tuned_pipe = build_pipeline(spec.model_name, spec.add_features, spec.drop_features, **tuned_params)
    oof = cross_val_predict(tuned_pipe, X_train, y_train, cv=make_cv(), method="predict_proba", n_jobs=config.N_JOBS)[:, 1]
    threshold = find_best_threshold(y_train, oof, metric="recall",
                                    min_precision=config.THRESHOLD_MIN_PRECISION)
    oof_metrics = compute_metrics(y_train, oof, threshold)
    decisions.append(
        f"E5: threshold tuned on out-of-fold training predictions to maximise recall with precision >= "
        f"{config.THRESHOLD_MIN_PRECISION:.2f} -> {threshold:.2f} "
        f"(OOF F1 {oof_metrics['f1']:.3f}, recall {oof_metrics['recall']:.3f}, precision {oof_metrics['precision']:.3f})."
    )

    final_pipe = search.best_estimator_
    proba = final_pipe.predict_proba(X_test)[:, 1]
    cv_scores = {m: float(oof_metrics[m]) for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    cv_scores["f1_std"] = float(search.cv_results_["std_test_score"][search.best_index_])
    final = RunResult(tuned_spec, final_pipe, cv_scores, compute_metrics(y_test, proba, threshold),
                      proba, time.time() - start)
    results.append(final)

    decisions.append(
        f"E5 vs {best_features.spec.run_id}: tuning changed CV F1 (threshold 0.5) from "
        f"{best_features.cv_scores['f1']:.3f} to {search.best_score_:.3f}; at the tuned threshold the OOF "
        f"recall is {oof_metrics['recall']:.3f}. (E5's cv_* columns are out-of-fold scores at the tuned threshold.)"
    )
    decisions.append(
        f"FINAL MODEL: {MODEL_LABELS[spec.model_name]} (run E5), threshold {threshold:.2f}. Selected on "
        f"cross-validated training performance only; test set used once for the final report."
    )
    return {"results": results, "final": final, "threshold": threshold, "oof_proba": oof,
            "grid_search": search, "decisions": decisions}


def results_table(results: List[RunResult]) -> pd.DataFrame:
    """Combine every run into one comparison DataFrame."""
    return pd.DataFrame([r.as_row() for r in results])
