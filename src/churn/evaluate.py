"""Evaluation: metrics, threshold tuning, charts, fairness and error analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config  # noqa: E402


def _check_inputs(y_true: Sequence[int], y_prob: Sequence[float]) -> tuple:
    """Validate and convert evaluation inputs to NumPy arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true has shape {y_true.shape} but y_prob has {y_prob.shape}.")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate on an empty set.")
    if np.any((y_prob < 0) | (y_prob > 1)) or np.isnan(y_prob).any():
        raise ValueError("Probabilities must be between 0 and 1 with no NaN values.")
    return y_true, y_prob


def compute_metrics(y_true, y_prob, threshold: float = config.DEFAULT_THRESHOLD) -> Dict[str, float]:
    """All classification metrics required by the assignment, at a given threshold.

    Returns accuracy, precision, recall, F1 and the confusion-matrix counts,
    plus ROC-AUC and PR-AUC (threshold-independent ranking quality).
    """
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")
    y_true, y_prob = _check_inputs(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if both_classes else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, metric: str = "f1",
                        min_precision: Optional[float] = None) -> float:
    """Choose the probability cut-off that maximises ``metric`` (F1 or recall).

    This must be called on *training* (out-of-fold) predictions, never on the
    test set, otherwise the test score would be optimistically biased.

    Args:
        metric: "f1" or "recall".
        min_precision: Optional floor on precision (only used with "recall").
    """
    y_true, y_prob = _check_inputs(y_true, y_prob)
    if metric not in {"f1", "recall"}:
        raise ValueError("metric must be 'f1' or 'recall'.")
    best_t, best_score = config.DEFAULT_THRESHOLD, -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            if min_precision is not None and prec < min_precision:
                continue
            score = recall_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_t, best_score = float(round(t, 2)), score
    return best_t


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics: Dict[str, float], title: str, path: Path) -> None:
    """Draw a labelled 2x2 confusion matrix from the counts in ``metrics``."""
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    names = [["True negative\n(stayed, predicted stay)", "False positive\n(stayed, flagged)"],
             ["False negative\n(churned, MISSED)", "True positive\n(churned, caught)"]]
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=8, color=colour)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: stay", "Predicted: churn"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: stayed", "Actual: churned"])
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_roc_pr_curves(curves: Dict[str, tuple], y_true, path: Path) -> None:
    """ROC and precision-recall curves for several models on the test set.

    Args:
        curves: {label: y_prob}.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, y_prob in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.3f})")
        ax2.plot(rec, prec, label=f"{label} (AP={average_precision_score(y_true, y_prob):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate (recall)")
    ax1.set_title("ROC curves (test set)")
    ax1.legend(fontsize=8)
    ax2.axhline(np.mean(y_true), color="k", ls="--", lw=1, label="Churn rate (no skill)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-recall curves (test set)")
    ax2.legend(fontsize=8)
    _save(fig, path)


def plot_threshold_tradeoff(y_true, y_prob, chosen: float, path: Path, title: str) -> None:
    """Precision, recall and F1 as the decision threshold changes."""
    y_true, y_prob = _check_inputs(y_true, y_prob)
    ts = np.arange(0.05, 0.951, 0.01)
    rows = [compute_metrics(y_true, y_prob, t) for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("precision", "recall", "f1"):
        ax.plot(ts, [r[key] for r in rows], label=key.capitalize())
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"Chosen threshold = {chosen:.2f}")
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="Default 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def plot_experiment_comparison(results: pd.DataFrame, path: Path) -> None:
    """Grouped bar chart of test metrics for every experiment run."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(max(9, len(results) * 1.3), 5))
    x = np.arange(len(results))
    width = 0.16
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * width, results[f"test_{m}"], width, label=m.replace("_", "-").upper()
               if m == "roc_auc" else m.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(results["run_id"], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on test set")
    ax.set_title("Experiment comparison (test set)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)


def plot_feature_importance(pipeline, X_test: pd.DataFrame, y_test, path: Path,
                            n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance on the test set (drop in ROC-AUC when a feature is shuffled).

    Works for any model, including the neural network, and is measured on the
    original (human-readable) features.
    """
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)
    imp = pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean,
                        "std": result.importances_std}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#4C72B0")
    ax.set_xlabel("Mean drop in ROC-AUC when shuffled")
    ax.set_title("Permutation feature importance (final model, test set)")
    _save(fig, path)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_mlp_loss_curve(pipeline, path: Path) -> None:
    """Training loss and validation score per epoch for the MLP (ANN) model."""
    mlp = pipeline.named_steps["model"]
    if not hasattr(mlp, "loss_curve_"):
        raise ValueError("The pipeline's model is not a fitted MLPClassifier.")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(mlp.loss_curve_, label="Training loss", color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log-loss")
    if getattr(mlp, "validation_scores_", None) is not None:
        ax2 = ax1.twinx()
        ax2.plot(mlp.validation_scores_, label="Validation accuracy", color="#DD8452")
        ax2.set_ylabel("Validation accuracy")
        ax2.legend(loc="center right")
    ax1.legend(loc="upper right")
    ax1.set_title("MLP (ANN) training curve with early stopping")
    _save(fig, path)


# --------------------------------------------------------------------------
# Fairness and error analysis
# --------------------------------------------------------------------------
def fairness_report(X_test: pd.DataFrame, y_true, y_pred,
                    groups: Sequence[str] = config.SENSITIVE_FEATURES) -> pd.DataFrame:
    """Per-group performance for sensitive attributes (Gender, Geography).

    Large gaps in recall or false-positive rate between groups mean the model
    treats groups differently, which must be reported to the stakeholder.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rows = []
    for attr in groups:
        for value in sorted(X_test[attr].dropna().unique()):
            mask = (X_test[attr] == value).to_numpy()
            yt, yp = y_true[mask], y_pred[mask]
            negatives = max((yt == 0).sum(), 1)
            rows.append({
                "attribute": attr, "group": value, "n": int(mask.sum()),
                "actual_churn_rate": yt.mean(),
                "predicted_churn_rate": yp.mean(),
                "recall": recall_score(yt, yp, zero_division=0),
                "precision": precision_score(yt, yp, zero_division=0),
                "false_positive_rate": ((yp == 1) & (yt == 0)).sum() / negatives,
                "f1": f1_score(yt, yp, zero_division=0),
            })
    return pd.DataFrame(rows)


def plot_fairness(report: pd.DataFrame, path: Path) -> None:
    """Bar chart of recall and false-positive rate per sensitive group."""
    labels = [f"{a}:\n{g}" for a, g in zip(report["attribute"], report["group"])]
    x = np.arange(len(report))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, report["recall"], 0.4, label="Recall (churners caught)")
    ax.bar(x + 0.2, report["false_positive_rate"], 0.4, label="False positive rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Fairness check: final model performance by group (test set)")
    ax.legend()
    _save(fig, path)


def error_analysis(X_test: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Compare average feature values of each outcome type (TP, FN, FP, TN).

    Shows what kind of churners the model misses (false negatives).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    outcome = np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 1) & (y_pred == 0),
         (y_true == 0) & (y_pred == 1), (y_true == 0) & (y_pred == 0)],
        ["TP (caught churner)", "FN (missed churner)", "FP (false alarm)", "TN (correct stay)"],
        default="unknown",  # must be a string: newer NumPy refuses to mix str choices with int default 0
    )
    frame = X_test.copy()
    frame["outcome"] = outcome
    frame["is_germany"] = (frame["Geography"] == "Germany").astype(float)
    frame["is_female"] = (frame["Gender"] == "Female").astype(float)
    numeric = config.NUMERIC_FEATURES + ["is_germany", "is_female"]
    summary = frame.groupby("outcome")[numeric].mean()
    summary.insert(0, "count", frame.groupby("outcome").size())
    return summary.round(2)


def check_success_criteria(metrics: Dict[str, float], criteria: List[Dict]) -> pd.DataFrame:
    """Compare final test metrics with the project's measurable success criteria."""
    rows = []
    for c in criteria:
        value = metrics[c["metric"]]
        rows.append({"criterion": c["text"], "metric": c["metric"], "target": c["target"],
                     "achieved": round(value, 3), "met": bool(value >= c["target"])})
    return pd.DataFrame(rows)
