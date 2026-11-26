"""Run the complete bank customer churn workflow end to end.

    python main.py            # full run (about 3-6 minutes)
    python main.py --quick    # smaller tuning grid (about 1 minute)
    python main.py --data path/to/Churn_Modelling.csv

Steps: load + validate -> audit -> clean -> EDA -> stratified split ->
5 experiments -> final evaluation on the test set -> fairness and error
analysis -> save model and every result to outputs/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from src.churn import config
from src.churn.data_loader import DataValidationError, load_raw_data
from src.churn.eda import run_eda, write_audit_report, write_eda_findings
from src.churn.evaluate import (
    check_success_criteria,
    compute_metrics,
    error_analysis,
    fairness_report,
    plot_confusion_matrix,
    plot_experiment_comparison,
    plot_fairness,
    plot_feature_importance,
    plot_mlp_loss_curve,
    plot_roc_pr_curves,
    plot_threshold_tradeoff,
)
from src.churn.experiments import results_table, run_all_experiments
from src.churn.models import MODEL_LABELS
from src.churn.predict import save_model
from src.churn.preprocessing import clean_data, save_processed_data, split_features_target, split_train_test

logger = logging.getLogger("churn")


def setup_logging() -> None:
    """Log to the console and to outputs/run.log."""
    config.ensure_output_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(config.OUTPUT_DIR / "run.log", mode="w", encoding="utf-8")],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # hide matplotlib's internal info messages


def run_pipeline(data_path=None, quick: bool = False) -> dict:
    """Execute every step and return the key results."""
    t0 = time.time()
    fig_dir, res_dir = config.FIGURES_DIR, config.RESULTS_DIR

    # 1. Load and validate ----------------------------------------------------
    logger.info("STEP 1/8  Loading data")
    raw = load_raw_data(data_path)

    # 2. Clean + audit --------------------------------------------------------
    logger.info("STEP 2/8  Cleaning and auditing data")
    clean, cleaning_report = clean_data(raw)
    write_audit_report(raw, cleaning_report, res_dir / "dataset_audit.md")
    save_processed_data(clean)
    (res_dir / "cleaning_report.json").write_text(json.dumps(cleaning_report, indent=2))

    # 3. EDA ------------------------------------------------------------------
    logger.info("STEP 3/8  Exploratory data analysis")
    findings = run_eda(clean, fig_dir)
    write_eda_findings(findings, res_dir / "eda_findings.md")

    # 4. Split ----------------------------------------------------------------
    logger.info("STEP 4/8  Stratified train/test split (%.0f/%.0f)", (1 - config.TEST_SIZE) * 100,
                config.TEST_SIZE * 100)
    X, y = split_features_target(clean)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    logger.info("Train: %d rows (churn %.1f%%) | Test: %d rows (churn %.1f%%)",
                len(X_train), y_train.mean() * 100, len(X_test), y_test.mean() * 100)
    # Ten real, unseen test customers (no identifiers, no target) for the live demo.
    sample_dir = config.PROJECT_ROOT / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    X_test.head(10).to_csv(sample_dir / "new_customers.csv", index=False)

    # 5. Experiments ----------------------------------------------------------
    logger.info("STEP 5/8  Running experiments (quick=%s)", quick)
    exp = run_all_experiments(X_train, y_train, X_test, y_test, quick=quick)
    table = results_table(exp["results"])
    table.round(4).to_csv(res_dir / "experiment_results.csv", index=False)
    plot_experiment_comparison(table, fig_dir / "model_01_experiment_comparison.png")

    # 6. Final evaluation on the held-out test set ----------------------------
    logger.info("STEP 6/8  Final evaluation on the test set")
    final, threshold = exp["final"], exp["threshold"]
    baseline = exp["results"][0]
    final_default = compute_metrics(y_test, final.test_proba, config.DEFAULT_THRESHOLD)
    plot_confusion_matrix(baseline.test_metrics, "Baseline (E1) - logistic regression, threshold 0.50",
                          fig_dir / "model_02_confusion_baseline.png")
    plot_confusion_matrix(final.test_metrics,
                          f"Final model (E5) - {MODEL_LABELS[final.spec.model_name]}, threshold {threshold:.2f}",
                          fig_dir / "model_03_confusion_final.png")
    curves = {f"{r.spec.run_id} {MODEL_LABELS[r.spec.model_name]}": r.test_proba
              for r in exp["results"] if r.spec.run_id in {"E1", "E3a", "E3b", "E3c", "E5"}}
    plot_roc_pr_curves(curves, y_test, fig_dir / "model_04_roc_pr_curves.png")
    plot_threshold_tradeoff(y_train, exp["oof_proba"], threshold, fig_dir / "model_05_threshold_tuning.png",
                            "Threshold tuning on out-of-fold TRAINING predictions")
    importance = plot_feature_importance(final.pipeline, X_test, y_test, fig_dir / "model_06_feature_importance.png")
    importance.round(4).to_csv(res_dir / "feature_importance.csv", index=False)
    mlp_run = next(r for r in exp["results"] if r.spec.model_name == "mlp_ann")
    plot_mlp_loss_curve(mlp_run.pipeline, fig_dir / "model_07_mlp_training_curve.png")

    # 7. Fairness, errors, success criteria -----------------------------------
    logger.info("STEP 7/8  Fairness, error analysis and success criteria")
    y_pred = (final.test_proba >= threshold).astype(int)
    fair = fairness_report(X_test, y_test, y_pred)
    fair.round(3).to_csv(res_dir / "fairness_report.csv", index=False)
    plot_fairness(fair, fig_dir / "model_08_fairness.png")
    errors = error_analysis(X_test, y_test, y_pred)
    errors.to_csv(res_dir / "error_analysis.csv")
    criteria = check_success_criteria(final.test_metrics, config.SUCCESS_CRITERIA)
    criteria.to_csv(res_dir / "success_criteria.csv", index=False)

    # 8. Save model and summary -----------------------------------------------
    logger.info("STEP 8/8  Saving model and summary")
    metadata = {
        "model": MODEL_LABELS[final.spec.model_name],
        "params": final.spec.model_params,
        "engineered_features": final.spec.add_features,
        "dropped_features": final.spec.drop_features,
        "test_metrics": final.test_metrics,
        "test_metrics_at_0.5": final_default,
        "random_state": config.RANDOM_STATE,
    }
    save_model(final.pipeline, threshold, metadata)
    (res_dir / "final_metrics.json").write_text(json.dumps(metadata, indent=2, default=str))
    write_summary(table, exp, final_default, criteria, fair, errors, importance, cleaning_report,
                  len(X_train), len(X_test), time.time() - t0)
    logger.info("Done in %.0f s. See outputs/results/run_summary.md", time.time() - t0)
    return {"table": table, "final_metrics": final.test_metrics, "threshold": threshold}


def write_summary(table, exp, final_default, criteria, fair, errors, importance, cleaning_report,
                  n_train, n_test, seconds) -> None:
    """Write a one-page Markdown summary of the run (numbers taken from this run)."""
    final, t = exp["final"], exp["threshold"]
    base = exp["results"][0].test_metrics
    fm = final.test_metrics
    cols = ["run_id", "model", "description", "cv_f1", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc"]
    show = table[cols].copy()
    for c in cols[3:]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")

    def md(df: pd.DataFrame) -> str:
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join([head, sep] + rows)

    lines = [
        "# Run Summary - Bank Customer Churn Prediction",
        "",
        f"Run time: {seconds:.0f} s | train rows: {n_train:,} | test rows: {n_test:,} | seed: {config.RANDOM_STATE}",
        f"Rows after cleaning: {cleaning_report['rows_out']:,} of {cleaning_report['rows_in']:,}",
        "",
        "## Experiments (selection uses cv_f1; test columns are for reporting only)",
        "",
        md(show),
        "",
        "## Decisions",
        "",
    ] + [f"- {d}" for d in exp["decisions"]] + [
        "",
        "## Final model vs baseline (held-out test set)",
        "",
        "| Metric | Baseline E1 (t=0.50) | Final E5 (t=0.50) | Final E5 (tuned t=" + f"{t:.2f}) |",
        "| --- | --- | --- | --- |",
    ] + [f"| {m} | {base[m]:.3f} | {final_default[m]:.3f} | {fm[m]:.3f} |"
         for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")] + [
        f"| missed churners (FN) | {base['fn']} | {final_default['fn']} | {fm['fn']} |",
        f"| false alarms (FP) | {base['fp']} | {final_default['fp']} | {fm['fp']} |",
        "",
        "## Success criteria",
        "",
        md(criteria),
        "",
        "## Top features (permutation importance)",
        "",
        md(importance.head(6).round(4)),
        "",
        "## Fairness (test set, final model)",
        "",
        md(fair.round(3)),
        "",
        "## Error analysis (mean feature values by outcome)",
        "",
        md(errors.reset_index()),
    ]
    (config.RESULTS_DIR / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bank customer churn prediction pipeline.")
    parser.add_argument("--data", help="Path to Churn_Modelling.csv (default: data/raw/).")
    parser.add_argument("--quick", action="store_true", help="Use a smaller tuning grid.")
    args = parser.parse_args()
    setup_logging()
    try:
        run_pipeline(args.data, quick=args.quick)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
