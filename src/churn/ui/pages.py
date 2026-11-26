"""The six ChurnGuard pages. All figures are read from the dataset, the saved
model and the results written by ``python main.py``."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config
from ..data_loader import DataValidationError
from ..predict import ModelLoadError, predict_churn
from . import data
from . import theme as t
from .components import (bar_rows, callout, card, chip, column_chart, esc, gauge, grid, kpi, legend,
                         page_header, pill, render, stat_row, table)

RUN_FIRST = "Run `python main.py` in the project folder first, then refresh this page."
ACTIONS = {
    "Low": "No action needed. Keep the normal service.",
    "Medium": "Watch list. Consider a light-touch check-in.",
    "High": "Prioritise for a retention call or offer this week.",
}


def _model_or_stop():
    try:
        return data.model_bundle()
    except FileNotFoundError:
        st.error("No trained model found. " + RUN_FIRST)
        st.stop()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()


def _met(value: float, metric: str):
    target = data.success_targets().get(metric)
    if target is None:
        return "", None
    ok = value >= target
    shown = f"{target:.0%}" if target < 1 and metric != "roc_auc" else f"{target:.2f}"
    return f"Target {shown} · {'met' if ok else 'not met'}", (t.GOOD[0] if ok else t.BAD[0])


# =============================================================================
# 1. Overview
# =============================================================================
def overview() -> None:
    stats = data.dataset_stats()
    meta = data.final_metrics()
    fair = data.fairness()
    page_header("Retention team dashboard", "Overview",
                pill(f"Churn Modelling dataset · {stats['rows']:,} customers"))

    cc = stats["country_counts"]
    tiles = [
        kpi("Customers analysed", f"{stats['rows']:,}",
            " · ".join(f"{k} {v:,}" for k, v in sorted(cc.items()))),
        kpi("Historical churn rate", data.fmt_pct(stats["churn_rate"]), f"{stats['churned']:,} customers left the bank"),
    ]
    if meta:
        tm = meta["test_metrics"]
        note, col = _met(tm["recall"], "recall")
        tiles.append(kpi("Churners caught (recall)", data.fmt_pct(tm["recall"]), note, col))
        note, col = _met(tm["roc_auc"], "roc_auc")
        tiles.append(kpi("Ranking quality (ROC-AUC)", f"{tm['roc_auc']:.3f}", note, col))
    grid(tiles, len(tiles))

    products = stats["products"]
    chart = column_chart([(f"{int(n)} product{'s' if n > 1 else ''} · {int(c):,}", m, data.fmt_pct(m))
                          for n, m, c in products], max_value=1.0,
                         colours=[t.PRIMARY if m > 0.5 else t.PRIMARY_LIGHT for _, m, _ in products])
    left = card(f'<h2 class="cg-h2">Churn rate by number of products</h2><p class="cg-sub" style="margin-bottom:18px">'
                f'The strongest signal in the data. Customers with 2 products are the safest; 3–4 products almost always leave.</p>{chart}',
                "grid-column:span 2", "cg-span2")
    if meta:
        tm = meta["test_metrics"]
        right = card(f'<h2 class="cg-h2">Test-set outcomes</h2><p class="cg-sub">{tm["tp"] + tm["fn"] + tm["fp"] + tm["tn"]:,} customers the model never saw</p>'
                     + stat_row("Churners caught", f'{tm["tp"]:,}', "Low") + stat_row("Churners missed", f'{tm["fn"]:,}', "High")
                     + stat_row("False alarms", f'{tm["fp"]:,}', "Medium") + stat_row("Correctly kept", f'{tm["tn"]:,}'))
    else:
        right = card(f'<h2 class="cg-h2">Test-set outcomes</h2><p class="cg-sub">{esc(RUN_FIRST)}</p>')
    render(f'<div class="cg-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin-bottom:16px">{left}{right}</div>')

    countries = sorted(stats["country_rate"].items(), key=lambda kv: -kv[1])
    c1 = card('<h2 class="cg-h2">Churn by country</h2>' + bar_rows(
        [(k, v, data.fmt_pct(v)) for k, v in countries], max(v for _, v in countries) * 2,
        [t.PRIMARY] + [t.PRIMARY_LIGHT] * (len(countries) - 1)))
    bands = sorted(stats["age_bands"], key=lambda kv: -kv[1])
    c2 = card('<h2 class="cg-h2">Churn by age band</h2>' + bar_rows(
        [(k, v, data.fmt_pct(v)) for k, v in bands], bands[0][1],
        [t.PRIMARY, t.PRIMARY_MID, t.PRIMARY_LIGHT, "#C6DAD3", "#C6DAD3"]))
    attention = '<h2 class="cg-h2">Needs attention</h2>'
    if fair is not None and not fair.empty:
        worst = fair.sort_values("false_positive_rate", ascending=False).iloc[0]
        others = fair[(fair["attribute"] == worst["attribute"]) & (fair["group"] != worst["group"])]["false_positive_rate"]
        attention += callout("Fairness gap.", f"{worst['group']} customers who stay are flagged {worst['false_positive_rate']:.1%} "
                             f"of the time, vs {others.min():.1%}–{others.max():.1%} for other {worst['attribute'].lower()} groups.", "High")
    attention += callout("Inactive members", f"churn at {data.fmt_pct(stats['inactive_rate'])} vs "
                         f"{data.fmt_pct(stats['active_rate'])} for active members.", "Medium")
    grid([c1, c2, card(attention)], 3)


# =============================================================================
# 2. Predict a customer
# =============================================================================
DEFAULT_CUSTOMER = {"Geography": "Germany", "Gender": "Female", "Age": 55, "CreditScore": 650, "Tenure": 5,
                    "NumOfProducts": 3, "Balance": 100000.0, "EstimatedSalary": 100000.0,
                    "HasCrCard": 1, "IsActiveMember": 0}


def _score(customer: dict, pipeline, threshold: float) -> pd.Series:
    return predict_churn(pd.DataFrame([customer]), pipeline, threshold).iloc[0]


def predict() -> None:
    pipeline, threshold, _ = _model_or_stop()
    page_header("Single customer", "Predict churn risk")
    current = st.session_state.setdefault("customer", dict(DEFAULT_CUSTOMER))

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.form("customer_form", border=True):
            render('<h2 class="cg-h2" style="margin-bottom:6px">Customer details</h2>'
                   '<div style="font-size:12px;font-weight:800;color:#5E6B66;letter-spacing:0.6px">PROFILE</div>')
            a, b = st.columns(2)
            countries = sorted(config.VALID_CATEGORIES["Geography"])
            geography = a.selectbox("Country", countries, index=countries.index(current["Geography"]))
            gender = b.segmented_control("Gender", ["Female", "Male"], default=current["Gender"])
            age = a.slider("Age", 18, 92, int(current["Age"]))
            credit = b.slider("Credit score", 350, 850, int(current["CreditScore"]))
            render('<div style="font-size:12px;font-weight:800;color:#5E6B66;letter-spacing:0.6px;margin-top:4px">RELATIONSHIP WITH THE BANK</div>')
            c, d = st.columns(2)
            tenure = c.slider("Tenure (years)", 0, 10, int(current["Tenure"]))
            products = d.segmented_control("Number of products", [1, 2, 3, 4], default=int(current["NumOfProducts"]))
            balance = c.number_input("Account balance", 0.0, 300000.0, float(current["Balance"]), 1000.0, format="%.2f")
            salary = d.number_input("Estimated salary", 0.0, 250000.0, float(current["EstimatedSalary"]), 1000.0, format="%.2f")
            card_flag = c.checkbox("Has credit card", bool(current["HasCrCard"]))
            active = d.checkbox("Active member", bool(current["IsActiveMember"]))
            if st.form_submit_button("Predict churn risk", type="primary", use_container_width=True):
                st.session_state["customer"] = current = {
                    "Geography": geography, "Gender": gender or current["Gender"], "Age": age, "CreditScore": credit,
                    "Tenure": tenure, "NumOfProducts": products or current["NumOfProducts"], "Balance": balance,
                    "EstimatedSalary": salary, "HasCrCard": int(card_flag), "IsActiveMember": int(active)}

    with right:
        result = _score(current, pipeline, threshold)
        p, risk = float(result["churn_probability"]), result["risk_band"]
        flagged = bool(result["predicted_churn"])
        low = config.LOW_RISK_CUTOFF
        render(card(
            f'<h2 class="cg-h2">Prediction</h2>{gauge(p, risk)}'
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:10px">{chip(f"{risk} risk", risk)}'
            f'<span class="cg-chip" style="background:{t.BG};color:{t.SUBTLE}">{"Flagged: likely to churn" if flagged else "Likely to stay"}</span></div>'
            f'<div style="margin-top:14px;padding:12px 14px;border-radius:9px;background:#F6F7F4;font-size:13px;line-height:1.5;color:{t.SUBTLE}">'
            f'<strong style="color:{t.INK}">Suggested action:</strong> {esc(ACTIONS[risk])}</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;color:{t.MUTED};margin-top:10px">'
            f'<span>Low &lt; {low:.2f}</span><span>Medium {low:.2f}–{threshold:.2f}</span><span>High ≥ {threshold:.2f}</span></div>',
            "margin-bottom:16px"))

        scenarios = [("Current customer", {})]
        if current["NumOfProducts"] != 2:
            scenarios.append(("Holds 2 products", {"NumOfProducts": 2}))
        if current["Balance"] > 0:
            scenarios.append(("Balance drops to 0", {"Balance": 0.0}))
        if not current["IsActiveMember"]:
            scenarios.append(("Becomes an active member", {"IsActiveMember": 1}))
        rows = []
        for name, change in scenarios:
            q = float(_score({**current, **change}, pipeline, threshold)["churn_probability"])
            diff = q - p
            if not change:
                delta = f'<span style="color:{t.MUTED}">—</span>'
            else:
                colour = t.GOOD[0] if diff < 0 else t.BAD[0]
                delta = f'<strong style="color:{colour}">{diff * 100:+.1f} pts</strong>'
            rows.append([name, f"<strong>{q:.1%}</strong>", delta])
        render(card('<h2 class="cg-h2">What-if scenarios</h2><p class="cg-sub" style="margin-bottom:10px">Same customer with one thing changed. '
                    'These are patterns in the data, not proof of cause.</p>'
                    + table(["Scenario", "Probability", "Change"], rows, right=(1, 2), raw_cols=(1, 2))))


# =============================================================================
# 3. Batch scoring
# =============================================================================
MAX_ROWS_SHOWN = 200


def batch() -> None:
    pipeline, threshold, _ = _model_or_stop()
    page_header("Many customers at once", "Batch scoring")
    sample_path = config.PROJECT_ROOT / "data" / "sample" / "new_customers.csv"

    up, side = st.columns([2, 1], gap="medium")
    with up:
        uploaded = st.file_uploader("Drop a customers CSV here", type=["csv"],
                                    help="Needs " + ", ".join(config.FEATURES) + ". Extra columns are ignored.")
    with side:
        render(f'<div class="cg-label" style="margin:6px 0 8px">No file handy?</div>')
        if st.button("Use 10 sample test customers", use_container_width=True, disabled=not sample_path.exists()):
            st.session_state["batch_sample"] = True
    render(f'<p class="cg-sub" style="margin:-6px 0 14px">Required columns: {esc(", ".join(config.FEATURES))}. '
           f'Rows with invalid values (e.g. unknown country, age 150) are rejected with an explanation.</p>')

    frame, source = None, ""
    try:
        if uploaded is not None:
            frame, source = pd.read_csv(uploaded), uploaded.name
            st.session_state["batch_sample"] = False
        elif st.session_state.get("batch_sample"):
            frame, source = pd.read_csv(sample_path), "new_customers.csv · unseen test-set customers, no identifiers"
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        render(callout("Could not read the file:", str(exc), "High"))

    if frame is not None:
        try:
            scored = predict_churn(frame, pipeline, threshold)
        except DataValidationError as exc:
            lines = "<br>".join(esc(line) for line in str(exc).splitlines())
            render(f'<div role="alert" style="border:1px solid #E7C9C4;background:#FBF4F2;border-radius:14px;padding:16px 20px">'
                   f'<div style="font-weight:800;color:{t.BAD[0]}">File failed validation, nothing was scored</div>'
                   f'<div style="font-size:13px;color:{t.SUBTLE};margin-top:8px;line-height:1.7;font-family:ui-monospace,Menlo,monospace;white-space:pre">{lines}</div></div>')
        else:
            scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
            n_high = int((scored["risk_band"] == "High").sum())
            grid([kpi("Customers scored", f"{len(scored):,}"),
                  kpi("High risk · flagged", f"{n_high:,}", value_colour=t.BAD[0]),
                  kpi("Average churn probability", f"{scored['churn_probability'].mean():.1%}")], 3)
            rows = []
            for i, r in scored.head(MAX_ROWS_SHOWN).iterrows():
                _, _, bar = t.RISK[r["risk_band"]]
                prob = (f'<div style="display:flex;align-items:center;gap:10px;justify-content:flex-end">'
                        f'<div style="width:90px;height:6px;background:{t.TRACK};border-radius:3px"><div style="width:{r["churn_probability"] * 100:.0f}%;'
                        f'height:6px;background:{bar};border-radius:3px"></div></div><strong style="width:52px;text-align:right">{r["churn_probability"]:.1%}</strong></div>')
                rows.append([i + 1, r["Geography"], r["Gender"], int(r["Age"]), int(r["NumOfProducts"]),
                             "Yes" if r["IsActiveMember"] else "No", f'{r["Balance"]:,.2f}', prob, chip(r["risk_band"], r["risk_band"])])
            more = f" · showing top {MAX_ROWS_SHOWN} of {len(scored):,}" if len(scored) > MAX_ROWS_SHOWN else ""
            render(card(f'<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;margin-bottom:10px">'
                        f'<h2 class="cg-h2">Ranked by churn probability</h2><span style="font-size:12px;color:{t.MUTED}">{esc(source)}{more}</span></div>'
                        + table(["#", "Country", "Gender", "Age", "Products", "Active", "Balance", "Churn probability", "Risk"],
                                rows, right=(3, 4, 6, 7), raw_cols=(7, 8), min_width=860), "margin-bottom:16px"))
            st.download_button("Download ranked list (CSV)", scored.to_csv(index=False).encode("utf-8"),
                               "churn_predictions.csv", "text/csv")

    with st.expander("Error-handling demo: score an invalid customer"):
        if st.button("Score a customer from 'Italy' aged 150"):
            bad = pd.DataFrame([{**DEFAULT_CUSTOMER, "Geography": "Italy", "Age": 150}])
            try:
                predict_churn(bad, pipeline, threshold)
            except DataValidationError as exc:
                st.error("Rejected by input validation:")
                st.code(str(exc))


# =============================================================================
# 4. Model performance
# =============================================================================
SHORT = {"E1": "Baseline, default settings", "E2": "+ balanced class weights", "E3a": "Random Forest, balanced",
         "E3b": "Gradient Boosting, balanced", "E3c": "MLP neural network (ANN)", "E4a": "+ 4 engineered features",
         "E4b": "− weak features", "E4c": "+ engineered − weak features", "E5": "Grid-search tuned + threshold"}


def _matrix(title: str, m: dict) -> str:
    def cell(v, label, tone):
        fg, bg = (t.RISK[tone][0], t.RISK[tone][1]) if tone else (t.INK, t.BG)
        return (f'<div style="padding:14px;border-radius:9px;background:{bg}"><strong class="cg-num" style="font-family:{t.HEAD};'
                f'font-size:24px;color:{fg}">{int(v):,}</strong><div style="font-size:12px;color:{t.SUBTLE}">{label}</div></div>')
    return card(f'<h2 class="cg-h2">{esc(title)}</h2><div style="display:grid;grid-template-columns:96px repeat(2,minmax(0,1fr));'
                f'gap:8px;font-size:12px;color:{t.MUTED};margin-top:12px"><span></span><span style="text-align:center">Predicted stay</span>'
                f'<span style="text-align:center">Predicted churn</span><span style="align-self:center">Actually stayed</span>'
                f'{cell(m["tn"], "correctly kept", None)}{cell(m["fp"], "false alarms", "Medium")}'
                f'<span style="align-self:center">Actually left</span>{cell(m["fn"], "missed churners", "High")}{cell(m["tp"], "churners caught", "Low")}</div>')


def performance() -> None:
    exp, meta = data.experiments(), data.final_metrics()
    if exp is None or meta is None:
        page_header("Model results", "Model performance")
        st.warning(RUN_FIRST)
        return
    page_header(f"{len(exp)} runs across 5 experiments · held-out test set of "
                f"{int(exp.iloc[0][['test_tn', 'test_fp', 'test_fn', 'test_tp']].sum()):,} customers", "Model performance")
    base = exp.iloc[0]
    b = {k: base[f"test_{k}"] for k in ("tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1", "roc_auc")}
    f = meta["test_metrics"]
    bars = ""
    for name, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]:
        bars += (f'<div style="font-size:13px;margin-top:12px"><div style="display:flex;justify-content:space-between;margin-bottom:6px">'
                 f'<span style="font-weight:700">{name}</span><span class="cg-num" style="color:{t.MUTED}">{b[key]:.3f} → <strong style="color:{t.INK}">{f[key]:.3f}</strong></span></div>'
                 f'<div style="height:7px;background:{t.TRACK};border-radius:4px"><div style="width:{b[key] * 100:.0f}%;height:7px;background:#C6DAD3;border-radius:4px"></div></div>'
                 f'<div style="height:7px;background:{t.TRACK};border-radius:4px;margin-top:4px"><div style="width:{f[key] * 100:.0f}%;height:7px;background:{t.PRIMARY};border-radius:4px"></div></div></div>')
    grid([_matrix(f"Baseline · {base['model']} (t = {base['test_threshold']:.2f})", b),
          _matrix(f"Final · tuned {meta['model']} (t = {f['threshold']:.2f})", f),
          card('<h2 class="cg-h2">Baseline → final</h2>' + legend([("Baseline", "#C6DAD3"), ("Final", t.PRIMARY)]) + bars)], 3)

    rows, highlight = [], None
    for i, r in exp.iterrows():
        if r["run_id"] == "E5":
            highlight = i
        rows.append([r["run_id"], r["model"], SHORT.get(r["run_id"], r["description"]), f'{r["cv_f1"]:.3f}', f'{r["test_accuracy"]:.3f}',
                     f'{r["test_precision"]:.3f}', f'{r["test_recall"]:.3f}', f'{r["test_f1"]:.3f}', f'{r["test_roc_auc"]:.3f}'])
    render(card('<h2 class="cg-h2">Experiment log</h2><p class="cg-sub" style="margin-bottom:10px">Models were chosen on cross-validated F1 from '
                'training data only. Test columns are reported once, for comparison.</p>'
                + table(["Run", "Model", "Change", "CV F1", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows,
                        right=range(3, 9), highlight=highlight, min_width=900), "margin-bottom:16px"))

    params = ", ".join(f"{k} = {v}" for k, v in meta.get("params", {}).items())
    extra = " with 4 engineered features" if meta.get("engineered_features") else ""
    grid([card(f'<h2 class="cg-h2">Why recall comes first</h2><p style="font-size:14px;line-height:1.6;color:{t.SUBTLE};margin:8px 0 0">'
               f'A missed churner costs the bank a customer; a false alarm costs one phone call. Predicting “stays” for everyone is already '
               f'{1 - data.dataset_stats()["churn_rate"]:.1%} accurate, so accuracy alone is misleading.</p>'),
          card(f'<h2 class="cg-h2">Final model settings</h2><p style="font-size:14px;line-height:1.6;color:{t.SUBTLE};margin:8px 0 0">'
               f'{esc(meta["model"])}{extra}: {esc(params)}. Threshold {f["threshold"]:.2f}, chosen to maximise recall with precision ≥ '
               f'{config.THRESHOLD_MIN_PRECISION:.2f} on training folds.</p>')], 2)

    with st.expander("Charts from the pipeline run"):
        for name in ["model_04_roc_pr_curves.png", "model_05_threshold_tuning.png", "model_07_mlp_training_curve.png"]:
            path = config.FIGURES_DIR / name
            if path.exists():
                st.image(str(path), width="stretch")


# =============================================================================
# 5. Data insights
# =============================================================================
LABELS = {"NumOfProducts": "Number of products", "IsActiveMember": "Active member", "Geography": "Country",
          "EstimatedSalary": "Estimated salary", "CreditScore": "Credit score", "HasCrCard": "Has credit card"}


def insights() -> None:
    s = data.dataset_stats()
    imp = data.importance()
    page_header("Dataset audit and exploratory analysis", "Data insights")

    audit = [("Rows × columns", f"{s['rows']:,} × {s['columns']}"), ("Missing values", f"{s['missing']:,}"),
             ("Duplicate rows", f"{s['duplicates']:,}"), ("Duplicate customer IDs", f"{s['duplicate_ids']:,}"),
             ("Churned / stayed", f"{s['churned']:,} / {s['stayed']:,}"), ("Zero-balance customers", data.fmt_pct(s["zero_balance_share"])),
             ("Identifiers removed", ", ".join(config.ID_COLUMNS))]
    audit_html = "".join(f'<div style="display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-top:1px solid {t.TRACK};font-size:14px">'
                         f'<span style="color:{t.MUTED}">{esc(a)}</span><strong style="text-align:right">{esc(b)}</strong></div>' for a, b in audit)
    corr = list(s["correlations"].items())
    corr_rows = [(LABELS.get(k, k), abs(v), f"{v:+.2f}") for k, v in corr[:3]]
    weak = corr[-3:]
    corr_rows.append((", ".join(LABELS.get(k, k) for k, _ in weak), max(abs(v) for _, v in weak), f"≈ {max(abs(v) for _, v in weak):.2f}"))
    rate = s["churn_rate"]
    balance = (f'<h2 class="cg-h2">Class balance</h2><div style="display:flex;height:44px;border-radius:9px;overflow:hidden;font-size:13px;'
               f'font-weight:700;margin-top:14px"><div style="width:{(1 - rate) * 100:.1f}%;background:{t.PRIMARY};color:#FFFFFF;display:flex;'
               f'align-items:center;padding-left:12px">Stayed {1 - rate:.1%}</div><div style="width:{rate * 100:.1f}%;background:{t.RISK["Medium"][2]};'
               f'color:#FFFFFF;display:flex;align-items:center;justify-content:center">{rate:.1%}</div></div>'
               f'<p class="cg-sub" style="margin-top:12px">Imbalanced target. Always guessing “stays” scores {1 - rate:.1%} accuracy, so models '
               f'are judged on recall and F1, with a stratified split.</p><h2 class="cg-h2" style="margin-top:18px">Strongest linear links</h2>'
               + bar_rows(corr_rows, max(r[1] for r in corr_rows), [t.PRIMARY_MID] * 4))
    if imp is not None:
        top = imp.head(6)
        rest = imp.iloc[6:]
        items = [(LABELS.get(r.feature, r.feature), max(r.importance, 0), f"{r.importance:.3f}") for r in top.itertuples()]
        if len(rest):
            items.append((", ".join(LABELS.get(x, x) for x in rest["feature"]), 0.0, "≈ 0"))
        drivers = ('<h2 class="cg-h2">What drives the model</h2><p class="cg-sub">Permutation importance: drop in ROC-AUC when a feature is shuffled</p>'
                   + bar_rows(items, max(i[1] for i in items)))
    else:
        drivers = f'<h2 class="cg-h2">What drives the model</h2><p class="cg-sub">{esc(RUN_FIRST)}</p>'
    grid([card('<h2 class="cg-h2">Dataset audit</h2><div style="margin-top:10px">' + audit_html + "</div>"), card(balance), card(drivers)], 3)

    def pair(title, a, av, b, bv, note):
        return card(f'<h2 class="cg-h2">{title}</h2><div style="display:flex;gap:12px;margin-top:12px">'
                    f'<div style="flex:1;padding:14px;border-radius:9px;background:{t.BG}"><div style="font-size:12px;color:{t.MUTED}">{a}</div>'
                    f'<div class="cg-num" style="font-family:{t.HEAD};font-weight:800;font-size:26px">{av}</div></div>'
                    f'<div style="flex:1;padding:14px;border-radius:9px;background:{t.PRIMARY_SOFT}"><div style="font-size:12px;color:{t.MUTED}">{b}</div>'
                    f'<div class="cg-num" style="font-family:{t.HEAD};font-weight:800;font-size:26px;color:{t.PRIMARY_DARK}">{bv}</div></div></div>'
                    f'<p class="cg-sub" style="margin-top:12px">{note}</p>')
    peak = max(s["age_bands"], key=lambda kv: kv[1])
    grid([pair("Age", "Median age, stayed", f"{s['median_age_stayed']:.0f}", "Median age, churned", f"{s['median_age_churned']:.0f}",
               f"Churn peaks at {peak[1]:.1%} for ages {peak[0]}, versus {s['age_bands'][0][1]:.1%} for under-30s."),
          pair("Engagement", "Active members churn", data.fmt_pct(s["active_rate"]), "Inactive members churn", data.fmt_pct(s["inactive_rate"]),
               "Inactivity nearly doubles the churn rate, and it is a lever the bank can act on."),
          pair("Balance", "Zero balance churn", data.fmt_pct(s["zero_balance_rate"]), "Has balance churn", data.fmt_pct(s["has_balance_rate"]),
               f"{s['zero_balance_share']:.1%} of customers hold zero balance. This motivated the engineered HasZeroBalance feature.")], 3)

    groups = sorted(s["country_rate"].items(), key=lambda kv: -kv[1]) + sorted(s["gender_rate"].items(), key=lambda kv: -kv[1])
    decisions = ("Identifiers dropped: no predictive meaning, and Surname/CustomerId are personal data.",
                 "Decimals kept in Balance and Salary. The original notebook's .astype(int) cut them off for 63% of rows.",
                 "Scaling and one-hot encoding happen inside the model pipeline, fitted on training data only, so no test data leaks in.",
                 "No rows removed: the audit found no missing, duplicate or out-of-range values.")
    grid([card('<h2 class="cg-h2">Churn by country and gender</h2>' + bar_rows([(k, v, data.fmt_pct(v)) for k, v in groups], 0.4)),
          card('<h2 class="cg-h2">Cleaning decisions</h2>' + "".join(
              f'<div style="font-size:14px;line-height:1.6;color:{t.SUBTLE};margin-top:8px">• {esc(d)}</div>' for d in decisions))], 2)

    with st.expander("All exploratory charts (Matplotlib)"):
        figs = sorted(config.FIGURES_DIR.glob("eda_*.png"))
        cols = st.columns(2)
        for i, path in enumerate(figs):
            cols[i % 2].image(str(path), width="stretch")


# =============================================================================
# 6. Fairness & responsible use
# =============================================================================
def fairness() -> None:
    fair = data.fairness()
    meta = data.final_metrics()
    thr = meta["test_metrics"]["threshold"] if meta else config.DEFAULT_THRESHOLD
    page_header(f"Final model on the test set · threshold {thr:.2f}", "Fairness & responsible use")
    render(f'<div role="note" style="border:1px solid #E7C9C4;background:#FBF4F2;border-radius:14px;padding:16px 20px;font-size:14px;'
           f'line-height:1.6;color:{t.SUBTLE};margin-bottom:16px"><strong style="color:{t.BAD[0]}">Classroom model: not ready for deployment.</strong> '
           f'It supports deciding who the retention team calls first. It must never make automated decisions about a customer.</div>')
    if fair is None:
        st.warning(RUN_FIRST)
        return
    rows, bars = [], ""
    for r in fair.itertuples():
        fpr_colour = t.BAD[0] if r.false_positive_rate >= 0.15 else t.INK
        label = "Country" if r.attribute == "Geography" else r.attribute
        rows.append([label, f"<strong>{esc(r.group)}</strong>", f"{r.n:,}", f"{r.actual_churn_rate:.1%}", f"{r.predicted_churn_rate:.1%}",
                     f"{r.recall:.1%}", f"{r.precision:.1%}", f'<strong style="color:{fpr_colour}">{r.false_positive_rate:.1%}</strong>'])
        bars += (f'<div style="display:grid;grid-template-columns:80px minmax(0,1fr);gap:10px;align-items:center;font-size:13px;margin-top:10px">'
                 f'<span style="font-weight:700">{esc(r.group)}</span><div>'
                 f'<div style="display:flex;align-items:center;gap:8px"><div style="height:8px;width:{r.recall * 100:.0f}%;background:{t.PRIMARY};border-radius:4px"></div><span>{r.recall:.0%}</span></div>'
                 f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px"><div style="height:8px;width:{r.false_positive_rate * 100:.0f}%;background:{t.RISK["Medium"][2]};border-radius:4px"></div><span>{r.false_positive_rate:.0%}</span></div></div></div>')
    geo = fair[fair["attribute"] == "Geography"].set_index("group")["false_positive_rate"]
    gen = fair[fair["attribute"] == "Gender"].set_index("group")["false_positive_rate"]
    worst = geo.idxmax()
    ratio = geo.max() / geo.drop(worst).mean()
    more_flagged = gen.idxmax()
    summary = (f"{worst} customers who stay are flagged about {ratio:.1f}× as often as customers in other countries. "
               f"{more_flagged} customers are flagged more often than {gen.idxmin().lower()} customers.")
    left = card('<h2 class="cg-h2">Performance by group</h2><p class="cg-sub" style="margin-bottom:10px">False positive rate = share of '
                'customers who stayed but were flagged anyway. Red marks rates of 15% or more.</p>'
                + table(["Attribute", "Group", "Customers", "Actual churn", "Flagged", "Recall", "Precision", "False positive rate"],
                        rows, right=range(2, 8), raw_cols=(1, 7), min_width=640), "grid-column:span 2", "cg-span2")
    right = card('<h2 class="cg-h2">Recall vs false alarms</h2>' + legend([("Recall", t.PRIMARY), ("False positive rate", t.RISK["Medium"][2])])
                 + bars + f'<p class="cg-sub" style="margin-top:12px">{esc(summary)}</p>')
    render(f'<div class="cg-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin-bottom:16px">'
           f'{left}{right}</div>')

    def lst(title, items, colour):
        li = "".join(f'<li style="margin:0 0 8px">{esc(i)}</li>' for i in items)
        return card(f'<h2 class="cg-h2">{esc(title)}</h2><ul style="margin:12px 0 0;padding-left:18px;font-size:14px;line-height:1.55;color:{t.SUBTLE}">{li}</ul>',
                    f"border-top:4px solid {colour}")
    grid([lst("Use it for", ["Ranking customers for retention calls", "Spotting segments to review, such as inactive or 3+ product holders"], t.RISK["Low"][2]),
          lst("Never use it for", ["Denying services, credit or pricing", "Any automated decision about a customer"], t.RISK["High"][2]),
          lst("Limitations", ["One public dataset, no dates or named bank", "Only 3 countries", "Gender and Country are model inputs",
                              "4-product group has only 60 customers"], t.RISK["Medium"][2]),
          lst("Before real use", ["Validate on the bank’s own recent data", "Fairness review with compliance", "Monitor drift over time",
                                  "A/B-test retention actions"], t.PRIMARY)], 4)
