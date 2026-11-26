"""First-launch screen: shown when no trained model exists yet (e.g. a fresh
deployment, or a tester who opened the app before running `python main.py`)."""
from __future__ import annotations

import streamlit as st

from .. import config
from . import theme as t
from .components import card, esc, render


def setup_screen(problem: str) -> None:
    """Explain what is missing and offer to train the model from the app."""
    render(f'<div class="cg-eyebrow">First launch</div><h1 class="cg-title">Set up ChurnGuard</h1>')
    render(card(
        f'<h2 class="cg-h2">The model has not been trained on this computer yet</h2>'
        f'<p class="cg-sub" style="margin-top:8px">{esc(problem)}</p>'
        f'<p style="font-size:14px;line-height:1.6;color:{t.SUBTLE};margin:12px 0 0">Training reads '
        f'<code>{esc(config.default_data_path().relative_to(config.PROJECT_ROOT))}</code>, runs the experiments, saves the '
        f'best model and writes every chart and table the app shows. The quick option below uses a smaller tuning grid '
        f'and takes about 2–3 minutes. For the full run (about 4–5 minutes) use <code>python main.py</code> in a terminal.</p>',
        "margin:18px 0"))
    if st.button("Train the model now (quick, about 2–3 minutes)", type="primary"):
        from main import run_pipeline, setup_logging  # project-root script; imported only when needed

        setup_logging()
        config.N_JOBS = 1  # one core: stays inside the ~1 GB memory limit of free cloud hosting
        with st.spinner("Training: cleaning data, running 5 experiments, tuning the final model…"):
            run_pipeline(quick=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Done. Loading the dashboard…")
        st.rerun()
