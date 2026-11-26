"""ChurnGuard: Streamlit app for the bank customer churn model.

Run from the project root, after `python main.py` has trained the model:

    streamlit run app.py

Pages (left sidebar):
  Analyse  - Overview, Data insights
  Predict  - Predict a customer, Batch scoring
  Govern   - Model performance, Fairness & responsible use
"""
from __future__ import annotations

import streamlit as st

from src.churn.predict import ModelLoadError
from src.churn.ui import pages, sidebar
from src.churn.ui.setup import setup_screen
from src.churn.ui import data as ui_data
from src.churn.ui.theme import CSS

st.set_page_config(page_title="ChurnGuard · Bank Churn Predictor", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

# (callable, title, Material icon name, url path)
PAGES = {
    "overview": st.Page(pages.overview, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    "insights": st.Page(pages.insights, title="Data insights", icon=":material/bar_chart:", url_path="insights"),
    "predict": st.Page(pages.predict, title="Predict a customer", icon=":material/person_search:", url_path="predict"),
    "batch": st.Page(pages.batch, title="Batch scoring", icon=":material/format_list_numbered:", url_path="batch"),
    "performance": st.Page(pages.performance, title="Model performance", icon=":material/target:", url_path="performance"),
    "fairness": st.Page(pages.fairness, title="Fairness & use", icon=":material/balance:", url_path="fairness"),
}
GROUPS = [
    ("Analyse", [(PAGES["overview"], "Overview", "space_dashboard"), (PAGES["insights"], "Data insights", "bar_chart")]),
    ("Predict", [(PAGES["predict"], "Predict a customer", "person_search"),
                 (PAGES["batch"], "Batch scoring", "format_list_numbered")]),
    ("Govern", [(PAGES["performance"], "Model performance", "target"), (PAGES["fairness"], "Fairness & use", "balance")]),
]

current = st.navigation(list(PAGES.values()), position="hidden")

# ---- Sidebar ---------------------------------------------------------------
model_problem = None
try:
    _, threshold, meta = ui_data.model_bundle()
except FileNotFoundError:
    threshold, meta, model_problem = 0.5, None, "No saved model was found in the models/ folder."
except ModelLoadError as exc:
    threshold, meta, model_problem = 0.5, None, str(exc)
try:
    stats = ui_data.dataset_stats()
except (FileNotFoundError, ValueError):
    stats = None

with st.sidebar:
    sidebar.brand()
    sidebar.model_status(meta)
    sidebar.navigation(GROUPS, current.title)
    sidebar.portfolio(stats)
    sidebar.risk_legend(threshold)
    sidebar.team()
    sidebar.footer(stats)

if stats is None:
    st.error("Dataset not found. Put Churn_Modelling.csv in data/raw/ (see README), then run `python main.py`.")
    st.stop()
if model_problem:
    setup_screen(model_problem)
    st.stop()

current.run()
