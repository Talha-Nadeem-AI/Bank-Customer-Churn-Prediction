"""Smoke tests for the Streamlit app and its HTML helpers."""
import pytest

from src.churn import config
from src.churn.ui.components import esc, gauge, table


def test_escape_blocks_html_injection():
    assert esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_table_escapes_cells_but_keeps_trusted_columns():
    html = table(["A", "B"], [["<b>x</b>", "<strong>ok</strong>"]], raw_cols=(1,))
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<strong>ok</strong>" in html


@pytest.mark.parametrize("p, shown", [(-0.5, "0.0%"), (0.958, "95.8%"), (1.7, "100.0%")])
def test_gauge_clamps_probability(p, shown):
    assert shown in gauge(p, "High")


def test_app_overview_runs_without_errors():
    """Runs the real app (default page) headlessly; needs the trained model and results."""
    if not config.FINAL_MODEL_PATH.exists():
        pytest.skip("Run `python main.py` first")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Overview" in text and "ChurnGuard" in text
