"""The ChurnGuard sidebar: brand, model status, grouped navigation, portfolio churn,
risk legend and dataset/course footer."""
from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from .. import config
from . import theme as t
from .components import esc, render


def _panel(inner: str, margin_top: int = 12) -> str:
    return (f'<div style="border-radius:14px;background:{t.SIDEBAR_PANEL};border:1px solid {t.SIDEBAR_LINE};'
            f'padding:14px;margin-top:{margin_top}px">{inner}</div>')


def brand() -> None:
    render(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:6px 4px 14px">
      <div style="width:40px;height:40px;border-radius:11px;background:#22342C;border:1px solid #34483F;display:flex;align-items:center;justify-content:center">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{t.BRASS}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"></path><path d="M7.5 14l3-3 2 2 4-4.5"></path></svg>
      </div>
      <div><div style="font-family:{t.HEAD};font-weight:800;font-size:20px;color:#FFFFFF;letter-spacing:-0.3px;line-height:1.1">ChurnGuard</div>
      <div style="font-size:12px;color:{t.SIDEBAR_MUTED};margin-top:2px">Bank retention analytics</div></div>
    </div>""")


def model_status(meta: Dict | None) -> None:
    if meta is None:
        render(_panel(f'<div style="font-size:11px;font-weight:800;color:#E3A49F;letter-spacing:0.6px">MODEL NOT TRAINED</div>'
                      f'<div style="font-size:12px;color:{t.SIDEBAR_MUTED};margin-top:4px">Run <code>python main.py</code></div>', 0))
        return
    tm = meta.get("test_metrics", {})
    recall, auc = tm.get("recall", 0.0), tm.get("roc_auc", 0.0)
    circ = 2 * 3.14159 * 15
    render(_panel(f"""
    <div style="display:flex;gap:12px;align-items:center">
      <svg width="54" height="54" viewBox="0 0 36 36" role="img" aria-label="Recall {recall:.0%}"><circle cx="18" cy="18" r="15" fill="none" stroke="{t.SIDEBAR_LINE}" stroke-width="4"></circle><circle cx="18" cy="18" r="15" fill="none" stroke="{t.BRASS}" stroke-width="4" stroke-linecap="round" stroke-dasharray="{recall * circ:.1f} {circ:.1f}" transform="rotate(-90 18 18)"></circle><text x="18" y="21" text-anchor="middle" font-size="9" font-weight="800" fill="#FFFFFF" font-family="Nunito, sans-serif">{recall:.0%}</text></svg>
      <div style="min-width:0">
        <div style="display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:800;color:#9FD3B4;letter-spacing:0.6px"><span style="width:7px;height:7px;border-radius:50%;background:#6FBF8E"></span>MODEL READY</div>
        <div style="font-size:14px;font-weight:700;color:#FFFFFF;margin-top:3px">{esc(meta.get("model", "Model"))}</div>
        <div style="font-size:12px;color:{t.SIDEBAR_MUTED}">Recall {recall:.1%} · AUC {auc:.3f}</div>
      </div>
    </div>""", 0))


def navigation(groups: List[Tuple[str, List[Tuple[object, str, str]]]], current_title: str) -> None:
    """Grouped links. The current page is drawn highlighted; the others are st.page_link."""
    for group, items in groups:
        render(f'<div style="font-size:11px;font-weight:800;letter-spacing:1.4px;color:#7F8F87;padding:14px 12px 2px">{esc(group.upper())}</div>')
        for page, title, icon in items:
            if title == current_title:
                render(f'<div aria-current="page" style="display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:10px;'
                       f'background:{t.SIDEBAR_ACTIVE};color:#FFFFFF;font-weight:700;font-size:14px;box-shadow:inset 3px 0 0 {t.BRASS};min-height:22px">'
                       f'<span style="font-family:\'Material Symbols Rounded\';font-size:20px;line-height:1;color:{t.BRASS}">{icon}</span>{esc(title)}</div>')
            else:
                st.page_link(page, label=title, icon=f":material/{icon}:")


def portfolio(stats: Dict | None) -> None:
    if not stats:
        return
    churned, stayed, rate = stats["churned"], stats["stayed"], stats["churn_rate"]
    render(_panel(f"""
    <div style="display:flex;justify-content:space-between;align-items:baseline"><span style="font-size:12px;color:{t.SIDEBAR_MUTED};font-weight:600">Portfolio churn</span>
    <span style="font-family:{t.HEAD};font-size:20px;font-weight:700;color:#FFFFFF">{rate:.1%}</span></div>
    <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:10px"><div style="width:{(1 - rate) * 100:.1f}%;background:#4E7A6E"></div><div style="width:{rate * 100:.1f}%;background:{t.BRASS}"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:{t.SIDEBAR_MUTED};margin-top:8px"><span>{stayed:,} stayed</span><span>{churned:,} left</span></div>
    """, 18))


def risk_legend(threshold: float) -> None:
    low = config.LOW_RISK_CUTOFF
    rows = [("#6FBF8E", "Low", f"&lt; {low:.2f}"), ("#D9A54A", "Medium", f"{low:.2f}–{threshold:.2f}"),
            ("#D9706A", "High · flagged", f"≥ {threshold:.2f}")]
    body = "".join(f'<div style="display:flex;align-items:center;gap:8px;margin-top:7px"><span style="width:9px;height:9px;border-radius:50%;background:{c}"></span>'
                   f'{n}<span style="margin-left:auto;color:{t.SIDEBAR_MUTED}">{v}</span></div>' for c, n, v in rows)
    render(f'<div style="border-radius:14px;border:1px solid {t.SIDEBAR_LINE};padding:12px 14px;margin-top:10px;font-size:12px;color:#C9D2CC">'
           f'<div style="font-size:11px;font-weight:800;letter-spacing:1.2px;color:#7F8F87">RISK LEVELS</div>{body}</div>')


def team() -> None:
    """Names, student IDs and roles of the project team."""
    rows = "".join(
        f'<div style="margin-top:8px"><div style="color:#FFFFFF;font-weight:700;font-size:12px">{esc(m["name"])} '
        f'<span style="color:{t.SIDEBAR_MUTED};font-weight:600">· {esc(m["id"])}</span></div>'
        f'<div style="color:{t.SIDEBAR_MUTED};font-size:11px">{esc(m["role"])}</div></div>'
        for m in config.TEAM)
    render(f'<div style="border-radius:14px;border:1px solid {t.SIDEBAR_LINE};padding:12px 14px;margin-top:10px">'
           f'<div style="font-size:11px;font-weight:800;letter-spacing:1.2px;color:#7F8F87">PROJECT TEAM</div>{rows}</div>')


def footer(stats: Dict | None) -> None:
    n = f'{stats["rows"]:,} customers' if stats else "dataset not loaded"
    render(f'<div style="padding:18px 6px 0;font-size:11px;line-height:1.6;color:#8D9A93">'
           f'<span style="display:inline-block;padding:3px 8px;border-radius:6px;background:#22342C;color:#C9D2CC;font-family:ui-monospace,Menlo,monospace">'
           f'{esc(config.default_data_path().name)}</span><div style="margin-top:6px">{n} · seed {config.RANDOM_STATE}</div>'
           f'<div>CUSIT · Programming for AI · Fall 2026</div></div>')
