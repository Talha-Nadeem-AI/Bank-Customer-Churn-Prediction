"""Small HTML building blocks used by the pages.

Every value that could come from a user's file is passed through ``esc`` so
an uploaded CSV can never inject HTML into the page.
"""
from __future__ import annotations

import html
from typing import Iterable, List, Optional, Sequence, Tuple

import streamlit as st

from . import theme as t


def esc(value) -> str:
    """HTML-escape any value for safe display."""
    return html.escape(str(value))


def render(markup: str) -> None:
    """Render an HTML block. Newlines are removed so Markdown never turns it into a code block."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def card(inner: str, style: str = "", cls: str = "") -> str:
    return f'<div class="cg-card {cls}" style="{style}">{inner}</div>'


def page_header(eyebrow: str, title: str, right: str = "") -> None:
    render(f"""
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:16px;margin-bottom:22px">
      <div><div class="cg-eyebrow">{esc(eyebrow)}</div><h1 class="cg-title">{esc(title)}</h1></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">{right}</div>
    </div>""")


def pill(text: str) -> str:
    return (f'<span style="padding:8px 12px;border-radius:999px;background:{t.CARD};border:1px solid #DADFD8;'
            f'font-size:13px;color:{t.SUBTLE}">{esc(text)}</span>')


def kpi(label: str, value: str, note: str = "", note_colour: Optional[str] = None, value_colour: str = t.INK) -> str:
    note_html = ""
    if note:
        colour = note_colour or t.MUTED
        weight = 700 if note_colour else 500
        note_html = f'<div style="font-size:12px;color:{colour};margin-top:4px;font-weight:{weight}">{esc(note)}</div>'
    return card(f'<div class="cg-label">{esc(label)}</div>'
                f'<div class="cg-kpi" style="color:{value_colour}">{esc(value)}</div>{note_html}')


def grid(items: Sequence[str], columns: int, gap: int = 16, margin_bottom: int = 16) -> None:
    render(f'<div class="cg-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr));gap:{gap}px;'
           f'margin-bottom:{margin_bottom}px">{"".join(items)}</div>')


def bar_rows(items: Iterable[Tuple[str, float, str]], max_value: float,
             colours: Optional[Sequence[str]] = None) -> str:
    """Horizontal bars: (label, value, display text)."""
    out = []
    for i, (label, value, shown) in enumerate(items):
        width = 0 if max_value <= 0 else max(0.0, min(100.0, value / max_value * 100))
        colour = (colours[i] if colours else t.PRIMARY)
        out.append(
            f'<div style="font-size:13px;margin-top:12px"><div style="display:flex;justify-content:space-between;gap:8px">'
            f'<span>{esc(label)}</span><strong class="cg-num">{esc(shown)}</strong></div>'
            f'<div style="height:8px;background:{t.TRACK};border-radius:4px;margin-top:6px">'
            f'<div style="width:{width:.1f}%;height:8px;background:{colour};border-radius:4px"></div></div></div>')
    return "".join(out)


def column_chart(items: Iterable[Tuple[str, float, str]], max_value: float, height: int = 220,
                 colours: Optional[Sequence[str]] = None) -> str:
    """Vertical bars: (caption under bar, value, label above bar)."""
    items = list(items)
    bars, captions = [], []
    for i, (caption, value, shown) in enumerate(items):
        h = 0 if max_value <= 0 else max(1.0, min(100.0, value / max_value * 92))
        colour = colours[i] if colours else t.PRIMARY
        bars.append(f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                    f'height:100%;gap:6px"><span class="cg-num" style="font-weight:700;font-size:14px">{esc(shown)}</span>'
                    f'<div style="width:100%;height:{h:.1f}%;background:{colour};border-radius:6px 6px 0 0"></div></div>')
        captions.append(f'<span style="flex:1;text-align:center">{esc(caption)}</span>')
    return (f'<div style="display:flex;align-items:flex-end;gap:28px;height:{height}px;padding:0 12px;'
            f'border-bottom:1px solid {t.BORDER}">{"".join(bars)}</div>'
            f'<div style="display:flex;gap:28px;padding:8px 12px 0;font-size:12px;color:{t.MUTED}">{"".join(captions)}</div>')


def chip(text: str, risk: str) -> str:
    fg, bg, _ = t.RISK.get(risk, (t.SUBTLE, t.TRACK, t.MUTED))
    return f'<span class="cg-chip" style="background:{bg};color:{fg}">{esc(text)}</span>'


def stat_row(label: str, value: str, tone: Optional[str] = None) -> str:
    """Coloured row used for test-set outcomes (tone = Low / Medium / High / None)."""
    if tone:
        fg, bg, _ = t.RISK[tone]
    else:
        fg, bg = t.INK, t.BG
    return (f'<div style="display:flex;justify-content:space-between;padding:12px 14px;border-radius:9px;'
            f'background:{bg};margin-top:10px"><span style="font-size:14px">{esc(label)}</span>'
            f'<strong class="cg-num" style="color:{fg}">{esc(value)}</strong></div>')


def table(headers: Sequence[str], rows: List[Sequence[str]], right: Iterable[int] = (),
          highlight: Optional[int] = None, min_width: int = 0, raw_cols: Iterable[int] = ()) -> str:
    """HTML table. Cells are escaped unless their column is listed in ``raw_cols`` (trusted HTML)."""
    right, raw_cols = set(right), set(raw_cols)
    head = "".join(f'<th class="{"r" if i in right else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = []
    for r_i, row in enumerate(rows):
        style = f' style="background:#EEF4F1"' if r_i == highlight else ""
        cells = "".join(
            f'<td class="{"r" if i in right else ""}">{cell if i in raw_cols else esc(cell)}</td>'
            for i, cell in enumerate(row))
        body.append(f"<tr{style}>{cells}</tr>")
    return (f'<div class="cg-scroll"><table class="cg-table" style="min-width:{min_width}px">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def gauge(probability: float, risk: str) -> str:
    """Half-circle gauge showing a probability between 0 and 1."""
    p = max(0.0, min(1.0, float(probability)))
    arc = 3.14159 * 90
    fg, _, bar = t.RISK[risk]
    return (f'<div style="display:flex;flex-direction:column;align-items:center">'
            f'<svg width="260" height="150" viewBox="0 0 220 125" role="img" aria-label="Churn probability {p:.1%}">'
            f'<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="{t.TRACK}" stroke-width="18" stroke-linecap="round"></path>'
            f'<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="{bar}" stroke-width="18" stroke-linecap="round" '
            f'stroke-dasharray="{p * arc:.1f} 400"></path></svg>'
            f'<div class="cg-num" style="margin-top:-78px;font-family:{t.HEAD};font-weight:800;font-size:44px;color:{fg}">{p:.1%}</div>'
            f'<div style="font-size:13px;color:{t.MUTED}">probability of leaving the bank</div></div>')


def callout(title: str, text: str, tone: str = "High") -> str:
    fg, bg, _ = t.RISK[tone]
    border = {"High": "#E7C9C4", "Medium": "#E9D8B5", "Low": "#CFE2D5"}[tone]
    return (f'<div style="padding:12px 14px;border-radius:9px;border:1px solid {border};background:{bg};'
            f'font-size:13px;line-height:1.55;margin-top:10px"><strong style="color:{fg}">{esc(title)}</strong> {esc(text)}</div>')


def legend(items: Sequence[Tuple[str, str]]) -> str:
    return ('<div style="display:flex;gap:14px;font-size:12px;color:%s;margin-top:8px">' % t.MUTED + "".join(
        f'<span style="display:flex;align-items:center;gap:6px"><span style="width:10px;height:10px;border-radius:2px;'
        f'background:{c}"></span>{esc(label)}</span>' for label, c in items) + "</div>")
