"""Visual theme for the ChurnGuard app: colour tokens and the global CSS.

Fonts: Bricolage Grotesque (headings, figures) + Nunito (body), from Google
Fonts. If the computer is offline the app falls back to the system sans-serif.
Colour is used only to signal state (risk level, met/not met).
"""
from __future__ import annotations

# ---- Colour tokens ---------------------------------------------------------
INK = "#1D2522"          # main text
MUTED = "#5E6B66"        # secondary text
SUBTLE = "#3E4A45"       # body text on cards
BG = "#F1F2EF"           # page background
CARD = "#FFFFFF"
BORDER = "#E1E4DE"
TRACK = "#ECEEEA"        # empty bar track
PRIMARY = "#2F6B62"      # muted teal
PRIMARY_DARK = "#234F49"
PRIMARY_SOFT = "#E6F0EC"
PRIMARY_MID = "#5C8F85"
PRIMARY_LIGHT = "#9DBDB4"
BRASS = "#C9A55C"
SIDEBAR = "#17241F"
SIDEBAR_PANEL = "#1F2F28"
SIDEBAR_LINE = "#2E4038"
SIDEBAR_ACTIVE = "#2A3B33"
SIDEBAR_TEXT = "#C3CDC7"
SIDEBAR_MUTED = "#9DAAA3"

# Risk / state colours: (text, soft background, bar)
RISK = {
    "Low": ("#2F6B4A", "#E8F1EB", "#4B8A63"),
    "Medium": ("#8A5A12", "#F7EEDD", "#B7832F"),
    "High": ("#9B3030", "#F6E7E5", "#A53A35"),
}
GOOD = RISK["Low"]
WARN = RISK["Medium"]
BAD = RISK["High"]

FONT_URL = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;"
            "12..96,700;12..96,800&family=Nunito:wght@400;500;600;700;800&display=swap")
HEAD = "'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif"
BODY = "Nunito, 'Segoe UI', system-ui, sans-serif"

CSS = f"""
<style>
@import url('{FONT_URL}');

/* ---------- page ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {BG}; color: {INK}; font-family: {BODY};
}}
.stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button, .stApp select,
.stApp [data-baseweb="select"] * {{ font-family: {BODY}; }}
.stApp h1, .stApp h2, .stApp h3 {{ font-family: {HEAD}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"], footer {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding: 2rem 2.5rem 3rem; }}
.cg-num {{ font-variant-numeric: tabular-nums; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: {SIDEBAR}; min-width: 280px !important; max-width: 280px !important; }}
[data-testid="stSidebar"] > div {{ background: {SIDEBAR}; }}
[data-testid="stSidebarContent"] {{ padding: 0.4rem 0.9rem 1.2rem; }}
[data-testid="stSidebarHeader"] {{ height: 0.5rem; padding: 0; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.25rem; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] {{ margin: 0 !important; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
  border-radius: 10px; padding: 0.55rem 0.75rem; min-height: 0; margin: 0;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background: #22342C; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a p,
[data-testid="stSidebar"] [data-testid="stPageLink"] a span {{
  color: {SIDEBAR_TEXT} !important; font-weight: 600; font-size: 14px;
}}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarCollapsedControl"] svg {{ color: {SIDEBAR_TEXT}; }}

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 10px; min-height: 44px; font-weight: 700; font-family: {BODY};
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
  background: {PRIMARY}; border-color: {PRIMARY}; color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK}; color: #FFFFFF;
}}
.stButton > button[kind="secondary"], .stDownloadButton > button {{
  background: {CARD}; border: 1px solid #C8CEC6; color: {INK};
}}
[data-testid="stForm"] {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.4rem 1.5rem;
}}
[data-testid="stFileUploaderDropzone"] {{
  background: {CARD}; border: 2px dashed #B7C0B9; border-radius: 14px;
}}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---------- override Streamlit's Markdown sizes for our own blocks ---------- */
[data-testid="stMarkdownContainer"] .cg-h2 {{ font-size: 17px !important; font-weight: 700 !important; padding: 0 !important;
  line-height: 1.3 !important; letter-spacing: 0 !important; }}
[data-testid="stMarkdownContainer"] .cg-title {{ font-size: 30px !important; font-weight: 800 !important; padding: 0 !important;
  line-height: 1.2 !important; }}
[data-testid="stMarkdownContainer"] p.cg-sub {{ font-size: 13px !important; }}
[data-testid="stMarkdownContainer"] .cg-card p {{ margin-bottom: 0; }}
[data-testid="stHeaderActionElements"], [data-testid="stMarkdownContainer"] h1 a[href^="#"], [data-testid="stMarkdownContainer"] h2 a[href^="#"] {{ display: none !important; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"] {{ margin: 0 !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}

/* ---------- custom blocks ---------- */
.cg-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; }}
.cg-h2 {{ margin: 0; font-family: {HEAD}; font-size: 17px; font-weight: 700; color: {INK}; }}
.cg-sub {{ margin: 4px 0 0; font-size: 13px; color: {MUTED}; line-height: 1.5; }}
.cg-label {{ font-size: 13px; color: {MUTED}; }}
.cg-kpi {{ font-family: {HEAD}; font-weight: 800; font-size: 32px; color: {INK}; margin-top: 6px;
          font-variant-numeric: tabular-nums; line-height: 1.15; }}
.cg-table {{ width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }}
.cg-table, .cg-table tr, .cg-table th, .cg-table td {{ border: none !important; background: transparent; }}
.cg-table th {{ text-align: left; color: {MUTED}; font-size: 12px; font-weight: 700; padding: 8px 10px; }}
.cg-table td {{ padding: 11px 10px !important; border-top: 1px solid {TRACK} !important; color: {INK}; }}
.cg-table th {{ padding: 8px 10px !important; }}
.cg-table .r {{ text-align: right; }}
.cg-scroll {{ overflow-x: auto; }}
.cg-chip {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 800; font-size: 12px; }}
.cg-eyebrow {{ font-size: 13px; color: {MUTED}; font-weight: 600; }}
.cg-title {{ margin: 2px 0 0; font-family: {HEAD}; font-weight: 800; font-size: 30px; letter-spacing: -0.5px; color: {INK}; }}
.cg-grid {{ display: grid; gap: 16px; }}
@media (max-width: 900px) {{ .cg-grid {{ grid-template-columns: 1fr !important; }} .cg-span2 {{ grid-column: auto !important; }} }}
</style>
"""
