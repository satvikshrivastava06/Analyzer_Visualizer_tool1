# ui_theme.py — Data Analyzer & Visualizer Tool Premium Design System
# Drop this file next to app.py and call inject_theme() at the top of your app.

import streamlit as st

# ─── Brand tokens ────────────────────────────────────────────────────────────
ACCENT       = "#6C63FF"   # electric indigo — primary CTA
ACCENT_GLOW  = "#6C63FF22" # translucent accent for hover states
SURFACE_1    = "#0D0F14"   # deepest background (page bg)
SURFACE_2    = "#13151C"   # card background
SURFACE_3    = "#1A1D28"   # elevated card / input background
SURFACE_4    = "#21253A"   # hover / active state
BORDER       = "#2A2E45"   # subtle border
BORDER_GLOW  = "#6C63FF55" # accent border on focus/hover
TEXT_1       = "#F0F2FF"   # primary text
TEXT_2       = "#8B90B4"   # secondary / muted
TEXT_3       = "#4B5073"   # placeholder / disabled
SUCCESS      = "#00D4A0"
WARNING      = "#FFB547"
DANGER       = "#FF5F5F"
INFO         = "#38B6FF"
CHART_COLORS = ["#6C63FF","#00D4A0","#FF6B6B","#FFB547","#38B6FF","#C77DFF","#4AE3B5","#FF8C69"]

# ─── Full CSS injection ───────────────────────────────────────────────────────
PREMIUM_CSS = f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Page ── */
html, body, [data-testid="stAppViewContainer"] {{
    background: {SURFACE_1} !important;
    font-family: 'DM Sans', sans-serif;
    color: {TEXT_1};
}}
[data-testid="stAppViewContainer"] > .main {{
    background: {SURFACE_1} !important;
}}
[data-testid="block-container"] {{
    padding: 1.5rem 2rem 4rem !important;
    max-width: 1400px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {SURFACE_2} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{
    color: {TEXT_2} !important;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {{
    color: {TEXT_1} !important;
}}
    [data-testid="stSidebar"] .stMarkdown code {{
        background: {SURFACE_3} !important;
        color: {ACCENT} !important;
        border: 1px solid {BORDER} !important;
        padding: 1px 5px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        z-index: 10;
        overflow: hidden;
    }}

/* ── Page title ── */
h1 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {TEXT_1} !important;
    letter-spacing: -0.02em;
    padding-bottom: 0 !important;
}}
h2 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: {TEXT_1} !important;
    margin-top: 0.5rem !important;
}}
h3 {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: {TEXT_2} !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    flex-wrap: wrap;
}}
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: {TEXT_2} !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease;
    white-space: nowrap;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
    color: {TEXT_1} !important;
    background: {SURFACE_3} !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {TEXT_1} !important;
    background: {ACCENT} !important;
    font-weight: 600 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-testid="stTabContent"] {{
    padding-top: 1.5rem !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: {SURFACE_3} !important;
    color: {TEXT_1} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}}
.stButton > button:hover {{
    background: {SURFACE_4} !important;
    border-color: {ACCENT} !important;
    color: {TEXT_1} !important;
    transform: translateY(-1px);
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}

/* ── Primary action buttons (class="primary-btn") ── */
[data-testid="stButton"][data-primary="true"] > button,
button[kind="primary"] {{
    background: {ACCENT} !important;
    border-color: {ACCENT} !important;
    color: #fff !important;
}}
button[kind="primary"]:hover {{
    background: #7c74ff !important;
    border-color: #7c74ff !important;
}}

/* ── Inputs, selects, textareas ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    background: {SURFACE_3} !important;
    color: {TEXT_1} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    caret-color: {ACCENT};
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_GLOW} !important;
}}
.stSelectbox > div > div > div:hover {{
    border-color: {BORDER_GLOW} !important;
}}

/* ── Labels ── */
.stTextInput label, .stTextArea label, .stSelectbox label,
.stNumberInput label, .stSlider label, .stCheckbox label,
.stRadio label, .stMultiSelect label {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: {TEXT_2} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}

/* ── Slider ── */
.stSlider [data-testid="stSlider"] > div {{
    background: {BORDER} !important;
}}
.stSlider [data-testid="stSlider"] > div > div {{
    background: {ACCENT} !important;
}}

/* ── Dataframes / tables ── */
/* NOTE: Do NOT set background/border-radius on the stDataFrame inner div —
   it paints over the Glide Data Grid canvas and makes the table invisible.
   Streamlit's native dark theme handles dataframe styling correctly. */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
}}

/* ── Code blocks ── */
.stCode, pre, code {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    background: {SURFACE_3} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: #a9b1d6 !important;
}}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.15s;
}}
[data-testid="metric-container"]:hover {{
    border-color: {BORDER_GLOW};
}}
[data-testid="stMetricLabel"] > div {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {TEXT_2} !important;
}}
[data-testid="stMetricValue"] > div {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {TEXT_1} !important;
}}
[data-testid="stMetricDelta"] > div {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}}

/* ── Alert/status boxes ── */
.stAlert {{
    border-radius: 10px !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    background-color: {SURFACE_3} !important;
}}
.stSuccess {{
    border-left: 3px solid {SUCCESS} !important;
    color: {SUCCESS} !important;
}}
.stWarning {{
    border-left: 3px solid {WARNING} !important;
    color: {WARNING} !important;
}}
.stError {{
    border-left: 3px solid {DANGER} !important;
    color: {DANGER} !important;
}}
.stInfo {{
    border-left: 3px solid {INFO} !important;
    color: {INFO} !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {SURFACE_2} !important;
    margin-bottom: 8px;
}}
[data-testid="stExpander"] summary {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: {TEXT_1} !important;
    padding: 12px 16px !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: {SURFACE_3} !important;
    border-radius: 9px;
}}
[data-testid="stExpander"] > div > div {{
    padding: 0 16px 14px !important;
    background: transparent !important;
}}

/* ── DataFrames & Tables ── */
[data-testid="stTable"], .element-container div[data-testid="stTable"] {{
    background-color: {SURFACE_1} !important;
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Selectboxes & Inputs ── */
[data-testid="stSelectbox"] > div {{
    background-color: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

/* ── Divider ── */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 1.5rem 0 !important;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: {TEXT_1} !important;
}}
[data-testid="stChatInput"] > div {{
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
[data-testid="stChatInput"] textarea {{
    background: transparent !important;
    color: {TEXT_1} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE_1}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

/* ── Section headers (custom class) ── */
.davt-section-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px;
    padding: 0 4px;
}}
.davt-section-header .icon {{
    width: 38px;
    height: 38px;
    background: {SURFACE_3};
    border: 1px solid {BORDER};
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.davt-section-header .title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {TEXT_1};
    letter-spacing: -0.01em;
}}
.davt-section-header .subtitle {{
    font-size: 12px;
    color: {TEXT_3};
    margin-left: auto;
    font-weight: 400;
}}

/* ── Stat cards (custom HTML) ── */
.stat-card {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s, transform 0.2s;
}}
.stat-card:hover {{
    border-color: {ACCENT}66;
    transform: translateY(-2px);
}}
.stat-card .label {{
    font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: {TEXT_3}; margin-bottom: 8px;
}}
.stat-card .value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700; color: {TEXT_1};
    line-height: 1;
}}
.stat-card .delta {{
    font-size: 12px; font-weight: 500;
    margin-top: 6px;
}}
.delta-pos {{ color: {SUCCESS}; }}
.delta-neg {{ color: {DANGER}; }}
.delta-neu {{ color: {TEXT_3}; }}

/* ── Tag / badge chips ── */
.chip {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 500;
    border: 1px solid;
}}
.chip-accent {{ background: {ACCENT_GLOW}; border-color: {ACCENT}44; color: {ACCENT}; }}
.chip-success {{ background: {SUCCESS}18; border-color: {SUCCESS}44; color: {SUCCESS}; }}
.chip-warning {{ background: {WARNING}18; border-color: {WARNING}44; color: {WARNING}; }}
.chip-danger {{ background: {DANGER}18; border-color: {DANGER}44; color: {DANGER}; }}
.chip-neutral {{ background: {SURFACE_3}; border-color: {BORDER}; color: {TEXT_2}; }}

/* ── Source pills (data source indicator) ── */
.source-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 6px;
    background: {SURFACE_3}; border: 1px solid {BORDER};
    font-size: 12px; color: {TEXT_2};
    font-family: 'JetBrains Mono', monospace;
}}
.source-pill .dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {SUCCESS}; box-shadow: 0 0 0 2px {SUCCESS}33;
    animation: pulse-dot 2s infinite;
}}
@keyframes pulse-dot {{
    0%, 100% {{ box-shadow: 0 0 0 2px {SUCCESS}33; }}
    50% {{ box-shadow: 0 0 0 4px {SUCCESS}22; }}
}}

/* ── Version history timeline ── */
.version-item {{
    display: flex; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid {BORDER};
    font-size: 12px;
}}
.version-item:last-child {{ border-bottom: none; }}
.version-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {ACCENT}; margin-top: 4px; flex-shrink: 0;
}}
.version-time {{ color: {TEXT_3}; font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
.version-action {{ color: {TEXT_1}; font-weight: 500; }}

/* ── Action card (cleaning operations) ── */
.action-card {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}}
.action-card:hover {{ border-color: {BORDER_GLOW}; }}
.action-card .action-title {{
    font-size: 13px; font-weight: 600; color: {TEXT_1};
    margin-bottom: 4px;
}}
.action-card .action-desc {{
    font-size: 12px; color: {TEXT_2}; line-height: 1.5;
}}

/* ── Model attribution badge ── */
.model-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {SURFACE_3}; border: 1px solid {BORDER};
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; color: {TEXT_3};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
}}
.model-badge .model-name {{ color: {ACCENT}; font-weight: 600; }}

/* ── Page header hero ── */
.page-hero {{
    background: linear-gradient(135deg, {SURFACE_2} 0%, {SURFACE_3} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 16px;
}}
.page-hero .hero-icon {{
    font-size: 2rem; line-height: 1;
    width: 52px; height: 52px;
    background: {ACCENT_GLOW};
    border: 1px solid {ACCENT}44;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.page-hero .hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; font-weight: 600; color: {TEXT_1};
    margin-bottom: 4px;
}}
.page-hero .hero-desc {{
    font-size: 13px; color: {TEXT_2}; line-height: 1.5;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 48px 24px;
    background: {SURFACE_2};
    border: 1px dashed {BORDER};
    border-radius: 12px;
}}
.empty-state .empty-icon {{ font-size: 2.5rem; margin-bottom: 12px; }}
.empty-state .empty-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: {TEXT_1};
    margin-bottom: 6px;
}}
.empty-state .empty-desc {{ font-size: 13px; color: {TEXT_2}; }}

/* ── Tooltip hint ── */
.hint-text {{
    font-size: 11px; color: {TEXT_3};
    display: flex; align-items: center; gap: 5px;
    margin-top: 6px;
}}

/* ── Anomaly row highlight ── */
.anomaly-row {{
    background: {DANGER}10;
    border: 1px solid {DANGER}33;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: flex; align-items: flex-start; gap: 10px;
    font-size: 13px;
}}
.anomaly-row .a-icon {{ color: {DANGER}; font-size: 14px; flex-shrink: 0; }}
.anomaly-row .a-text {{ color: {TEXT_1}; }}
.anomaly-row .a-sub {{ color: {TEXT_2}; font-size: 12px; margin-top: 2px; }}

/* ── Toggle/switch style ── */
[data-testid="stCheckbox"] label {{
    font-size: 13px !important;
    color: {TEXT_1} !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
}}

/* ── AgGrid dark theme override ── */
.ag-theme-streamlit {{
    --ag-background-color: {SURFACE_2} !important;
    --ag-header-background-color: {SURFACE_3} !important;
    --ag-odd-row-background-color: {SURFACE_1} !important;
    --ag-border-color: {BORDER} !important;
    --ag-row-hover-color: {SURFACE_4} !important;
    --ag-foreground-color: {TEXT_1} !important;
    --ag-header-foreground-color: {TEXT_2} !important;
    --ag-font-family: 'DM Sans', sans-serif !important;
    --ag-font-size: 13px !important;
}}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] > div {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
    background: {SURFACE_2};
}}
</style>
"""

# ─── Plotly dark layout ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(19,21,28,0)",
    plot_bgcolor="rgba(19,21,28,0)",
    font=dict(family="DM Sans, sans-serif", size=13, color=TEXT_2),
    title_font=dict(family="Space Grotesk, sans-serif", size=15, color=TEXT_1),
    legend=dict(
        bgcolor="rgba(26,29,40,0.9)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=TEXT_2, size=12),
    ),
    xaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_3, size=11),
        title_font=dict(color=TEXT_2),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_3, size=11),
        title_font=dict(color=TEXT_2),
        zeroline=False,
    ),
    margin=dict(l=16, r=16, t=44, b=16),
    colorway=CHART_COLORS,
)

def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─── Helper: section header ─────────────────────────────────────────────────
import html

def section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<span class="subtitle">{html.escape(subtitle)}</span>' if subtitle else ""
    return f"""<div class="davt-section-header"><div class="icon">{icon}</div><span class="title">{html.escape(title)}</span>{sub_html}</div>""".strip()

# ─── Helper: page hero banner ───────────────────────────────────────────────
def page_hero(icon: str, title: str, desc: str):
    return f"""<div class="page-hero"><div class="hero-icon">{icon}</div><div><div class="hero-title">{html.escape(title)}</div><div class="hero-desc">{html.escape(desc)}</div></div></div>""".strip()

# ─── Helper: empty state ────────────────────────────────────────────────────
def empty_state(icon: str, title: str, desc: str):
    return f"""<div class="empty-state"><div class="empty-icon">{icon}</div><div class="empty-title">{html.escape(title)}</div><div class="empty-desc">{html.escape(desc)}</div></div>""".strip()

# ─── Helper: KPI stat card ──────────────────────────────────────────────────
def stat_card(label: str, value: str, delta: str = "", delta_type: str = "neu", **kwargs):
    final_type = kwargs.get("delta_color", delta_type)
    if final_type == "inverse": final_type = "neg"
    if final_type == "normal": final_type = "pos"
    delta_html = f'<div class="delta delta-{final_type}">{html.escape(delta)}</div>' if delta else ""
    return f"""<div class="stat-card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div>{delta_html}</div>""".strip()

# ─── Helper: chip / badge ───────────────────────────────────────────────────
def chip(text: str, kind: str = "neutral") -> str:
    return f'<span class="chip chip-{kind}">{html.escape(text)}</span>'.strip()

# ─── Helper: source indicator pill ─────────────────────────────────────────
def source_pill(source_name: str) -> str:
    return f'<span class="source-pill"><span class="dot"></span>{html.escape(source_name)}</span>'.strip()

# ─── Helper: status / anomaly row ───────────────────────────────────────────
def anomaly_row(message: str, kind: str = "warning", detail: str = ""):
    icon_map = {"success": "✔", "warning": "⚠", "danger": "✖", "info": "ℹ"}
    color_map = {"success": SUCCESS, "warning": WARNING, "danger": DANGER, "info": INFO}
    icon = icon_map.get(kind, "⚠")
    color = color_map.get(kind, WARNING)
    detail_html = f"<div class='a-sub'>{html.escape(detail)}</div>" if detail else ""
    return f"""<div class="anomaly-row" style="background:{color}10; border-color:{color}33;"><div class="a-icon" style="color:{color}">{icon}</div><div><div class="a-text">{html.escape(message)}</div>{detail_html}</div></div>""".strip()

# ─── Helper: action card wrapper ────────────────────────────────────────────
def action_card(category: str, title: str, desc: str):
    return f"""<div class="action-card"><div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 4px;"><div class="action-title">{html.escape(title)}</div><div class="chip chip-accent" style="font-size:10px; padding:2px 8px;">{html.escape(category)}</div></div><div class="action-desc">{html.escape(desc)}</div></div>""".strip()

# ── Design tokens (Data Analyzer & Visualizer Tool Theme) ──────────────────────
def model_badge(model_name: str):
    short = model_name.replace("models/", "").replace("groq/", "⚡ ")
    return f'<div class="model-badge">▲ powered by <span class="model-name">{html.escape(short)}</span></div>'.strip()

# ─── Main inject function — call once at top of app.py ──────────────────────
def inject_theme():
    """Inject the full Data Analyzer & Visualizer Tool premium CSS. Call once after st.set_page_config()."""
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
