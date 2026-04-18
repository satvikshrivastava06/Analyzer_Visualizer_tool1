# Data Analyzer & Visualizer Tool v2.1 — Refactored Design System
# Unauthorized copying, modification, or distribution is strictly prohibited.

import streamlit as st
import pandas as pd
import duckdb
import os
import datetime
from typing import cast

# --- Setup ---
st.set_page_config(
    page_title="Data Analyzer & Visualizer Tool — AI Analytics Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui_theme import (
    inject_theme, apply_plotly_theme, section_header, page_hero,
    empty_state, stat_card, chip, source_pill, anomaly_row,
    action_card, model_badge, CHART_COLORS, PLOTLY_LAYOUT,
    ACCENT, SURFACE_2, SURFACE_3, BORDER, TEXT_1, TEXT_2, TEXT_3,
    SUCCESS, WARNING, DANGER, INFO,
)
inject_theme()

# --- Custom Modules ---
from modules.ai_engine import safe_ai_call, validate_sql_query, log_version_action, get_ai_missions
from modules.ingestion import load_from_upload, load_from_url

# 1. Initialize API Keys and DB Connection
if 'db_conn' not in st.session_state:
    db_conn = duckdb.connect(':memory:')
    db_name = "local"
    try:
        md = st.secrets.get("motherduck", {})
        if md.get("token"):
            os.environ["MOTHERDUCK_TOKEN"] = md["token"]
            db_conn = duckdb.connect(f'md:{md.get("db_name","my_db")}')
            db_name = md.get("db_name","my_db")
    except Exception:
         pass
    st.session_state.db_conn = db_conn
    st.session_state.db_name = db_name

conn = st.session_state.db_conn
db_name = st.session_state.db_name

# Optional Gemini Key
gemini_key = st.secrets.get("gemini", {}).get("api_key")

# Optional Groq Key
groq_key = st.secrets.get("groq", {}).get("api_key")

# Initialize session state for data
if 'current_table' not in st.session_state:
    try:
        table_check = conn.execute(f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{db_name}' AND table_name = 'executive_kpi_data'").fetchone()
        if table_check:
            st.session_state.current_table = "executive_kpi_data"
            st.session_state.df_preview = conn.execute("SELECT * FROM executive_kpi_data LIMIT 1000").df()
        else:
            st.session_state.current_table = None
    except Exception:
        st.session_state.current_table = None

if 'df_preview' not in st.session_state:
    st.session_state.df_preview = None

if 'version_history' not in st.session_state:
    st.session_state.version_history = []
if 'ai_recommendations' not in st.session_state:
    st.session_state.ai_recommendations = []
if 'generated_rec_charts' not in st.session_state:
    st.session_state.generated_rec_charts = {}

# Visualization Launch Flags
if 'viz_engine_launched' not in st.session_state:
    st.session_state.viz_engine_launched = False
if 'pyg_launched' not in st.session_state:
    st.session_state.pyg_launched = False
if 'echarts_launched' not in st.session_state:
    st.session_state.echarts_launched = False

# Reset flags if dataframe changes
if 'last_table' not in st.session_state:
    st.session_state.last_table = st.session_state.current_table

if st.session_state.last_table != st.session_state.current_table:
    st.session_state.viz_engine_launched = False
    st.session_state.pyg_launched = False
    st.session_state.echarts_launched = False
    st.session_state.last_table = st.session_state.current_table

# Sidebar: Governance & Lineage (v2 Style)
with st.sidebar:
    st.markdown(
        """<div style="padding:4px 0 16px"><div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:#F0F2FF;letter-spacing:-0.02em">⬡ Data Analyzer & Visualizer Tool</div><div style="font-size:11px;color:#4B5073;margin-top:2px">AI Analytics Platform</div></div>""".strip(),
        unsafe_allow_html=True,
    )

    # Connection status
    conn_ok = conn is not None
    if conn_ok:
        status_html = f"""<div style="background:{SURFACE_3};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;margin-bottom:12px"><div style="display:flex;align-items:center;gap:7px"><div style="width:7px;height:7px;border-radius:50%;background:{SUCCESS};box-shadow:0 0 0 3px {SUCCESS}33"></div><span style="font-size:12px;color:{TEXT_2};font-weight:500">Connected · {db_name}</span></div></div>""".strip()
    else:
        status_html = f"""<div style="background:{SURFACE_3};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;margin-bottom:12px"><div style="display:flex;align-items:center;gap:7px"><div style="width:7px;height:7px;border-radius:50%;background:{DANGER};box-shadow:0 0 0 3px {DANGER}33"></div><span style="font-size:12px;color:{TEXT_2};font-weight:500">Disconnected</span></div></div>""".strip()
    
    st.markdown(status_html, unsafe_allow_html=True)

    # Active dataset
    if st.session_state.df_preview is not None:
        df_s = st.session_state.df_preview
        rows, cols_ = df_s.shape
        st.markdown(
            f"""<div style="background:{SURFACE_3};border:1px solid {BORDER};
                    border-radius:8px;padding:10px 12px;margin-bottom:16px">
                <div style="font-size:11px;color:{TEXT_3};text-transform:uppercase;
                            letter-spacing:.06em;margin-bottom:6px">Active dataset</div>
                <div style="display:flex;gap:16px">
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;
                                    font-size:1.1rem;font-weight:700;color:{TEXT_1}">
                            {rows:,}
                        </div>
                        <div style="font-size:11px;color:{TEXT_3}">rows</div>
                    </div>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;
                                    font-size:1.1rem;font-weight:700;color:{TEXT_1}">
                            {cols_}
                        </div>
                        <div style="font-size:11px;color:{TEXT_3}">columns</div>
                    </div>
                </div>
                <div style="font-size:11px;color:{ACCENT};margin-top:6px;
                            font-family:'JetBrains Mono',monospace">
                    {st.session_state.current_table or "—"}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="background:{SURFACE_3};border:1px dashed {BORDER};
                    border-radius:8px;padding:10px 12px;margin-bottom:16px;
                    text-align:center">
                <div style="font-size:12px;color:{TEXT_3}">No dataset loaded</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # AI status
    ai_ok = bool(gemini_key or groq_key)
    ai_label = "AI enabled" if ai_ok else "AI disabled — add API key"
    ai_color = ACCENT if ai_ok else DANGER
    st.markdown(
        f'<div style="font-size:11px;color:{ai_color};margin-bottom:16px;'
        f'display:flex;align-items:center;gap:6px">'
        f'<span style="font-size:13px">{"◆" if ai_ok else "◇"}</span>{ai_label}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Version history
    st.markdown(
        f'<div style="font-size:11px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:.07em;color:{TEXT_3};margin-bottom:10px">Action history</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.version_history:
        st.markdown(
            f'<div style="font-size:12px;color:{TEXT_3}">No actions yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for item in reversed(st.session_state.version_history[-8:]):
            with st.expander(f"{item['action']}", expanded=False):
                st.code(item.get("code", ""), language="sql" if "SQL" in item['action'] else "python")
                if "details" in item and item["details"]:
                    st.caption(item["details"])

# Main Header
st.markdown(
    f"""<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:1.5rem">
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;
                        font-weight:700;color:{TEXT_1};letter-spacing:-0.02em">
                ⬡ Data Analyzer & Visualizer Tool
            </div>
            <div style="font-size:13px;color:{TEXT_3};margin-top:2px">
                Upload data · Clean · Explore · Ask AI · Export
            </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
            <span style="font-size:11px;color:{TEXT_3};
                         font-family:'JetBrains Mono',monospace">v2.0</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# Tabs Definition
tab1, tab_etl, tab_comp, tab_forecasting, tab_journalist, tab1b, tab2, tab3, tab4, tab_pyg, tab_altair, tab_echarts, tab_dashboard, tab5 = st.tabs([
    "⬡ Ingest", "⟳ ETL", "🤝 Compare", "▲ Forecast", "✦ Journalist",
    "◉ Executive", "✦ Clean", "◆ AI Chat", "◈ Visualize", 
    "⊞ BI Explorer", "≋ Altair", "◎ Premium Viz", "▣ Dashboard", "↓ Export"
])

# --- Tab Implementations (Modularized) ---

with tab1:
    from views.tab_ingestion import render_tab_ingestion
    render_tab_ingestion(conn, db_name, get_ai_missions)

with tab_etl:
    from views.tab_etl import render_tab_etl
    render_tab_etl(conn)

with tab_comp:
    from views.tab_comparison import render_tab_comparison
    render_tab_comparison()

with tab_forecasting:
    from views.tab_forecasting import render_tab_forecasting
    render_tab_forecasting()

with tab_journalist:
    from views.tab_journalist import render_tab_journalist
    render_tab_journalist(conn)

with tab1b:
    from views.tab_summary import render_tab_summary
    render_tab_summary(conn)

with tab2:
    from views.tab_cleaning import render_tab_cleaning
    render_tab_cleaning(conn)

with tab3:
    from views.tab_assistant import render_tab_assistant
    render_tab_assistant(conn, gemini_key, groq_key)

from visualization import render_viz_tab

with tab4:
    if st.session_state.df_preview is not None:
        st.markdown("""
            <div style="background:#1A1D28; border:1px solid #2A2E45; border-radius:12px; padding:24px; text-align:center; margin-bottom:20px">
                <div style="font-size:24px; margin-bottom:12px">◈</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:600; color:#F0F2FF; margin-bottom:8px">
                    Advanced Visualization Engine
                </div>
                <div style="font-size:13px; color:#8B90B4; margin-bottom:20px">
                    Initialize the 20+ chart catalog (Standard, Statistical, Composition, Advanced). 
                    Click below to unlock the visualization suite.
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Initialize Visualizer ↗", key="launch_viz_engine"):
            st.session_state.viz_engine_launched = True

        if st.session_state.get("viz_engine_launched"):
            render_viz_tab(
                st.session_state.df_preview,
                safe_ai_call=safe_ai_call,
                gemini_key=gemini_key,
                groq_key=groq_key,
            )
    else:
        st.markdown(empty_state("◈", "No data loaded", "Load a dataset first."), unsafe_allow_html=True)

with tab_pyg:
    if st.session_state.df_preview is not None:
        from pyg_viz import render_walker_tab
        render_walker_tab(st.session_state.df_preview)
    else:
        st.markdown(empty_state("⊞", "No data loaded", "Load a dataset to use the BI Explorer."), unsafe_allow_html=True)

with tab_altair:
    from views.tab_visualizations import render_tab_altair
    render_tab_altair(CHART_COLORS)

with tab_echarts:
    if st.session_state.df_preview is not None:
        st.markdown("""
            <div style="background:#1A1D28; border:1px solid #2A2E45; border-radius:12px; padding:24px; text-align:center; margin-bottom:20px">
                <div style="font-size:24px; margin-bottom:12px">◎</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:600; color:#F0F2FF; margin-bottom:8px">
                    Premium ECharts Suite
                </div>
                <div style="font-size:13px; color:#8B90B4; margin-bottom:20px">
                    Unlock 8 sophisticated, animated visualizations (KPI Gauges, Radar, Sunburst, Activity Calendar).
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Initialize Premium Charts ↗", key="launch_echarts_final"):
            st.session_state.echarts_launched = True
            
        if st.session_state.get("echarts_launched"):
            from premium_viz import render_premium_viz_tab
            render_premium_viz_tab(cast(pd.DataFrame, st.session_state.df_preview))
    else:
        st.markdown(empty_state("◎", "No data loaded", "Load a dataset to view premium charts."), unsafe_allow_html=True)

with tab_dashboard:
    from views.tab_visualizations import render_tab_dashboard
    render_tab_dashboard()

with tab5:
    from views.tab_export import render_tab_export
    render_tab_export()

# Footer
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
if st.button("⬆️ Back to Top"):
    st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
