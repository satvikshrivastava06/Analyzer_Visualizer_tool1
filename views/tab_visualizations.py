import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from streamlit_echarts import st_echarts
import streamlit.components.v1 as components
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
from typing import cast
from ui_theme import page_hero, section_header, anomaly_row, CHART_COLORS, apply_plotly_theme, PLOTLY_LAYOUT

@st.cache_resource
def get_pyg_renderer(df: pd.DataFrame) -> StreamlitRenderer:
    return StreamlitRenderer(df, spec_io_mode="r")

def render_tab_pyg():
    st.markdown(page_hero("⊞", "BI Explorer (PyGWalker)", "Build Tableau-style visualizations instantly using an interactive canvas."), unsafe_allow_html=True)
    
    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load data in the Ingest tab first."), unsafe_allow_html=True)
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        st.markdown("""
            <div style="background:#1A1D28; border:1px solid #2A2E45; border-radius:12px; padding:24px; text-align:center; margin-bottom:20px">
                <div style="font-size:24px; margin-bottom:12px">🚀</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:600; color:#F0F2FF; margin-bottom:8px">
                    Interactive BI Engine Ready
                </div>
                <div style="font-size:13px; color:#8B90B4; margin-bottom:20px">
                    Click the button below to initialize the high-performance Graphic Walker canvas. 
                    This will prevent automatic popups and stabilize the session.
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Initialize BI Explorer ↗", key="launch_pyg"):
            st.session_state.pyg_launched = True

        if st.session_state.get("pyg_launched"):
            try:
                with st.spinner("Initializing PyGWalker engine..."):
                    renderer = get_pyg_renderer(df)
                    renderer.explorer()
                st.caption("Powered by PyGWalker. Note: Very large datasets (> 100k rows) may slow down the browser in this view.")
            except Exception as e:
                st.error(f"Could not load PyGWalker: {e}")

def render_tab_altair(BRAND_COLORS):
    st.markdown(page_hero("≋", "Altair Insights", "Declarative Statistical Insights using a grammar-of-graphics approach."), unsafe_allow_html=True)
    
    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load data in the Ingest tab first."), unsafe_allow_html=True)
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(num_cols) >= 2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("≋", "Interactive Multi-View Explorer", "Filter and explore relationships"), unsafe_allow_html=True)
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                x_alt = st.selectbox("X-Axis (Scatter)", num_cols, index=0, key="alt_x")
            with col_a2:
                y_alt = st.selectbox("Y-Axis (Scatter)", num_cols, index=min(1, len(num_cols)-1), key="alt_y")
            
            color_alt = st.selectbox("Color Segment", ["None"] + cat_cols, key="alt_color")
            
            brush = alt.selection_interval()
            base = alt.Chart(df).encode(
                color=alt.Color(f'{color_alt}:N', scale=alt.Scale(range=BRAND_COLORS)) if color_alt != "None" else alt.value(BRAND_COLORS[0])
            )
            
            scatter = base.mark_point(filled=True, size=60).encode(
                x=alt.X(x_alt),
                y=alt.Y(y_alt),
                tooltip=df.columns.tolist()
            ).add_params(brush).properties(width=500, height=400)
            
            if color_alt != "None":
                bars_y = alt.Y(f'{color_alt}:N').title("Segment")
            else:
                bars_y = alt.Y('Dataset:N').title("Dataset")
                
            bars = base.mark_bar().encode(
                x='count()',
                y=bars_y
            ).transform_calculate(
                Dataset='"Total"'
            ).transform_filter(brush).properties(width=500, height=150)
            
            st.altair_chart(scatter & bars, use_container_width=True)
            st.caption("💡 **Tip:** Click and drag on the scatter plot to select a region. The bar chart below will dynamically update to show the distribution of the selected points.")
        else:
            st.markdown(anomaly_row("At least two numerical columns are required for Altair Interactive insights.", "warning"), unsafe_allow_html=True)

def render_tab_echarts():
    st.markdown(page_hero("◎", "Premium Viz (ECharts)", "Highly performant, animated, and sophisticated open-source visualizations."), unsafe_allow_html=True)
    
    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load data in the Ingest tab first."), unsafe_allow_html=True)
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        st.markdown("""
            <div style="background:#1A1D28; border:1px solid #2A2E45; border-radius:12px; padding:24px; text-align:center; margin-bottom:20px">
                <div style="font-size:24px; margin-bottom:12px">◎</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:600; color:#F0F2FF; margin-bottom:8px">
                    Premium ECharts Suite
                </div>
                <div style="font-size:13px; color:#8B90B4; margin-bottom:20px">
                    Access 8 world-class animated visualizations (KPI Gauges, Radar, Sunburst, etc).
                    Click below to initialize the engine.
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Initialize Premium Charts ↗", key="launch_echarts_tab_fixed"):
            st.session_state.echarts_launched = True
        
        if st.session_state.get("echarts_launched"):
            from premium_viz import render_premium_viz_tab
            render_premium_viz_tab(df)

def render_tab_dashboard():
    st.markdown(page_hero("▣", "Visual Dashboard", "Auto-generated Power BI-style analytics from your active dataset."), unsafe_allow_html=True)

    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "📥 Please load data in the Ingest tab first to see your dashboard."), unsafe_allow_html=True)
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        with st.expander("⚙️ Dashboard Controls", expanded=False):
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                primary_num = st.selectbox("Primary Metric", num_cols, key="dash_primary_num") if num_cols else None
            with ctrl2:
                secondary_num = st.selectbox("Secondary Metric", [c for c in num_cols if c != primary_num], key="dash_sec_num") if len(num_cols) > 1 else None
            with ctrl3:
                primary_cat = st.selectbox("Category Column", cat_cols, key="dash_primary_cat") if cat_cols else None

        if not num_cols:
            st.markdown(anomaly_row("⚠️ Your dataset has no numerical columns. Please load a richer dataset.", "warning"), unsafe_allow_html=True)
        else:
            primary_num = st.session_state.get("dash_primary_num") or num_cols[0]
            secondary_num = st.session_state.get("dash_sec_num") or (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            primary_cat = st.session_state.get("dash_primary_cat") or (cat_cols[0] if cat_cols else None)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("▣", "Key Performance Indicators", "High-level metrics summary"), unsafe_allow_html=True)
            
            from ui_theme import stat_card
            kpi_cols = num_cols[:4]
            kpi_card_cols = st.columns(len(kpi_cols))

            for i, col_name in enumerate(kpi_cols):
                series = df[col_name].dropna()
                val = series.mean()
                pct_change = ((series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0]) * 100) if len(series) > 1 and series.iloc[0] != 0 else 0
                fmt_val = f"{val:,.0f}" if abs(val) >= 1000 else f"{val:,.2f}"

                with kpi_card_cols[i]:
                    st.markdown(stat_card(
                        col_name[:18],
                        fmt_val,
                        delta=f"{pct_change:+.1f}% trend",
                        delta_type="pos" if pct_change >= 0 else "neg"
                    ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("▣", "Distribution & Comparison", "Macro trends"), unsafe_allow_html=True)
            r2c1, r2c2 = st.columns(2)

            with r2c1:
                if primary_cat:
                    agg = df.groupby(primary_cat)[primary_num].sum().reset_index().sort_values(primary_num, ascending=False).head(12)
                    fig_bar = px.bar(
                        agg, x=primary_cat, y=primary_num,
                        title=f"{primary_num} by {primary_cat}",
                        color=primary_num,
                        color_continuous_scale=[[0, "#312e81"], [0.5, "#818cf8"], [1, "#38bdf8"]],
                        text_auto=True
                    )
                    fig_bar.update_traces(textposition="outside", textfont_color="#94a3b8")
                    apply_plotly_theme(fig_bar)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    fig_hist = px.histogram(df, x=primary_num, title=f"Distribution of {primary_num}",
                                            nbins=30, color_discrete_sequence=["#818cf8"])
                    apply_plotly_theme(fig_hist)
                    st.plotly_chart(fig_hist, use_container_width=True)

            with r2c2:
                if len(num_cols) >= 2:
                    scatter_kwargs = dict(
                        x=primary_num, y=secondary_num,
                        title=f"{primary_num} vs {secondary_num}",
                        color_discrete_sequence=CHART_COLORS
                    )
                    if primary_cat:
                        scatter_kwargs["color"] = primary_cat
                    fig_scatter = px.scatter(
                        df.sample(min(len(df), 2000), random_state=42),
                        **scatter_kwargs,
                        opacity=0.75,
                        size_max=12
                    )
                    fig_scatter.update_traces(marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.2)")))
                    apply_plotly_theme(fig_scatter)
                    st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("▣", "Hierarchical & Statistical Views", "Deep dive insights"), unsafe_allow_html=True)
            r3c1, r3c2 = st.columns(2)

            dashboard_figs = []

            with r3c1:
                if primary_cat and len(cat_cols) >= 1:
                    try:
                        path_cols = cat_cols[:2] if len(cat_cols) >= 2 else cat_cols[:1]
                        fig_tree = px.treemap(
                            df.dropna(subset=[primary_num]),
                            path=path_cols,
                            values=primary_num,
                            title=f"Treemap: {primary_num}",
                            color=primary_num,
                            color_continuous_scale=[[0, "#1e1b4b"], [0.5, "#6d28d9"], [1, "#e879f9"]]
                        )
                        apply_plotly_theme(fig_tree)
                        st.plotly_chart(fig_tree, use_container_width=True)
                        dashboard_figs.append(fig_tree)
                    except Exception:
                        st.markdown(anomaly_row("Treemap could not be generated with these dimensions.", "info"), unsafe_allow_html=True)
                else:
                    st.markdown(anomaly_row("Select a category column to see hierarchical data.", "info"), unsafe_allow_html=True)

            with r3c2:
                if primary_cat:
                    fig_box = px.box(
                        df, x=primary_cat, y=primary_num,
                        title=f"{primary_num} Distribution by {primary_cat}",
                        color=primary_cat,
                        color_discrete_sequence=CHART_COLORS
                    )
                    apply_plotly_theme(fig_box)
                    fig_box.update_layout(showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)
                    dashboard_figs.append(fig_box)
                else:
                    fig_box = px.box(
                        df, y=primary_num,
                        title=f"{primary_num} Box Plot",
                        color_discrete_sequence=["#f87171"]
                    )
                    apply_plotly_theme(fig_box)
                    st.plotly_chart(fig_box, use_container_width=True)
                    dashboard_figs.append(fig_box)

            # Insert earlier figures too
            if 'fig_bar' in locals(): dashboard_figs.insert(0, fig_bar)
            if 'fig_hist' in locals(): dashboard_figs.insert(0, fig_hist)
            if 'fig_scatter' in locals(): dashboard_figs.insert(1, fig_scatter)
            
            st.session_state.dashboard_figs = dashboard_figs
