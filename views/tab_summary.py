import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import cast
from ui_theme import page_hero, section_header, stat_card, anomaly_row, CHART_COLORS, apply_plotly_theme
from modules.analysis import detect_anomalies
from modules.ai_engine import explain_anomalies

def render_tab_summary(conn):
    st.markdown(page_hero("◉", "Executive Summary", "High-level strategic overview restricting cognitive load to core KPIs."), unsafe_allow_html=True)
    
    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load data in the Ingest tab first."), unsafe_allow_html=True)
        return

    df = cast(pd.DataFrame, st.session_state.df_preview)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("◉", "Core KPIs", "Macro-level business metrics"), unsafe_allow_html=True)
    
    kpi_cols = st.columns(4)
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    with kpi_cols[0]:
        st.markdown(stat_card("Total Records", f"{len(df):,}"), unsafe_allow_html=True)
            
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not date_cols:
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    
    for i, col in enumerate(num_cols[:3]):
        if i+1 < len(kpi_cols):
            with kpi_cols[i+1]:
                avg_val = df[col].mean()
                delta_val = None
                delta_str = "No temporal context"
                
                if date_cols:
                    try:
                        t_col = date_cols[0]
                        temp_df = df.sort_values(by=t_col, ascending=False)
                        mid_point = len(temp_df) // 2
                        current_period_avg = temp_df.iloc[:mid_point][col].mean()
                        prev_period_avg = temp_df.iloc[mid_point:][col].mean()
                        
                        if prev_period_avg != 0 and not np.isnan(prev_period_avg):
                            delta_val = ((current_period_avg - prev_period_avg) / prev_period_avg) * 100
                            delta_str = f"{delta_val:+.1f}% vs Prev Period"
                    except Exception:
                        pass
                
                st.markdown(stat_card(
                    f"Avg {col}", 
                    f"{avg_val:,.2f}", 
                    delta=delta_str, 
                    delta_type="pos" if delta_val and delta_val > 0 else "neg"
                ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("◈", "Strategic Overview", "Sample Visual"), unsafe_allow_html=True)
    
    if len(num_cols) >= 2:
        fig = px.scatter(
            df, 
            x=num_cols[0], 
            y=num_cols[1], 
            title=f"Strategic Alignment: {num_cols[0]} vs {num_cols[1]}",
            color_discrete_sequence=CHART_COLORS,
            opacity=0.7
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, width='stretch')
    else:
        st.markdown(anomaly_row("Insufficient numeric columns to generate a strategic scatter overview.", "warning"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🔍", "AI Anomaly Diagnostics", "Automated statistical scanning with AI-driven interpretation of outliers."), unsafe_allow_html=True)
    
    if st.button("🚀 Run Deep Diagnostic Scan", use_container_width=True):
        with st.spinner("Analyzing statistical signals..."):
            found_anomalies = detect_anomalies(df)
            if not found_anomalies:
                st.markdown(anomaly_row("Clean Health Bill: No critical statistical anomalies detected (Z-score > 3).", "success"), unsafe_allow_html=True)
            else:
                for item in found_anomalies:
                    col = item['column']
                    count = item['count']
                    outlier_df = item['outliers']
                    
                    with st.expander(f"🚩 Column: {col} ({count} outliers detected)", expanded=True):
                        st.dataframe(outlier_df.head(10), width='stretch')
                        st.markdown("---")
                        st.markdown("**AI Interpretation:**")
                        explanation = explain_anomalies(df, outlier_df, col)
                        st.write(explanation)
