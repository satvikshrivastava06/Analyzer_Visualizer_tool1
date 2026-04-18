import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import cast
from modules.ai_engine import safe_ai_call
from ui_theme import page_hero, section_header, anomaly_row, apply_plotly_theme, CHART_COLORS

def render_tab_journalist(conn):
    st.markdown(page_hero("◫", "Journalist Lab", "Authoritative, data-driven storytelling inspired by modern journalism."), unsafe_allow_html=True)
    
    j_sub1, j_sub2 = st.tabs(["🔗 API Connector", "📈 Trend Aggregator (Smoothing)"])
    
    with j_sub1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("🔗", "Connect to External Datasets", "Live asset ingestion"), unsafe_allow_html=True)
        
        dataset_options = {
            "Presidential Polls (2020)": "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/presidential_polls.csv",
            "NBA Elo Ratings": "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv",
            "Trump Approval Ratings": "https://raw.githubusercontent.com/fivethirtyeight/data/master/trump-approval-ratings/approval_topline.csv"
        }
        
        selected_j_ds = st.selectbox("Choose Dataset", list(dataset_options.keys()))
        if st.button("🚀 Fetch & Ingest", use_container_width=True):
            with st.spinner(f"Connecting to Open Data..."):
                try:
                    url = dataset_options[selected_j_ds]
                    df = pd.read_csv(url)
                    conn.execute("CREATE OR REPLACE TABLE journalist_data AS SELECT * FROM df")
                    st.session_state.current_table = "journalist_data"
                    st.session_state.df_preview = df
                    st.markdown(anomaly_row(f"Successfully ingested {selected_j_ds} into DuckDB!", "success"), unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(anomaly_row(f"Ingestion failed: {e}", "warning"), unsafe_allow_html=True)

    with j_sub2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("📈", "Statistical Smoothing", "Apply rolling averages to noisy time-series data to reveal true trends."), unsafe_allow_html=True)
        
        if st.session_state.df_preview is None:
            st.markdown(anomaly_row("Please load a dataset using the Connector first.", "info"), unsafe_allow_html=True)
        else:
            df = st.session_state.df_preview
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            time_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
            
            col_agg1, col_agg2, col_agg3 = st.columns(3)
            with col_agg1:
                x_axis = st.selectbox("Time Axis (X)", time_cols)
            with col_agg2:
                y_axis = st.selectbox("Value Axis (Y)", numerical_cols)
            with col_agg3:
                window = st.slider("Smoothing Window (Days/Obs)", 1, 30, 7)
            
            if st.button("📊 Generate Trend Analysis", use_container_width=True):
                try:
                    # Rolling average
                    df_sorted = df.sort_values(by=x_axis)
                    df_sorted['smoothed'] = df_sorted[y_axis].rolling(window=window, center=True).mean()
                    
                    fig = go.Figure()
                    # Raw data (faded)
                    fig.add_trace(go.Scatter(x=df_sorted[x_axis], y=df_sorted[y_axis], name="Raw Model", mode='markers', marker=dict(color=CHART_COLORS[1], opacity=0.3)))
                    # Smoothed trend
                    fig.add_trace(go.Scatter(x=df_sorted[x_axis], y=df_sorted['smoothed'], name=f"{window}-Point Rolling Avg", line=dict(color=CHART_COLORS[3], width=4)))
                    
                    fig.update_layout(title=f"Narrative Trend Analysis: {y_axis}")
                    apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(section_header("🎙️", "Journalistic Commentary", "AI-driven interpretation"), unsafe_allow_html=True)
                    prompt = f"Write a hard-hitting data-journalism-style summary of this trend where {y_axis} is analyzed over {x_axis} with a {window}-point smoothing window. Use data-driven language, maintaining an authoritative yet accessible voice."
                    summary, _ = safe_ai_call(prompt)
                    
                    st.markdown(f'<div class="ai-chat-msg bot-msg">{summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(anomaly_row(f"Analysis failed: {e}", "warning"), unsafe_allow_html=True)
