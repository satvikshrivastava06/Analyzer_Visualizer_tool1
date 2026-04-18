import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import cast
from ui_theme import page_hero, section_header, anomaly_row, apply_plotly_theme, CHART_COLORS

def render_tab_forecasting():
    st.markdown(page_hero("◮", "Forecasting Lab", "Training AI models on historical data to predict future trends."), unsafe_allow_html=True)
    
    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load a dataset first in the Ingest tab."), unsafe_allow_html=True)
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2:
            st.markdown(anomaly_row("Need at least 2 numerical columns for forecasting.", "warning"), unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("📈", "Data Selection", "Time and target variables"), unsafe_allow_html=True)
            col_x = st.selectbox("Time/X-Axis", num_cols, index=0)
            col_y = st.selectbox("Target Value/Y-Axis", [c for c in num_cols if c != col_x], index=0)
            
            hist_df = df.dropna(subset=[col_x, col_y]).sort_values(by=col_x)
            
            from ui_theme import stat_card
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(stat_card("Historical Data Points", f"{len(hist_df):,}"), unsafe_allow_html=True)
            with col_m2:
                st.markdown(stat_card("Historical Span", f"{hist_df[col_x].min()} to {hist_df[col_x].max()}"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("🤖", "AI Training Engine", "Forecast future periods"), unsafe_allow_html=True)
            poly_degree = st.slider("Polynomial Complexity (Degree)", 1, 4, 2)
            steps_ahead = st.slider("Periods to Forecast Ahead", 1, 50, 10)
            
            if st.button("🚀 Train & Predict", use_container_width=True):
                with st.spinner("Training ML model..."):
                    X = hist_df[[col_x]].values
                    y = hist_df[col_y].values
                    
                    poly = PolynomialFeatures(degree=poly_degree)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    
                    # Predict future
                    last_x = hist_df[col_x].max()
                    step = 1 if 'year' in col_x.lower() else (hist_df[col_x].max() - hist_df[col_x].min())/len(hist_df) if len(hist_df)>1 else 1
                    future_Xs = np.array([last_x + i*step for i in range(1, steps_ahead+1)]).reshape(-1, 1)
                    future_X_poly = poly.transform(future_Xs)
                    predictions = model.predict(future_X_poly)
                    
                    pred_df = pd.DataFrame({
                        col_x: future_Xs.flatten(),
                        "AI_Prediction": predictions
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_df[col_x], y=hist_df[col_y], name="Historical", mode='markers', marker=dict(color=CHART_COLORS[0])))
                    fig.add_trace(go.Scatter(x=pred_df[col_x], y=pred_df['AI_Prediction'], name=f"Forecast (Deg {poly_degree})", line=dict(color=CHART_COLORS[3], width=3)))
                    fig.update_layout(title=f"Forecast: {col_y}", xaxis_title=col_x, yaxis_title=col_y)
                    apply_plotly_theme(fig)
                    st.plotly_chart(fig, width='stretch')
