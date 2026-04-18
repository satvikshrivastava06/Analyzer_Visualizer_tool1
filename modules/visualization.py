import streamlit as st
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components
import altair as alt


def render_pygwalker(df: pd.DataFrame):
    """
    Renders PyGWalker securely without writing JSON config to the ephemeral file system.
    """
    html_code = pyg.to_html(df)
    components.html(html_code, height=800, scrolling=True)

def render_altair_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """
    Renders a standard Altair chart with safe dynamic bindings.
    """
    if color_col:
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=x_col,
            y=y_col,
            color=color_col,
            tooltip=[x_col, y_col, color_col]
        ).interactive()
    else:
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=x_col,
            y=y_col,
            tooltip=[x_col, y_col]
        ).interactive()
    st.altair_chart(chart, use_container_width=True)
