import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px

# Configuration
st.set_page_config(page_title="MotherDuck Data Visualizer", page_icon="🦆", layout="wide")

st.title("🦆 MotherDuck Data Visualizer")

# Retrieve MotherDuck token from secrets
try:
    md_token = st.secrets["motherduck"]["token"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create `.streamlit/secrets.toml` with your MotherDuck token.")
    st.stop()
except KeyError:
    st.error("MotherDuck token not found in secrets. Please add `[motherduck]` section with `token` to `.streamlit/secrets.toml`.")
    st.stop()

# Connect to MotherDuck
@st.cache_resource
def get_connection():
    # Note: in a real scenario you might pass the token via environment variable MOTHERDUCK_TOKEN
    # But duckdb connect also supports passing it in the connection string or setting it.
    # The standard way with MotherDuck is setting the motherduck_token configuration
    conn = duckdb.connect(f'md:?motherduck_token={md_token}')
    return conn

try:
    conn = get_connection()
except Exception as e:
    st.error(f"Failed to connect to MotherDuck: {e}")
    st.info("Please ensure your MotherDuck token is valid.")
    st.stop()

# Sidebar for Query Input
st.sidebar.header("Data Query")
st.sidebar.markdown("""
Enter your SQL query below. You can query MotherDuck sample datasets e.g.,
`SELECT * FROM sample_data.nyc.taxi_trips LIMIT 100`
""")

default_query = "SELECT * FROM sample_data.nyc.taxi_trips LIMIT 100"
query = st.sidebar.text_area("SQL Query", value=default_query, height=150)

if st.sidebar.button("Run Query"):
    with st.spinner("Executing query on MotherDuck..."):
        try:
            # Execute query and fetch as Pandas DataFrame
            df = conn.execute(query).df()
            st.session_state['df'] = df
            st.success("Query executed successfully!")
        except Exception as e:
            st.error(f"Error executing query: {e}")

# Main Area
if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    
    st.subheader("Data Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Toggle to show raw data
    with st.expander("Show Raw Data"):
        st.dataframe(df)
        
    st.divider()
    
    # Visualization Section
    st.subheader("Visualization")
    
    if len(df.columns) < 2:
        st.warning("Not enough columns to create a visualization. Need at least 2 columns.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
            
        with col2:
            x_axis = st.selectbox("X-Axis", df.columns.tolist())
            
        with col3:
            y_axis = st.selectbox("Y-Axis", df.columns.tolist(), index=min(1, len(df.columns) - 1))
            
        # Draw the selected chart
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
            
else:
    st.info("👆 Enter a SQL query in the sidebar and click 'Run Query' to load data.")

# Clean up connection
# Normally Streamlit handles connection lifecycle for cached resources,
# but it's good practice to ensure resources aren't leaked.
