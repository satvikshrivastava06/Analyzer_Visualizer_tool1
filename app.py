import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import google.generativeai as genai
import os
import json
import numpy as np

# ==========================================
# Configuration & Setup
# ==========================================
st.set_page_config(page_title="Data Analyser and Visualiser Tool - ( By Satvik Shrivastava, Yash Kamal Koshti and Vansh Dhakad )", page_icon="🧠", layout="wide")
st.title("🧠 Data Analyser and Visualiser Tool - ( By Satvik Shrivastava, Yash Kamal Koshti and Vansh Dhakad )")

# 1. Initialize API Keys
try:
    md_token = st.secrets["motherduck"]["token"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create `.streamlit/secrets.toml`.")
    st.stop()
except KeyError:
    st.error("MotherDuck token not found in `.streamlit/secrets.toml` under `[motherduck]` `token`.")
    st.stop()

# Optional Gemini Key
gemini_key = None
try:
    gemini_key = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=gemini_key)
except (FileNotFoundError, KeyError):
    st.warning("Google Gemini API Key not found. AI features will be disabled. Add to `.streamlit/secrets.toml` under `[gemini] api_key`.")

# 2. Database Connection
if 'db_conn' not in st.session_state:
    import os
    os.environ["MOTHERDUCK_TOKEN"] = md_token
    st.session_state.db_conn = duckdb.connect(f'md:?motherduck_token={md_token}')

conn = st.session_state.db_conn

# Initialize session state for data
if 'current_table' not in st.session_state:
    st.session_state.current_table = None
if 'df_preview' not in st.session_state:
    st.session_state.df_preview = None

# ==========================================
# UI Layout: Tabs
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Data Ingestion", 
    "🧹 Smart Clean", 
    "🤖 AI Assistant", 
    "📊 Generative Viz", 
    "💾 Export Options"
])

# ------------------------------------------
# Tab 1: Data Ingestion (Scalable loading)
# ------------------------------------------
with tab1:
    st.header("Upload or Query Data")
    st.markdown("""
    Load data directly into DuckDB. For large files, use the upload feature to process out-of-core.
    """)
    
    # Quick Demo Data
    st.subheader("⚡ Quick Demo")
    st.markdown("Load a generated dataset to test the tool's features without connecting to a database.")
    if st.button("Load Dummy Data"):
        with st.spinner("Generating dummy dataset..."):
            np.random.seed(42)
            dates = pd.date_range(start="2023-01-01", periods=100)
            demo_df = pd.DataFrame({
                "Date": np.random.choice(dates, 500),
                "Product": np.random.choice(["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor"], 500),
                "Category": np.random.choice(["Electronics", "Accessories"], 500, p=[0.7, 0.3]),
                "Revenue": np.random.normal(500, 150, 500).round(2),
                "Cost": np.random.normal(300, 100, 500).round(2),
                "Quantity": np.random.randint(1, 10, 500),
                "Defective": np.random.choice([True, False], 500, p=[0.05, 0.95]),
                "Rating": np.random.uniform(1.0, 5.0, 500).round(1)
            })
            # Introduce missing values for "Smart Clean" tab Demo
            demo_df.loc[10:30, 'Revenue'] = np.nan
            demo_df.loc[40:60, 'Category'] = np.nan
            
            conn.execute("CREATE OR REPLACE VIEW current_data AS SELECT * FROM demo_df")
            st.session_state.current_table = "current_data"
            st.session_state.df_preview = demo_df
            st.success("Dummy data loaded successfully! Explore it in the other tabs.")

    st.divider()
    
    # Option A: MotherDuck Query
    st.subheader("Query MotherDuck Database")
    default_query = "SELECT * FROM sample_data.nyc.taxi LIMIT 5000"
    query = st.text_area("SQL Query", value=default_query, height=100)
    
    if st.button("Load Query Results"):
        with st.spinner("Executing query..."):
            try:
                # Store as a view for future tabs without keeping entire set in Pandas memory
                conn.execute(f"CREATE OR REPLACE VIEW current_data AS {query}")
                st.session_state.current_table = "current_data"
                # Keep a small preview in pandas for the UI
                st.session_state.df_preview = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Preview
    if st.session_state.df_preview is not None:
        st.subheader("Data Preview (First 1000 rows)")
        st.dataframe(st.session_state.df_preview)
        
        # Get count
        total_rows = conn.execute(f"SELECT COUNT(*) FROM {st.session_state.current_table}").fetchone()[0]
        st.caption(f"Total Rows in Engine: {total_rows}")

# ------------------------------------------
# Tab 2: Smart Clean (Automated Profiling)
# ------------------------------------------
with tab2:
    st.header("Smart Data Cleaning & Profiling")
    if st.session_state.current_table is None:
        st.info("Please load data in the 'Data Ingestion' tab first.")
    else:
        df = st.session_state.df_preview
        
        st.subheader("Data Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numerical Summary**")
            st.dataframe(df.describe())
        with col2:
            st.write("**Missing Values (%)**")
            missing = (df.isnull().sum() / len(df)) * 100
            st.dataframe(missing[missing > 0].to_frame(name="Missing %"))
            
        st.subheader("Automated Cleaning Actions")
        
        if st.button("Run Smart Imputation (Heuristic)"):
            with st.spinner("Cleaning data..."):
                # Simple logic to create a cleaned view in duckdb
                # In a real app we would build a dynamic SQL query based on pandas types.
                # For this demo, we'll do the cleaning on the preview dataframe.
                cleaned_df = df.copy()
                for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                for col in cleaned_df.select_dtypes(include=['object']).columns:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown")
                
                st.session_state.df_preview = cleaned_df
                # Register the cleaned pandas df back to Duckdb (as a table)
                conn.execute("CREATE OR REPLACE TABLE current_data AS SELECT * FROM cleaned_df")
                st.success("Imputed missing values! (Median for nums, Mode for categorical)")
                st.experimental_rerun()

# ------------------------------------------
# Tab 3: AI Assistant (Natural Language)
# ------------------------------------------
with tab3:
    st.header("Chat with your Data")
    if not gemini_key:
        st.error("Please add your Gemini API Key to use this feature.")
    elif st.session_state.current_table is None:
        st.info("Please load data first.")
    else:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your data (e.g., 'What is the average fare amount?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # 1. Get Schema
                    schema_df = conn.execute(f"DESCRIBE {st.session_state.current_table}").df()
                    schema_str = schema_df[['column_name', 'column_type']].to_string()
                    
                    # 2. Ask Gemini
                    model = genai.GenerativeModel('gemini-pro')
                    ai_prompt = f"""
                    You are a data analyst assistant. You have a table named '{st.session_state.current_table}'.
                    Schema:
                    {schema_str}
                    
                    The user asked: "{prompt}"
                    
                    Please provide a friendly, helpful, and concise answer explaining how to find this or what it might mean.
                    If you can write a DuckDB SQL query to get the answer, provide it within a ```sql block.
                    """
                    try:
                        response = model.generate_content(ai_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Error generation response: {e}")

# ------------------------------------------
# Tab 4: Generative Viz (LLM recommended)
# ------------------------------------------
with tab4:
    st.header("Intelligent Visualization")
    if st.session_state.current_table is None:
        st.info("Please load data first.")
    else:
        df = st.session_state.df_preview
        
        st.markdown("### Manual Chart Creation")
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart", ["Bar", "Scatter", "Histogram", "Box"])
        with col2:
            x_ax = st.selectbox("X-Axis", df.columns.tolist())
        with col3:
            y_ax = st.selectbox("Y-Axis", df.columns.tolist() + ["None"])

        if st.button("Plot Manual"):
            if chart_type == "Bar" and y_ax != "None":
                st.plotly_chart(px.bar(df, x=x_ax, y=y_ax), use_container_width=True)
            elif chart_type == "Scatter" and y_ax != "None":
                st.plotly_chart(px.scatter(df, x=x_ax, y=y_ax), use_container_width=True)
            elif chart_type == "Histogram":
                st.plotly_chart(px.histogram(df, x=x_ax), use_container_width=True)
            elif chart_type == "Box" and y_ax != "None":
                st.plotly_chart(px.box(df, x=x_ax, y=y_ax), use_container_width=True)
                
        st.divider()
        st.markdown("### AI Recommended Charts")
        if not gemini_key:
            st.warning("Add Gemini API key to get smart chart recommendations based on your schema.")
        else:
            if st.button("Generate Recommendations"):
                with st.spinner("AI is analyzing schema for optimal visualizations..."):
                    schema_str = df.dtypes.to_string()
                    try:
                        model = genai.GenerativeModel('gemini-pro')
                        prompt = f"""
                        Given this pandas dataframe schema:
                        {schema_str}
                        
                        Suggest 3 insightful Plotly Express charts to visualize this data. 
                        Format your response as a numbered list with the chart title, the Plotly function to use (px.scatter, px.histogram, etc.), and the exact x and y column names from the schema.
                        """
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error("Failed to fetch suggestions.")

# ------------------------------------------
# Tab 5: Export Pipeline (Reproducibility)
# ------------------------------------------
with tab5:
    st.header("Reproducible Analysis")
    st.markdown("""
    One of the major loopholes in data analysis is lack of reproducibility. 
    Click below to export this exact session logic as a standalone Python script!
    """)
    
    script_template = """
import duckdb
import pandas as pd

# 1. Connect and Load
conn = duckdb.connect('md:') # Assumes MOTHERDUCK_TOKEN is in env

# 2. Run Query
query = \"\"\"SELECT * FROM sample_data.nyc.taxi LIMIT 5000\"\"\"
print("Executing robust fetching...")
df = conn.execute(query).df()

# 3. Clean
print("Applying standard AI-assistant cleaning...")
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

print("Pipeline finished successfully! Shape:", df.shape)
    """
    
    st.code(script_template, language="python")
    st.download_button("Download Script (reproducible_analysis.py)", script_template, file_name="reproducible_analysis.py")
    
    req_txt = "streamlit\npandas\nduckdb\nplotly\ngoogle-generativeai\nnumpy"
    st.download_button("Download dependencies (requirements.txt)", req_txt, file_name="session_requirements.txt")
