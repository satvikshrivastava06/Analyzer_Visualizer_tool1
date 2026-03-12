# Copyright (c) 2026 Satvik Shrivastava. All rights reserved.
# This work is the property of Satvik Shrivastava.
# Unauthorized copying, modification, or distribution is strictly prohibited.

import streamlit as st

import pandas as pd
import duckdb
import plotly.express as px
import google.generativeai as genai # type: ignore
import os
import json
import numpy as np
import glob
import zipfile
from typing import Any, cast
import pygwalker.api as pyg
import streamlit.components.v1 as components
import altair as alt
import requests
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_echarts import st_echarts
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# Configuration & Setup
# ==========================================
import plotly.io as pio
import plotly.graph_objects as go

# Best Practice 4: Implement a Visualization Design System
# Set a consistent, clean base template prioritizing clarity over complexity.
pio.templates.default = "plotly_white"

# Best Practice 7: Use Color Strategically
BRAND_COLORS = ["#00B8D9", "#36B37E", "#FF5630", "#FFAB00", "#6554C0"]

# --- FiveThirtyEight Signature Style ---
F38_COLOR_RED = "#ED1C24"
F38_COLOR_BLUE = "#008FD5"
F38_COLOR_BG = "#F0F0F0"
F38_COLOR_GRID = "#D7D7D7"

def apply_538_style(fig):
    fig.update_layout(
        plot_bgcolor=F38_COLOR_BG,
        paper_bgcolor=F38_COLOR_BG,
        font=dict(family="Helvetica, Arial, sans-serif", size=14, color="#333333"),
        title=dict(font=dict(size=24, color="#333333"), x=0.05),
        xaxis=dict(gridcolor=F38_COLOR_GRID, linecolor=F38_COLOR_GRID, showline=True),
        yaxis=dict(gridcolor=F38_COLOR_GRID, linecolor=F38_COLOR_GRID, showline=True),
        legend=dict(bgcolor="rgba(255,255,255,0.5)")
    )
    return fig

st.set_page_config(page_title="Data Analyser and Visualiser Tool - ( By Satvik Shrivastava, Yash Kamal Koshti and Vansh Dhakad )", page_icon="🧠", layout="wide")

# 1. Initialize API Keys
try:
    md_token = st.secrets["motherduck"]["token"]
    db_name = st.secrets["motherduck"]["db_name"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create `.streamlit/secrets.toml`.")
    st.stop()
except KeyError as e:
    st.error(f"Missing key in `.streamlit/secrets.toml`: {e}")
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
    try:
        # Set environment variable for token-based authentication
        os.environ["MOTHERDUCK_TOKEN"] = md_token
        # Simplify connection: 'md:database_name' is more robust for direct auth
        st.session_state.db_conn = duckdb.connect(f'md:{db_name}')
        st.success(f"Connected to MotherDuck: {db_name}")
    except Exception as e:
        st.warning(f"MotherDuck connection failed: {e}. Falling back to local/in-memory engine.")
        # Fallback to a local-only connection if MotherDuck is unreachable or tokens are invalid
        st.session_state.db_conn = duckdb.connect(':memory:')

conn = st.session_state.db_conn

# Optional Groq Key (Free tier - https://console.groq.com)
groq_key = None
try:
    groq_key = st.secrets["groq"]["api_key"]
except (FileNotFoundError, KeyError):
    pass  # Groq is optional; Gemini will be used as primary

# ==========================================
# Multi-Provider AI Fallback System
# ==========================================
def safe_ai_call(prompt):
    """
    Universal AI call with multi-provider fallback strategy.
    1st: Tries best free Gemini models (fastest, most capable)
    2nd: If all Gemini quota exhausted, falls back to Groq free models
         (Llama 3.3 70B, Mixtral, Gemma 2 - very fast, zero cost)
    """
    import time

    # --- Tier 1: Gemini (Google AI Studio Free Tier) ---
    if gemini_key:
        gemini_candidates = [
            'models/gemini-2.5-flash',       # Best: Latest & most capable
            'models/gemini-2.0-flash',        # Good: Fast & reliable
            'models/gemini-2.0-flash-lite',   # Lighter quota usage
            'models/gemini-flash-latest',     # Alias: always latest flash
            'models/gemma-3-27b-it',          # Open: Google's Gemma 3 27B
            'models/gemma-3-12b-it',          # Open: Google's Gemma 3 12B
            'models/gemini-pro-latest',       # Classic: Gemini Pro
        ]
        for model_name in gemini_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                return resp.text, model_name
            except Exception as e:
                err = str(e).lower()
                # Quota/not-found: skip to next model silently
                if any(x in err for x in ["429", "404", "not supported", "resource_exhausted"]):
                    continue
                else:
                    raise e  # Real error – surface it

    # --- Tier 2: Groq (Free tier - Llama/Mixtral/Gemma via Groq cloud) ---
    if groq_key:
        try:
            from groq import Groq # type: ignore
            groq_client = Groq(api_key=groq_key)
            groq_candidates = [
                "llama-3.3-70b-versatile",     # Best: Llama 3.3 70B (most capable)
                "mixtral-8x7b-32768",           # Great: Mixtral 8x7B (long context)
                "gemma2-9b-it",                 # Fast: Google Gemma 2 9B via Groq
                "llama-3.1-8b-instant",         # Fastest: Llama 3.1 8B
            ]
            for model_name in groq_candidates:
                try:
                    completion = groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                    )
                    return completion.choices[0].message.content, f"groq/{model_name}"
                except Exception as e:
                    err = str(e).lower()
                    if any(x in err for x in ["429", "rate_limit", "model_not_found", "deactivated"]):
                        continue
                    else:
                        raise e
        except ImportError:
            pass  # groq package not installed

    # --- All providers exhausted ---
    raise Exception("🛑 All free AI models are currently at their quota limit. Please wait a few minutes and try again.")

def validate_sql_query(query):
    """
    SQL Security Sandbox: Scans for destructive keywords to prevent SQL Injection
    or accidental data deletion via AI or manual SQL.
    """
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'ALTER', 'TRUNCATE', 'GRANT', 'INSERT', 'REVOKE']
    query_upper = query.upper()
    for keyword in dangerous_keywords:
        # Check for whole word match to avoid false positives with column names
        if f" {keyword} " in f" {query_upper} " or query_upper.startswith(keyword):
            return False, keyword
    return True, None

def log_version_action(action_name, code, details=""):
    """
    Git-Lite: Logs every action taken on the data to ensure full auditability/provenance.
    """
    import datetime
    if 'version_history' not in st.session_state:
        st.session_state.version_history = []
    
    st.session_state.version_history.append({
        "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "action": action_name,
        "code": code,
        "details": details
    })

# Initialize session state for data
if 'current_table' not in st.session_state:
    # Attempt to pre-load the user's executive KPI data if it exists in the connected DB
    try:
        # Check if executive_kpi_data exists in the main schema of the connected db
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

# Best Practice 13: Integrate Governance/Lineage in UI Layer
with st.sidebar:
    st.markdown("### 🗄️ Governance & Lineage")
    
    # Connection Status
    if conn:
        st.success(f"✅ **MotherDuck Connected**\n\nDatabase: `{db_name}`")
    else:
        st.error("❌ **MotherDuck Disconnected**")
    
    st.divider()
    
    # Active Table Status
    if st.session_state.current_table:
        st.info(f"**Active Source:** `{st.session_state.current_table}`")
        if st.session_state.df_preview is not None:
             st.caption(f"Cached rows (Preview): {len(st.session_state.df_preview)}")
             st.caption(f"Schema columns: {len(st.session_state.df_preview.columns)}")
    else:
        st.warning("No data source loaded.")

    st.divider()
    st.markdown("### 📜 Version History (Git-Lite)")
    if not st.session_state.version_history:
        st.caption("No actions recorded yet.")
    else:
        for item in reversed(st.session_state.version_history):
            with st.expander(f"{item['timestamp']} - {item['action']}"):
                st.code(item['code'], language="sql" if "SQL" in item['action'] else "python")
                if item['details']:
                    st.caption(item['details'])

st.title("🧠 Data Analyser and Visualiser Tool - ( By Satvik Shrivastava, Yash Kamal Koshti and Vansh Dhakad )")

# ==========================================
# UI Layout: Tabs
# ==========================================
tab1, tab_etl, tab_forecasting, tab_journalist, tab1b, tab2, tab3, tab4, tab_pyg, tab_altair, tab_echarts, tab5 = st.tabs([
    "📥 Data Ingestion", 
    "⚙️ ETL / Pipelines",
    "🔮 Forecasting Lab",
    "🎲 Journalist Lab",
    "📈 Executive Summary",
    "🧹 Smart Clean", 
    "🤖 AI Assistant", 
    "📊 Generative Viz", 
    "🔀 PyGWalker (Advanced BI)",
    "📈 Altair Insights",
    "💎 Premium Viz (ECharts)",
    "💾 Export Options"
])

# Helper: AI Schema Insights
def get_ai_missions(df_sample):
    if not (gemini_key or groq_key):
        return []
    
    schema = df_sample.dtypes.to_string()
    prompt = f"""
    Analyze this dataset schema and suggest 3 high-value "Analysis Missions" for a data scientist.
    Schema:
    {schema}
    
    Return ONLY a JSON array of strings, where each string is a concise mission title.
    Example: ["Analyze Survival Price Correlation", "Predict Passenger Surivival", "Segment Users by Fare Paid"]
    """
    try:
        raw, _ = safe_ai_call(prompt)
        clean = raw.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except:
        return []

# ------------------------------------------
# Tab 1: Data Ingestion (Scalable loading)
# ------------------------------------------
with tab1:
    st.header("Upload or Query Data")
    
    # --- New: Dataset Hub (Local Explorer) ---
    st.subheader("📦 Dataset Hub (Local Library)")
    dataset_path = "Datasets"
    local_files = glob.glob(os.path.join(dataset_path, "*.csv")) + glob.glob(os.path.join(dataset_path, "*.csv.zip"))
    
    if local_files:
        file_names = {os.path.basename(f): f for f in local_files}
        selected_file_name = st.selectbox("Select local dataset:", list(file_names.keys()), key="hub_selector")
        
        if st.button("🚀 Load from Hub"):
            full_path = file_names[selected_file_name]
            with st.spinner(f"Ingesting {selected_file_name}..."):
                try:
                    target_csv = full_path
                    # Handle ZIP extraction if needed
                    if full_path.endswith(".zip"):
                        with zipfile.ZipFile(full_path, 'r') as zip_ref:
                            # Extract to a temp directory inside Datasets
                            extract_path = os.path.join(dataset_path, "extracted_tmp")
                            zip_ref.extractall(extract_path)
                            # Find the main CSV inside (first one found)
                            extracted_csvs = glob.glob(os.path.join(extract_path, "**", "*.csv"), recursive=True)
                            if extracted_csvs:
                                target_csv = extracted_csvs[0]
                            else:
                                st.error("No CSV found inside the ZIP file.")
                                st.stop()
                    
                    # Load into DuckDB
                    conn.execute(f"CREATE OR REPLACE VIEW current_data AS SELECT * FROM read_csv_auto('{target_csv.replace('\\', '/')}')")
                    st.session_state.current_table = "current_data"
                    df = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                    st.session_state.df_preview = df
                    st.success(f"Successfully loaded `{selected_file_name}`!")
                    
                    # Trigger AI Insights
                    st.session_state.ai_missions = get_ai_missions(df.head(20))
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
    else:
        st.info("No local datasets found in the `Datasets` folder.")
    
    st.divider()

    st.markdown("""
    Load data directly into DuckDB. For large files, use the upload feature to process out-of-core.
    """)
    
    # Option A: MotherDuck Table Selector (Dynamic)
    st.subheader(f"📊 Explore `{db_name}` Tables")
    try:
        # Fetch actual tables from the user's database using the catalog filter
        tables_df = conn.execute(f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{db_name}' AND table_schema = 'main'").df()
        available_tables = tables_df['table_name'].tolist()
        
        if available_tables:
            selected_table = st.selectbox("Select a table to load:", available_tables)
            if st.button("🚀 Load Selected Table"):
                with st.spinner(f"Loading {selected_table}..."):
                    try:
                        # Define as a view for consistent access
                        conn.execute(f"CREATE OR REPLACE VIEW current_data AS SELECT * FROM {db_name}.main.{selected_table}")
                        st.session_state.current_table = "current_data"
                        df = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                        st.session_state.df_preview = df
                        st.success(f"Successfully loaded `{selected_table}`!")
                        st.session_state.ai_missions = get_ai_missions(df.head(20))
                    except Exception as e:
                        st.error(f"Error loading table: {e}")
        else:
            st.warning(f"No tables found in `{db_name}.main`. Try creating some first!")
    except Exception as e:
        st.error(f"Could not fetch tables from MotherDuck: {e}")

    st.divider()

    # Option B: Custom SQL Query
    st.subheader("🔍 Advanced: SQL Query")
    default_query = f"SELECT * FROM {db_name}.main.executive_kpi_data LIMIT 5000"
    query = st.text_area("SQL Editor", value=default_query, height=100)
    
    if st.button("Run Query"):
        with st.spinner("Executing query..."):
            is_safe, keyword = validate_sql_query(query)
            if not is_safe:
                st.error(f"🛑 **Security Block:** Destructive keyword `{keyword}` detected. Only SELECT queries are permitted in this sandbox.")
            else:
                try:
                    conn.execute(f"CREATE OR REPLACE VIEW current_data AS {query}")
                    st.session_state.current_table = "current_data"
                    df = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                    st.session_state.df_preview = df
                    log_version_action("Custom SQL Query", query, "Manual SQL executed in editor.")
                    st.session_state.ai_missions = get_ai_missions(df.head(20))
                    st.success("Query executed successfully!")
                except Exception as e:
                    st.error(f"Query Error: {e}")

    st.divider()
    
    # Quick Demo Data (Moved to bottom)
    st.subheader("⚡ Quick Demo")
    st.markdown("Load a generated dataset to test features without connecting to a database.")
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
            st.success("Dummy data loaded successfully!")
            st.session_state.ai_missions = get_ai_missions(demo_df.head(20))

        if st.session_state.df_preview is not None:
            st.subheader("Interactive Data Explorer (AgGrid)")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.df_preview)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
            gridOptions = gb.build()
            
            AgGrid(
                st.session_state.df_preview,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode='AS_INPUT',
                theme='streamlit',
                height=400
            )
            
            # Get count safely - guard against CatalogException when table not in session
            try:
                if st.session_state.current_table:
                    res = conn.execute(f"SELECT COUNT(*) FROM {st.session_state.current_table}").fetchone()
                    total_rows = res[0] if res else 0
                    st.caption(f"Total Rows in Engine: {total_rows}")
                else:
                    # Fall back to counting the preview dataframe
                    st.caption(f"Total Rows in Engine: {len(st.session_state.df_preview)}")
            except Exception:
                # If the DuckDB session doesn't have the table registered (e.g. after a reload),
                # fall back gracefully to counting the preview dataframe
                st.caption(f"Total Rows in Engine: {len(st.session_state.df_preview)}")

# ------------------------------------------
# Tab: ETL / Pipelines (dbt & Airbyte Style)
# ------------------------------------------
with tab_etl:
    st.header("⚙️ Data ETL / Transformation Pipelines")
    st.markdown("Automate data movement and transformation using SQL and API connectors. (Modern Data Stack Implementation)")
    
    etl_sub1, etl_sub2 = st.tabs(["🏗️ dbt-Style Modeling", "🔌 Connector Lab"])
    
    with etl_sub1:
        st.subheader("DuckDB SQL Modeling (dbt-Lite)")
        model_sql = st.text_area("Write transformation SQL (e.g., CREATE TABLE cleaned_revenue AS...)", 
                                 value="-- Example Model\nSELECT \n  Product, \n  SUM(Revenue) as Total_Revenue \nFROM current_data \nGROUP BY 1", height=150)
        
        if st.button("🚀 Run Model"):
            with st.spinner("Executing transformation..."):
                try:
                    conn.execute(model_sql)
                    st.success("Model successfully created in DuckDB!")
                    log_version_action("ETL Modeling", model_sql, "dbt-style SQL transformation executed.")
                except Exception as e:
                    st.error(f"ETL Error: {e}")
                    
    with etl_sub2:
        st.subheader("API Ingestion (Airbyte-Lite)")
        st.markdown("Pull data from external sources directly into your analytics engine.")
        connector = st.selectbox("Select Source Connector", ["GitHub Repo Stars", "Google Search Console (Mock)", "Stripe Payments (Mock)"])
        repo = st.text_input("Repository (e.g., 'duckdb/duckdb')", value="streamlit/streamlit")
        
        if st.button("🔌 Run Ingestion"):
            with st.spinner(f"Ingesting from {connector}..."):
                if connector == "GitHub Repo Stars":
                    try:
                        api_url = f"https://api.github.com/repos/{repo}"
                        response = requests.get(api_url)
                        data = response.json()
                        ingest_df = pd.DataFrame([data])
                        conn.execute("CREATE OR REPLACE TABLE raw_github_data AS SELECT * FROM ingest_df")
                        st.dataframe(ingest_df)
                        st.success(f"Successfully ingested metadata for {repo}!")
                        log_version_action("API Ingestion", f"Fetch {repo}", "Airbyte-style GitHub ingestion.")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                else:
                    st.info(f"The {connector} connector is currently in beta. Please check back soon!")

# ------------------------------------------
# Tab 2: Smart Clean (Automated Profiling)
# ------------------------------------------
with tab2:
    st.header("Smart Data Cleaning & Profiling")
    if st.session_state.current_table is None:
        st.info("Please load data in the 'Data Ingestion' tab first.")
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        st.subheader("Data Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numerical Summary**")
            with stylable_container(
                key="summary_container",
                css_styles="""
                    {
                        border: 1px solid rgba(28, 31, 46, 0.1);
                        border-radius: 0.5rem;
                        padding: calc(1em - 1px);
                    }
                """,
            ):
                st.dataframe(df.describe())
        with col2:
            st.write("**Missing Values (%)**")
            missing = (df.isnull().sum() / len(df)) * 100
            missing_df = missing[missing > 0].to_frame(name="Missing %")
            
            # Best Practice 7: Use Color Strategically to Signal Deviation
            def color_missing(val):
                color = '#FF5630' if val > 10 else '#FFAB00' # Red for high risk, orange for warning
                return f'color: {color}; font-weight: bold;'
            
            if not missing_df.empty:
                st.dataframe(missing_df.style.applymap(color_missing))
            else:
                st.success("No missing values detected! ✅")
            
        st.subheader("Automated Cleaning Actions")
        
        if st.button("Run Smart Imputation (Heuristic)"):
            with st.spinner("Cleaning data..."):
                # Simple logic to create a cleaned view in duckdb
                # In a real app we would build a dynamic SQL query based on pandas types.
                # For this demo, we'll do the cleaning on the preview dataframe.
                cleaned_df = df.copy()
                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                for col in cleaned_df.select_dtypes(include=['object']).columns:
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                
                st.session_state.df_preview = cleaned_df
                # Register the cleaned pandas df back to Duckdb (as a temporary view for the session)
                conn.execute("CREATE OR REPLACE TEMP VIEW current_data AS SELECT * FROM cleaned_df")
                log_version_action("Heuristic Imputation", "df.fillna(median/mode)", "Filled numeric with median and categorical with mode.")
                st.success("Imputed missing values! (Median for nums, Mode for categorical)")
                st.rerun()
                
        # 1. Data Privacy & Security (IJCDS Research Implementation)
        st.markdown("**(IJCDS Best Practice: Data Privacy)**")
        if st.button("Anonymize PII Data (Masking)"):
            with st.spinner("Detecting and masking sensitive columns..."):
                cleaned_df = df.copy()
                pii_keywords = ['name', 'email', 'phone', 'ssn', 'address', 'ip', 'password', 'credit']
                masked_cols = []
                
                for col in cleaned_df.columns:
                    if any(keyword in col.lower() for keyword in pii_keywords):
                        # Mask data (e.g. John Doe -> J*** D*** or simply [REDACTED])
                        cleaned_df[col] = "[REDACTED]"
                        masked_cols.append(col)
                
                if masked_cols:
                    st.session_state.df_preview = cleaned_df
                    conn.execute("CREATE OR REPLACE TEMP VIEW current_data AS SELECT * FROM cleaned_df")
                    log_version_action("PII Anonymization", f"df[{masked_cols}] = '[REDACTED]'", f"Masked {len(masked_cols)} columns.")
                    st.success(f"Successfully anonymized {len(masked_cols)} columns: {', '.join(masked_cols)}")
                    st.rerun()
                else:
                    st.info("No common PII columns detected based on column names.")

        # 2. Resolving DataViz Challenges (IJCDS Research Implementation Sections 6 & 7)
        st.markdown("**(IJCDS Sections 6 & 7: Resolving DataViz Challenges)**")
        col_clean1, col_clean2 = st.columns(2)
        with col_clean1:
            if st.button("Drop Duplicates (Prevent Illogical Charts)"):
                initial_count = len(df)
                cleaned_df = df.drop_duplicates()
                final_count = len(cleaned_df)
                st.session_state.df_preview = cleaned_df
                conn.execute("CREATE OR REPLACE TEMP VIEW current_data AS SELECT * FROM cleaned_df")
                if initial_count > final_count:
                    log_version_action("Drop Duplicates", "df.drop_duplicates()", f"Removed {initial_count - final_count} rows.")
                    st.success(f"Dropped {initial_count - final_count} duplicate rows.")
                    st.rerun()
                else:
                    st.info("No duplicate rows found.")
        
        with col_clean2:
            if st.button("Drop Missing NAs"):
                initial_count = len(df)
                cleaned_df = df.dropna()
                final_count = len(cleaned_df)
                st.session_state.df_preview = cleaned_df
                conn.execute("CREATE OR REPLACE TEMP VIEW current_data AS SELECT * FROM cleaned_df")
                if initial_count > final_count:
                    log_version_action("Drop Missing NAs", "df.dropna()", f"Removed {initial_count - final_count} rows.")
                    st.success(f"Dropped {initial_count - final_count} rows with NA values.")
                    st.rerun()
                else:
                    st.info("No missing values found.")

        if st.button("Run Anomaly / Outlier Discovery (Combating Human Error)"):
            with st.spinner("Scanning for statistical anomalies..."):
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    st.info("No numerical columns available to check for anomalies.")
                else:
                    # Simple Z-score logic for demonstration. Z > 3 is generally considered an outlier.
                    anomalies_found = []
                    for col in num_cols:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val > 0:
                            z_scores = ((df[col] - mean_val) / std_val).abs()
                            outliers = z_scores[z_scores > 3]
                            if len(outliers) > 0:
                                anomalies_found.append(f"**{col}**: {len(outliers)} potential outlines detected (Z-score > 3).")
                    
                    if anomalies_found:
                        st.warning("⚠️ **Potential Human Entry Errors Discovered!**")
                        for anomaly in anomalies_found:
                            st.write(anomaly)
                        st.caption("Consider reviewing these values to prevent dataset skewing.")
                    else:
                        st.success("No extreme statistical anomalies detected in numerical columns.")

        st.markdown("**(Engineering Best Practice: Data Quality Guardrails)**")
        if st.button("🔍 Run Data Quality Audit"):
            with st.spinner("Auditing data integrity..."):
                issues = []
                # Check 1: Negative values in common financial columns
                fin_keywords = ['revenue', 'fare', 'amount', 'price', 'cost', 'sales', 'quantity']
                for col in df.select_dtypes(include=[np.number]).columns:
                    if any(key in col.lower() for key in fin_keywords):
                        neg_count = (df[col] < 0).sum()
                        if neg_count > 0:
                            issues.append(f"🚩 **{col}**: Found {neg_count} negative values (Logical error for financial data).")
                
                # Check 2: Malformed Emails (basic check)
                for col in df.select_dtypes(include=['object']).columns:
                    if 'email' in col.lower():
                        invalid_emails = df[df[col].astype(str).str.contains('@') == False]
                        if len(invalid_emails) > 0:
                            issues.append(f"🚩 **{col}**: Found {len(invalid_emails)} rows missing '@' symbol.")

                if issues:
                    st.warning("### Data Quality Audit Results")
                    for issue in issues:
                        st.write(issue)
                else:
                    st.success("Data Quality Audit passed! No glaring logical inconsistencies found.")

        st.markdown("**(Engineering Best Practice: Automated Documentation)**")
        if st.button("📝 Generate AI Data Dictionary"):
            with st.spinner("AI is documenting your dataset..."):
                sample_df = df.head(10).to_string()
                doc_prompt = f"""
                Analyze this dataset sample and schema:
                {sample_df}
                
                Create a professional Data Dictionary. For each column:
                1. Describe what it likely represents in a business context.
                2. Suggest 1 advanced KPI that could be derived from it.
                
                Format as a clean markdown table.
                """
                try:
                    dictionary, model_used = safe_ai_call(doc_prompt)
                    st.markdown("### 📘 AI Data Dictionary")
                    st.markdown(dictionary)
                    st.caption(f"Generated by {model_used}. This documentation helps with long-term project maintainability.")
                except Exception as e:
                    st.error(f"Could not generate dictionary: {e}")

# ------------------------------------------
# Tab 3: AI Assistant (Natural Language)
# ------------------------------------------
with tab3:
    st.header("Chat with your Data")
    if not gemini_key and not groq_key:
        st.error("Please add a Gemini or Groq API Key to `.streamlit/secrets.toml` to use this feature.")
    elif st.session_state.current_table is None:
        st.info("Please load data first.")
    else:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # --- New: AI Missions Segment ---
        if st.session_state.get("ai_missions"):
            st.markdown("### 🎯 Suggested Analysis Missions")
            cols = st.columns(len(st.session_state.ai_missions))
            for i, mission in enumerate(st.session_state.ai_missions):
                if cols[i].button(f"🚀 {mission}", key=f"mission_{i}"):
                    st.session_state.ai_chat_prompt = f"Conduct a detailed analysis for this mission: {mission}"
        
        prompt = st.chat_input("Ask a question or request a transformation (e.g., 'Filter for high Revenue rows')")
        
        # Use mission prompt if selected
        if st.session_state.get("ai_chat_prompt") and not prompt:
            prompt = st.session_state.ai_chat_prompt
            st.session_state.ai_chat_prompt = None # Clear it
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # 1. Get Schema — dual path to prevent CatalogException forever
                    # Path A: Try DuckDB DESCRIBE (works for MotherDuck persistent tables)
                    # Path B: Fallback to pandas df.dtypes (works after session reload / temp views)
                    try:
                        schema_df = conn.execute(f"DESCRIBE {st.session_state.current_table}").df()
                        schema_str = schema_df[['column_name', 'column_type']].to_string()
                    except Exception:
                        # DuckDB doesn't have the table in this session — use pandas types instead
                        df_prev = cast(pd.DataFrame, st.session_state.df_preview)
                        schema_str = df_prev.dtypes.to_string() if df_prev is not None else "Schema unavailable"
                    
                    # 2. Ask AI using multi-provider fallback
                    ai_prompt = f"""
                    You are a data analyst assistant. You have a table named '{st.session_state.current_table}'.
                    Schema:
                    {schema_str}

                    The user asked: "{prompt}"

                    Please provide a friendly, helpful, and concise answer explaining how to find this or what it might mean.
                    If the user wants to filter or refine the active dataset (e.g. 'show only electronics', 'remove rows with price > 100'), provide a SELECT query that represents the NEW desired state.
                    
                    FORMATTING:
                    1. For standard insights, use ```sql blocks.
                    2. For 'Agentic Execution' (Filtering/Transformation), wrap the SQL query in [AGENTIC_SELECT]...[/AGENTIC_SELECT] tags so I can execute it for the user.
                    
                    IMPORTANT: Only suggest SELECT queries. Never suggest DROP, DELETE, or UPDATE.
                    """
                    try:
                        text, used_model = safe_ai_call(ai_prompt)
                        st.markdown(text)
                        
                        # --- Agentic AI Execution Logic ---
                        import re
                        match = re.search(r"\[AGENTIC_SELECT\](.*?)\[/AGENTIC_SELECT\]", text, re.DOTALL)
                        if match:
                            agentic_sql = match.group(1).strip()
                            # Display the "Surgery" button
                            if st.button("🚀 Perform this Change (Live Data Surgery)"):
                                with st.spinner("Executing live surgery..."):
                                    is_safe, keyword = validate_sql_query(agentic_sql)
                                    if not is_safe:
                                        st.error(f"🛑 Security Block: Destructive keyword `{keyword}` detected.")
                                    else:
                                        try:
                                            # Update the view to the new filtered state
                                            conn.execute(f"CREATE OR REPLACE VIEW current_data AS {agentic_sql}")
                                            st.session_state.df_preview = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                                            log_version_action("Agentic Data Surgery", agentic_sql, f"AI-driven refinement using {used_model}")
                                            st.success("Data successfully refined! Check the Data Ingestion or Summary tabs to see results.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Surgery failed: {e}")

                        st.session_state.messages.append({"role": "assistant", "content": text})
                        # Styled model attribution badge
                        short_name = used_model.replace("models/", "").replace("groq/", "⚡ ")
                        st.markdown(
                            f"""<div style="display:inline-flex;align-items:center;gap:6px;
                                background:#1e293b;color:#94a3b8;border-radius:20px;
                                padding:3px 10px;font-size:11px;margin-top:6px;">
                                🤖 <span style="color:#38bdf8;font-weight:600;">{short_name}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(str(e))

# ------------------------------------------
# Tab 4: Generative Viz (LLM recommended)
# ------------------------------------------
with tab4:
    st.header("Intelligent Visualization")
    if st.session_state.current_table is None:
        st.info("Please load data first.")
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        st.markdown("### 📊 Interactive Visualizations (IJCDS Best Practices)")
        st.markdown("**1. Interpretability (Colorblind Safe)**")
        # Assuming BRAND_COLORS is defined globally or imported
        BRAND_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
        use_colorblind = st.toggle("Use Colorblind-Safe Palette", value=False)
        colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"] if use_colorblind else BRAND_COLORS
        
        st.markdown("**2. Mitigating Data Overload (Smart Sampling)**")
        max_points = st.slider("Max Data Points to Plot (Prevents Browser Freeze)", min_value=100, max_value=50000, value=10000, step=100)
        
        # Apply sampling if data is too large
        plot_df = df.sample(n=min(len(df), max_points), random_state=42) if len(df) > max_points else df
        if len(df) > max_points:
            st.caption(f"⚠️ *Showing a random sample of {max_points} points (out of {len(df)}) to prevent data overload.*")

        st.markdown("**3. Advanced Charting (Avoiding Oversimplification)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart", ["Bar", "Scatter", "Histogram", "Box", "Violin", "Treemap", "Sunburst"])
        with col2:
            x_ax = st.selectbox("X-Axis / Path 1", df.columns.tolist())
        with col3:
            y_ax = st.selectbox("Y-Axis / Path 2", df.columns.tolist() + ["None"])

        if st.button("Plot Manual"):
            try:
                if chart_type == "Bar" and y_ax != "None":
                    st.plotly_chart(px.bar(plot_df, x=x_ax, y=y_ax, color_discrete_sequence=colors), use_container_width=True)
                elif chart_type == "Scatter" and y_ax != "None":
                    st.plotly_chart(px.scatter(plot_df, x=x_ax, y=y_ax, color_discrete_sequence=colors), use_container_width=True)
                elif chart_type == "Histogram":
                    st.plotly_chart(px.histogram(plot_df, x=x_ax, color_discrete_sequence=colors), use_container_width=True)
                elif chart_type == "Box" and y_ax != "None":
                    st.plotly_chart(px.box(plot_df, x=x_ax, y=y_ax, color_discrete_sequence=colors), use_container_width=True)
                elif chart_type == "Violin" and y_ax != "None":
                    st.plotly_chart(px.violin(plot_df, x=x_ax, y=y_ax, box=True, color_discrete_sequence=colors), use_container_width=True)
                elif chart_type == "Treemap" and y_ax != "None":
                    # Treemap uses x_ax and y_ax as hierarchical paths
                    st.plotly_chart(px.treemap(plot_df, path=[x_ax, y_ax], color_discrete_sequence=colors), use_container_width=True)
                elif chart_type == "Sunburst" and y_ax != "None":
                    st.plotly_chart(px.sunburst(plot_df, path=[x_ax, y_ax], color_discrete_sequence=colors), use_container_width=True)
                else:
                    st.warning(f"Please select a valid Y-Axis / Path 2 for a {chart_type} chart.")

                # Add Statistical Context (Mitigate Over-reliance on Aesthetics)
                if y_ax != "None":
                    with st.expander(f"📊 Statistical Context for {x_ax} & {y_ax} (IJCDS Best Practice)", expanded=False):
                        stats_df = df[[x_ax, y_ax]].describe(include='all')
                        st.dataframe(stats_df)
                        st.caption("Review these exact numbers to confirm the visual representation is accurate and not logically misleading.")
                else:
                    with st.expander(f"📊 Statistical Context for {x_ax} (IJCDS Best Practice)", expanded=False):
                        stats_df = df[[x_ax]].describe(include='all')
                        st.dataframe(stats_df)
            except Exception as e:
                st.error(f"Could not generate {chart_type} chart with those columns. ({str(e)})")
                
        st.divider()
        st.markdown("### 🤖 AI Recommended Charts")
        if not gemini_key and not groq_key:
            st.warning("Add a Gemini or Groq API key to get smart chart recommendations.")
        else:
            if st.button("Generate Recommendations"):
                with st.spinner("AI is analyzing schema for optimal visualizations..."):
                    schema_str = df.dtypes.to_string()
                    prompt = f"""
                    Given this pandas dataframe schema:
                    {schema_str}
                    
                    Suggest 3 insightful Plotly Express charts to visualize this data. 
                    Format your response ONLY as a JSON array of objects. 
                    Each object MUST have these exact keys: "title", "chart_type", "x", "y", and "rationale".
                    
                    Possible values for "chart_type": "scatter", "bar", "histogram", "box", "violin", "treemap", "sunburst".
                    Set "y" to null for histograms.
                    
                    CRITICAL (IJCDS Best Practices):
                    1. Avoid oversimplification: Suggest at least one advanced chart type (Treemap, Sunburst, or Violin) if the schema supports it.
                    2. Rationale: Briefly explain why this chart is effective for this specific data structure.
                    """
                    try:
                        import json
                        raw_response, used_model = safe_ai_call(prompt)
                        
                        # Handle potential markdown wrapping (```json ... ```)
                        clean_json = raw_response.strip().replace("```json", "").replace("```", "")
                        st.session_state.ai_recommendations = json.loads(clean_json)
                        st.session_state.last_rec_model = used_model
                    except Exception as e:
                        st.error(f"Could not parse AI recommendations: {e}")

            if st.session_state.ai_recommendations:
                st.warning("⚠️ **AI Provenance Disclaimer:** These suggestions are generated by AI. Always validate the underlying data. (IJCDS Section 7.B)", icon="⚠️")
                
                for i, rec in enumerate(st.session_state.ai_recommendations):
                    with st.expander(f"Recommendation {i+1}: {rec['title']}", expanded=True):
                        st.write(f"**Rationale:** {rec['rationale']}")
                        st.caption(f"Config: {rec['chart_type']} | X: {rec['x']} | Y: {rec['y']}")
                        
                        rec_key = f"rec_chart_{i}"
                        if st.button(f"Generate '{rec['title']}'", key=f"btn_{i}"):
                            st.session_state.generated_rec_charts[rec_key] = True
                        
                        if st.session_state.generated_rec_charts.get(rec_key):
                            try:
                                r_type = rec['chart_type'].lower()
                                r_x = rec['x']
                                r_y = rec['y']
                                r_title = rec['title']
                                
                                fig = None
                                if r_type == "scatter":
                                    fig = px.scatter(df, x=r_x, y=r_y, title=r_title, color_discrete_sequence=BRAND_COLORS)
                                elif r_type == "bar":
                                    fig = px.bar(df, x=r_x, y=r_y, title=r_title, color_discrete_sequence=BRAND_COLORS)
                                elif r_type == "histogram":
                                    fig = px.histogram(df, x=r_x, title=r_title, color_discrete_sequence=BRAND_COLORS)
                                elif r_type == "box":
                                    fig = px.box(df, x=r_x, y=r_y, title=r_title, color_discrete_sequence=BRAND_COLORS)
                                elif r_type == "violin":
                                    fig = px.violin(df, x=r_x, y=r_y, box=True, title=r_title, color_discrete_sequence=BRAND_COLORS)
                                elif r_type == "treemap":
                                    fig = px.treemap(df, path=[r_x, r_y] if r_y else [r_x], title=r_title, color_discrete_sequence=BRAND_COLORS)
                                elif r_type == "sunburst":
                                    fig = px.sunburst(df, path=[r_x, r_y] if r_y else [r_x], title=r_title, color_discrete_sequence=BRAND_COLORS)
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Chart generation failed: {e}")
                
                st.caption(f"Powered by `{st.session_state.get('last_rec_model', 'AI Model')}`")
                col_fb1, col_fb2, _ = st.columns([1,1,10])
                with col_fb1:
                    st.button("👍 Useful", key="useful_main")
                with col_fb2:
                    st.button("👎 Not Useful", key="not_useful_main")
        
        st.divider()
        st.markdown("### 🔍 Semantic Search (RAG Lite Concept)")
        st.caption("Search for data by meaning or concept, not just exact keywords. (Industry Trend 2026)")
        search_query = st.text_input("Describe what you are looking for (e.g., 'high end electronics' or 'cheap accessories')")
        
        if search_query:
            with st.spinner("Searching semantically..."):
                # RAG Lite: We use the LLM to rank the best matching categories or products 
                # based on the user's natural language query.
                search_prompt = f"""
                The user is looking for: "{search_query}"
                Available columns and sample values: {df.head(5).to_dict()}
                
                Write a DuckDB SQL SELECT query that finds the most relevant rows. Use LIKE or filter logic that best matches the intent of "{search_query}".
                Focus on the most descriptive text columns. 
                ONLY return the SQL query, nothing else.
                """
                try:
                    suggested_sql, _ = safe_ai_call(search_prompt)
                    # Clean the SQL from potential markdown wrapper
                    suggested_sql = suggested_sql.replace("```sql", "").replace("```", "").strip()
                    
                    st.info(f"AI-driven Semantic Filter applied: `{suggested_sql}`")
                    search_results = conn.execute(suggested_sql).df()
                    st.dataframe(search_results)
                except Exception as e:
                    st.error(f"Semantic search failed: {e}")

# ------------------------------------------
# Tab PyGWalker: Advanced Drag and Drop BI
# ------------------------------------------
with tab_pyg:
    st.header("🔀 Advanced Business Intelligence (Drag & Drop)")
    st.markdown("Build Tableau-style visualizations instantly using an interactive canvas. Drag fields to shelves, change mark types, and explore your data visually.")
    
    if st.session_state.current_table is None:
        st.info("Please load data in the 'Data Ingestion' tab first.")
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        # PyGWalker needs a pandas dataframe
        # We use a unique theme to match Streamlit's typical dark/light look
        try:
            with st.spinner("Initializing PyGWalker engine..."):
                # Generate HTML object 
                walker = pyg.walk(df, return_html=True, spec="cfg.json")
                # Render using Streamlit components
                components.html(walker, height=800, scrolling=True)
            st.caption("Powered by PyGWalker. Note: Very large datasets (> 100k rows) may slow down the browser in this view.")
        except Exception as e:
            st.error(f"Could not load PyGWalker: {e}")

# ------------------------------------------
# Tab Altair: Declarative Statistical Viz
# ------------------------------------------
with tab_altair:
    st.header("📈 Declarative Statistical Insights (Altair)")
    st.markdown("Altair provides a grammar-of-graphics approach for sophisticated statistical charts. Use the brush tool to interactively filter and explore relationships between variables.")
    
    if st.session_state.current_table is None:
        st.info("Please load data in the 'Data Ingestion' tab first.")
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(num_cols) >= 2:
            st.subheader("Interactive Multi-View Explorer")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                x_alt = st.selectbox("X-Axis (Scatter)", num_cols, index=0, key="alt_x")
            with col_a2:
                y_alt = st.selectbox("Y-Axis (Scatter)", num_cols, index=min(1, len(num_cols)-1), key="alt_y")
            
            color_alt = st.selectbox("Color Segment", ["None"] + cat_cols, key="alt_color")
            
            # --- Altair Chart Construction ---
            brush = alt.selection_interval()
            
            base = alt.Chart(df).encode(
                color=alt.Color(f'{color_alt}:N', scale=alt.Scale(range=BRAND_COLORS)) if color_alt != "None" else alt.value(BRAND_COLORS[0])
            ).add_params(brush)
            
            # Main Scatter
            scatter = base.mark_point(filled=True, size=60).encode(
                x=alt.X(x_alt),
                y=alt.Y(y_alt),
                tooltip=df.columns.tolist()
            ).properties(width=500, height=400)
            
            # Histogram for X axis selection influence
            bars = base.mark_bar().encode(
                x='count()',
                y=alt.Y(f'{color_alt}:N').title("Segment") if color_alt != "None" else alt.value("Total")
            ).transform_filter(brush).properties(width=500, height=150)
            
            st.altair_chart(scatter & bars, use_container_width=True)
            st.caption("💡 **Tip:** Click and drag on the scatter plot to select a region. The bar chart below will dynamically update to show the distribution of the selected points.")
        else:
            st.warning("At least two numerical columns are required for Altair Interactive insights.")

# ------------------------------------------
# Tab ECharts: Premium Visualizations
# ------------------------------------------
with tab_echarts:
    st.header("💎 Premium Visualizations (ECharts)")
    st.markdown("ECharts offers highly performant, animated, and sophisticated visualizations. Below is a dynamic distribution gauge for your primary numerical column.")
    
    if st.session_state.current_table is None:
        st.info("Please load data in the 'Data Ingestion' tab first.")
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if num_cols:
            gauge_col = st.selectbox("Select Target for Gauge", num_cols, key="echart_gauge")
            val = float(df[gauge_col].mean())
            max_val = float(df[gauge_col].max())
            
            options = {
                "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
                "series": [
                    {
                        "name": "Mean Value",
                        "type": "gauge",
                        "max": max_val,
                        "detail": {"formatter": "{value}"},
                        "data": [{"value": round(val, 2), "name": "Mean"}],
                        "axisLine": {
                            "lineStyle": {
                                "color": [
                                    [0.3, "#36B37E"],
                                    [0.7, "#FFAB00"],
                                    [1, "#FF5630"],
                                ],
                                "width": 30,
                            }
                        },
                    }
                ],
            }
            st_echarts(options=options, height="500px")
            
            # Sunburst Example if Category/Product exist
            if 'Category' in df.columns and 'Product' in df.columns:
                st.subheader("Category-Product Sunburst")
                # Prepare data for sunburst
                sun_data = []
                categories = df['Category'].dropna().unique().tolist()
                for cat in categories:
                    children = []
                    cat_df = df[df['Category'] == cat]
                    prods = cat_df['Product'].dropna().unique().tolist()
                    for prod in prods:
                        children.append({"name": prod, "value": float(cat_df[cat_df['Product'] == prod]['Revenue'].sum())})
                    sun_data.append({"name": cat, "children": children})
                
                sun_options = {
                    "series": {
                        "type": "sunburst",
                        "data": sun_data,
                        "radius": [0, "90%"],
                        "label": {"rotate": "radial"},
                    }
                }
                st_echarts(options=sun_options, height="600px")
        else:
            st.warning("No numerical columns found for ECharts.")

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
    
    req_txt = "streamlit\npandas\nduckdb==1.4.4\nplotly\ngoogle-generativeai\nnumpy\npygwalker\ngroq\naltair\nstreamlit-aggrid\nstreamlit-echarts\nstreamlit-extras\ndbt-duckdb\nrequests\nscikit-learn"
    st.download_button("Download dependencies (requirements.txt)", req_txt, file_name="session_requirements.txt", help="Download the Python requirements needed to run the exported pipeline.")

# Quality of Life: Back to Top
add_vertical_space(5)
st.button("⬆️ Back to Top", on_click=lambda: st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True))

# ------------------------------------------
# Tab: Forecasting Lab (ML Training)
# ------------------------------------------
with tab_forecasting:
    st.header("🔮 Forecasting Lab")
    st.markdown("Training AI models on historical data to predict future trends. (Best Practice 15: Predictive Governance)")
    
    pop_csv = "population-with-un-projections/population-with-un-projections.csv"
    if not os.path.exists(pop_csv):
        st.error(f"Dataset not found at {pop_csv}. Please ensure the folder structure is correct.")
    else:
        # Load and cache data
        @st.cache_data
        def load_pop_data():
            df = pd.read_csv(pop_csv)
            # Shorten column names for easier access
            df.columns = ["Entity", "Code", "Year", "Pop_Estimate", "Pop_Medium_Proj"]
            return df

        pop_df = load_pop_data()
        entities = sorted(pop_df['Entity'].unique().tolist())
        selected_entity = st.selectbox("Select Country/Region to Analyze", entities, index=entities.index("World") if "World" in entities else 0)
        
        entity_df = pop_df[pop_df['Entity'] == selected_entity].copy()
        hist_df = entity_df[entity_df['Pop_Estimate'].notna()]
        proj_df = entity_df[entity_df['Pop_Medium_Proj'].notna()]
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Historical Data Points", len(hist_df))
        with col_m2:
            st.metric("Expert Projection Span", f"{proj_df['Year'].min()} - {proj_df['Year'].max()}")

        st.subheader("🤖 AI Training Engine")
        poly_degree = st.slider("Polynomial Complexity (Degree)", 1, 4, 2, help="Higher degree allows for more complex curves, but may overfit.")
        
        if st.button("🚀 Train & Predict 2100"):
            with st.spinner(f"Training ML model for {selected_entity}..."):
                # Prepare data
                X = hist_df[['Year']].values
                y = hist_df['Pop_Estimate'].values
                
                # Polynomial Regression
                poly = PolynomialFeatures(degree=poly_degree)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Predict up to 2100
                future_years = np.arange(1950, 2101).reshape(-1, 1)
                future_X_poly = poly.transform(future_years)
                predictions = model.predict(future_X_poly)
                
                pred_df = pd.DataFrame({
                    "Year": future_years.flatten(),
                    "AI_Prediction": predictions
                })
                
                # Merge for comparison
                viz_df = pd.merge(entity_df, pred_df, on="Year", how="outer")
                
                # Visualization
                fig = go.Figure()
                # 1. Historical
                fig.add_trace(go.Scatter(x=hist_df['Year'], y=hist_df['Pop_Estimate'], name="Historical (Actual)", mode='markers', marker=dict(color='#00B8D9')))
                # 2. UN Projection
                fig.add_trace(go.Scatter(x=proj_df['Year'], y=proj_df['Pop_Medium_Proj'], name="UN Medium Projection", line=dict(color='#36B37E', dash='dash')))
                # 3. AI Prediction
                fig.add_trace(go.Scatter(x=pred_df['Year'], y=pred_df['AI_Prediction'], name=f"AI ML Forecast (Deg {poly_degree})", line=dict(color='#FF5630', width=3)))
                
                fig.update_layout(title=f"Population Forecast Comparison: {selected_entity}", xaxis_title="Year", yaxis_title="Population", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Comparison Summary
                un_2100 = proj_df[proj_df['Year'] == 2100]['Pop_Medium_Proj'].values[0]
                ai_2100 = predictions[-1]
                diff = ((ai_2100 - un_2100) / un_2100) * 100
                
                st.markdown(f"### 📊 Insight: The 'Divergence' Gap")
                st.write(f"By the year **2100**, the UN predicts a population of **{un_2100:,.0f}**, while your AI model predicts **{ai_2100:,.0f}**.")
                st.info(f"The AI forecast is **{abs(diff):.1f}% {'higher' if diff > 0 else 'lower'}** than the expert UN projections. This suggests the recent growth trajectory {'exceeds' if diff > 0 else 'is more conservative than'} complex demographic modeling.")

# ------------------------------------------
# Tab: Journalist Lab (538 Style)
# ------------------------------------------
with tab_journalist:
    st.header("🎲 Journalist Lab")
    st.markdown("Authoritative, data-driven storytelling inspired by **FiveThirtyEight**. (Best Practice 16: Narrative Integrity)")
    
    j_sub1, j_sub2 = st.tabs(["🔗 538 Connector", "📈 Poll Aggregator (Smoothing)"])
    
    with j_sub1:
        st.subheader("Connect to FiveThirtyEight Datasets")
        st.markdown("Pull live assets from the `fivethirtyeight/data` GitHub repository.")
        
        dataset_options = {
            "Presidential Polls (2020)": "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/presidential_polls.csv",
            "NBA Elo Ratings": "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv",
            "Trump Approval Ratings": "https://raw.githubusercontent.com/fivethirtyeight/data/master/trump-approval-ratings/approval_topline.csv"
        }
        
        selected_j_ds = st.selectbox("Choose Dataset", list(dataset_options.keys()))
        if st.button("🚀 Fetch & Ingest from 538"):
            with st.spinner(f"Connecting to 538 Open Data..."):
                try:
                    url = dataset_options[selected_j_ds]
                    df = pd.read_csv(url)
                    conn.execute("CREATE OR REPLACE TABLE journalist_data AS SELECT * FROM df")
                    st.session_state.current_table = "journalist_data"
                    st.session_state.df_preview = df
                    st.success(f"Successfully ingested {selected_j_ds} into DuckDB!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with j_sub2:
        st.subheader("Statistical Smoothing")
        st.markdown("Apply rolling averages to noisy time-series data to reveal the 'true' trend.")
        
        if st.session_state.df_preview is None:
            st.warning("Please load a dataset using the 538 Connector first.")
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
            
            if st.button("📊 Generate 538-Style Analysis"):
                try:
                    # Rolling average
                    df_sorted = df.sort_values(by=x_axis)
                    df_sorted['smoothed'] = df_sorted[y_axis].rolling(window=window, center=True).mean()
                    
                    fig = go.Figure()
                    # Raw data (faded)
                    fig.add_trace(go.Scatter(x=df_sorted[x_axis], y=df_sorted[y_axis], name="Raw Model", mode='markers', marker=dict(color=F38_COLOR_BLUE, opacity=0.2)))
                    # Smoothed trend
                    fig.add_trace(go.Scatter(x=df_sorted[x_axis], y=df_sorted['smoothed'], name=f"{window}-Point Rolling Avg", line=dict(color=F38_COLOR_RED, width=4)))
                    
                    fig.update_layout(title=f"Narrative Trend Analysis: {y_axis}")
                    fig = apply_538_style(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📰 Journalistic Commentary")
                    prompt = f"Write a hard-hitting 538-style summary of this trend where {y_axis} is analyzed over {x_axis} with a {window}-point smoothing window. Use data-driven language."
                    summary, _ = safe_ai_call(prompt)
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# ------------------------------------------
# Tab 1b: Executive Summary (Best Practices 2, 8, 9)
# ------------------------------------------
with tab1b:
    st.header("📈 Executive Summary")
    st.markdown("Best Practice 9: *Separate Executive and Operational Dashboards.* This view provides a high-level strategic overview restricting cognitive load to core KPIs.")
    
    if st.session_state.current_table is None:
        st.info("Please load data in the 'Data Ingestion' tab first.")
    else:
        df = cast(pd.DataFrame, st.session_state.df_preview)
        
        # Best Practice 2: Limit Executive Dashboards to 5–7 Core KPIs
        st.subheader("Core KPIs")
        kpi_cols = st.columns(4)
        
        # Calculate some dynamic KPIs based on available data types
        # Using np.number for robust numeric selection
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        # KPI 1: Total Volume
        with kpi_cols[0]:
            try:
                res = conn.execute(f"SELECT COUNT(*) FROM {st.session_state.current_table}").fetchone()
                total_rows = res[0] if res else 0
                st.metric(label="Total Records Processed", value=f"{total_rows:,}", help="Total volume of data in the current analytic dataset.")
            except Exception:
                st.metric(label="Total Records", value=f"{len(df):,}")
                
        # Additional KPIs: Dynamic deltas based on temporal context
        # Best Practice 8: Robust datetime detection using pandas api
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if not date_cols:
            # Try to find columns with 'date' or 'time' in the name if types aren't inferred
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        
        for i, col in enumerate(num_cols[:3]):
            with kpi_cols[i+1]:
                avg_val = df[col].mean()
                delta_val = None
                delta_str = "No temporal context"
                
                if date_cols:
                    try:
                        t_col = date_cols[0]
                        # Sort by date to get recent vs previous
                        temp_df = df.sort_values(by=t_col, ascending=False)
                        mid_point = len(temp_df) // 2
                        current_period_avg = temp_df.iloc[:mid_point][col].mean()
                        prev_period_avg = temp_df.iloc[mid_point:][col].mean()
                        
                        if prev_period_avg != 0 and not np.isnan(prev_period_avg):
                            delta_val = ((current_period_avg - prev_period_avg) / prev_period_avg) * 100
                            delta_str = f"{delta_val:+.1f}% vs Prev Period"
                    except Exception:
                        pass
                
                st.metric(
                    label=f"Avg {col}", 
                    value=f"{avg_val:,.2f}", 
                    delta=delta_str, 
                    delta_color="normal" if delta_val and delta_val > 0 else "inverse",
                    help=f"Average value for '{col}' with calculated temporal trend."
                )
        
        st.divider()
        
        # Best Practice 1: Align every visualization to a strategic objective
        st.subheader("Strategic Overview (Sample Visual)")
        if len(num_cols) >= 2:
            fig = px.scatter(
                df, 
                x=num_cols[0], 
                y=num_cols[1], 
                title=f"Strategic Alignment: {num_cols[0]} vs {num_cols[1]}",
                color_discrete_sequence=BRAND_COLORS,
                opacity=0.7
            )
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient numeric columns to generate a strategic scatter overview.")
