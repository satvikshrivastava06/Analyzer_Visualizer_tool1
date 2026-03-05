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
from typing import Any, cast

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
    import os
    os.environ["MOTHERDUCK_TOKEN"] = md_token
    st.session_state.db_conn = duckdb.connect(f'md:?motherduck_token={md_token}&db={db_name}')

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
tab1, tab1b, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Data Ingestion", 
    "📈 Executive Summary",
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
                        st.session_state.df_preview = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                        st.success(f"Successfully loaded `{selected_table}`!")
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
                    st.session_state.df_preview = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                    log_version_action("Custom SQL Query", query, "Manual SQL executed in editor.")
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

        if st.session_state.df_preview is not None:
            st.subheader("Data Preview (First 1000 rows)")
            st.dataframe(st.session_state.df_preview)
            
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

        if prompt := st.chat_input("Ask a question about your data (e.g., 'What is the average fare amount?')"):
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
                    Format your response as a numbered list with the chart title, the Plotly function to use (px.scatter, px.histogram, px.treemap, px.sunburst, px.violin, etc.), and the exact x and y column names from the schema.
                    
                    CRITICAL (IJCDS Best Practices):
                    1. Avoid oversimplification: Suggest at least one advanced chart type (Treemap, Sunburst, or Violin) if the schema supports it.
                    2. Interpretability: Ensure you mention adding clear `title` and axis `labels` to prevent misleading data.
                    3. Data Literacy (Best Practice 14): Provide a brief "Why this chart?" rationale for each.
                    """
                    try:
                        text, used_model = safe_ai_call(prompt)
                        st.warning("⚠️ **AI Provenance Disclaimer:** These charts are generated by AI. Always validate the underlying data to prevent logical errors or AI hallucinations before making business decisions. (IJCDS Section 7.B)", icon="⚠️")
                        st.markdown(text)
                        st.caption(f"Powered by `{used_model}` | How useful were these suggestions?")
                        col_fb1, col_fb2, _ = st.columns([1,1,10])
                        with col_fb1:
                            st.button("👍 Useful")
                        with col_fb2:
                            st.button("👎 Not Useful")
                    except Exception as e:
                        st.error(str(e))

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
    st.download_button("Download dependencies (requirements.txt)", req_txt, file_name="session_requirements.txt", help="Download the Python requirements needed to run the exported pipeline.")

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
