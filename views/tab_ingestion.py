import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import zipfile
import requests
from typing import cast
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from modules.ai_engine import validate_sql_query, log_version_action
from modules.ingestion import load_google_sheet
from ui_theme import page_hero, section_header, source_pill, chip, action_card

def render_tab_ingestion(conn, db_name, get_ai_missions_fn):
    st.markdown(page_hero("⬡", "Data Ingestion", "Connect to MotherDuck, local CSV drops, or Google Sheets."), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("📁", "Dataset Hub (Local Library)", "Load locally stored datasets"), unsafe_allow_html=True)
    
    dataset_path = "Datasets"
    local_files = glob.glob(os.path.join(dataset_path, "*.csv")) + glob.glob(os.path.join(dataset_path, "*.csv.zip"))
    
    if local_files:
        file_names = {os.path.basename(f): f for f in local_files}
        
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_file_name = st.selectbox("Select local dataset:", list(file_names.keys()), key="hub_selector", label_visibility="collapsed")
        with col_btn:
             if st.button("🚀 Load", use_container_width=True):
                 full_path = file_names[selected_file_name]
                 with st.spinner(f"Ingesting {selected_file_name}..."):
                     try:
                         target_csv = full_path
                         if full_path.endswith(".zip"):
                             with zipfile.ZipFile(full_path, 'r') as zip_ref:
                                 extract_path = os.path.join(dataset_path, "extracted_tmp")
                                 zip_ref.extractall(extract_path)
                                 extracted_csvs = glob.glob(os.path.join(extract_path, "**", "*.csv"), recursive=True)
                                 if extracted_csvs:
                                     target_csv = extracted_csvs[0]
                                 else:
                                     st.error("No CSV found inside the ZIP file.")
                                     st.stop()
                         
                         conn.execute(f"CREATE OR REPLACE VIEW current_data AS SELECT * FROM read_csv_auto('{target_csv.replace('\\\\', '/')}')")
                         st.session_state.current_table = "current_data"
                         df = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                         st.session_state.df_preview = df
                         st.success(f"Successfully loaded `{selected_file_name}`!")
                         st.session_state.ai_missions = get_ai_missions_fn(df.head(20))
                     except Exception as e:
                         st.error(f"Failed to load dataset: {e}")
    else:
        st.info("No local datasets found in the `Datasets` folder.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🦆", "DuckDB / MotherDuck Explorer", f"Explore tables in {db_name}.main"), unsafe_allow_html=True)
    
    try:
        tables_df = conn.execute(f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{db_name}' AND table_schema = 'main'").df()
        available_tables = tables_df['table_name'].tolist()
        
        if available_tables:
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                selected_table = st.selectbox("Select a table to load:", available_tables, label_visibility="collapsed")
            with col_btn:
                if st.button("🚀 Load Table", use_container_width=True):
                    with st.spinner(f"Loading {selected_table}..."):
                        try:
                            conn.execute(f"CREATE OR REPLACE VIEW current_data AS SELECT * FROM {db_name}.main.{selected_table}")
                            st.session_state.current_table = "current_data"
                            df = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                            st.session_state.df_preview = df
                            st.success(f"Successfully loaded `{selected_table}`!")
                            st.session_state.ai_missions = get_ai_missions_fn(df.head(20))
                        except Exception as e:
                            st.error(f"Error loading table: {e}")
        else:
            st.warning(f"No tables found in `{db_name}.main`. Try creating some first!")
    except Exception as e:
        st.error(f"Could not fetch tables: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("⚡", "Advanced: SQL Query", "Execute a custom DuckDB query"), unsafe_allow_html=True)

    default_query = f"SELECT * FROM {db_name}.main.executive_kpi_data LIMIT 5000"
    query = st.text_area("SQL Editor", value=default_query, height=100, label_visibility="collapsed")
    
    if st.button("Run Query ⚡", key="run_query_btn"):
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
                    st.session_state.ai_missions = get_ai_missions_fn(df.head(20))
                    st.success("Query executed successfully!")
                except Exception as e:
                    st.error(f"Query Error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🌐", "Google Sheets Connector", "Load data from a public Google Sheets URL"), unsafe_allow_html=True)
    sheet_url = st.text_input("Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0", label_visibility="collapsed")
    
    if st.button("🚀 Extract Sheet"):
        if not sheet_url:
            st.warning("Please paste a URL first.")
        else:
            with st.spinner("Extracting data from Google Sheets..."):
                try:
                    df = load_google_sheet(sheet_url)
                    conn.execute("CREATE OR REPLACE VIEW current_data AS SELECT * FROM df")
                    st.session_state.current_table = "current_data"
                    st.session_state.df_preview = df
                    st.success("Google Sheet loaded successfully!")
                    st.session_state.ai_missions = get_ai_missions_fn(df.head(20))
                    log_version_action("Google Sheets Ingestion", sheet_url, f"Loaded sheet with {len(df)} rows.")
                except Exception as e:
                    st.error(f"Failed to load Google Sheet: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🎮", "Quick Demo", "Load a generated dataset to test features"), unsafe_allow_html=True)
    if st.button("Load Dummy Data ⚡"):
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
            demo_df.loc[10:30, 'Revenue'] = np.nan
            demo_df.loc[40:60, 'Category'] = np.nan
            
            conn.execute("CREATE OR REPLACE VIEW current_data AS SELECT * FROM demo_df")
            st.session_state.current_table = "current_data"
            st.session_state.df_preview = demo_df
            st.success("Dummy data loaded successfully!")
            st.session_state.ai_missions = get_ai_missions_fn(demo_df.head(20))

        if st.session_state.df_preview is not None:
             st.markdown("<br>", unsafe_allow_html=True)
             st.markdown(section_header("👁️", "Preview", "Interactive Data Explorer (AgGrid)"), unsafe_allow_html=True)
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
             
             try:
                 if st.session_state.current_table:
                     res = conn.execute(f"SELECT COUNT(*) FROM {st.session_state.current_table}").fetchone()
                     total_rows = res[0] if res else 0
                     st.caption(f"Total Rows in Engine: {total_rows}")
                 else:
                     st.caption(f"Total Rows in Engine: {len(st.session_state.df_preview)}")
             except Exception:
                 st.caption(f"Total Rows in Engine: {len(st.session_state.df_preview)}")
