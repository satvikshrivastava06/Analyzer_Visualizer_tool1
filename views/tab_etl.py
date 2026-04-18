import streamlit as st
import pandas as pd
from ui_theme import page_hero, section_header, action_card, anomaly_row, TEXT_3, SURFACE_3, BORDER
from modules.ai_engine import log_version_action

def render_tab_etl(conn):
    st.markdown(page_hero("⟳", "ETL Engine", "Design, save, and execute multi-step SQL data transformation pipelines."), unsafe_allow_html=True)
    
    if st.session_state.get("current_table") is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Go to the Ingest tab to load a dataset first."), unsafe_allow_html=True)
        return
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🛠️", "Pipeline Builder", "Chain multiple standard SQL statements"), unsafe_allow_html=True)
    
    table_name = st.session_state.current_table
    default_pipeline = f"""-- Step 1: Base projection
CREATE OR REPLACE VIEW step_1 AS 
SELECT * FROM {table_name} 
WHERE 1=1;

-- Step 2: Final Output 
-- NOTE: Last statement must create/update 'current_data' view
CREATE OR REPLACE VIEW current_data AS 
SELECT * FROM step_1;
"""
    
    if "etl_pipeline" not in st.session_state:
        st.session_state.etl_pipeline = default_pipeline
        
    pipeline_code = st.text_area("SQL Pipeline Definition (DuckDB dialect):", value=st.session_state.etl_pipeline, height=280)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Execute Pipeline", use_container_width=True):
            st.session_state.etl_pipeline = pipeline_code
            with st.spinner("Executing pipeline..."):
                try:
                    # Clean up statements to avoid executing empty strings
                    statements = []
                    current_stmt = ""
                    for line in pipeline_code.split('\\n'):
                        # Ignore pure comments for statement division
                        if line.strip().startswith('--'):
                            continue
                        current_stmt += line + "\\n"
                        if ';' in line:
                            statements.append(current_stmt)
                            current_stmt = ""
                            
                    if current_stmt.strip():
                        statements.append(current_stmt)
                        
                    for stmt in statements:
                        if stmt.strip():
                            conn.execute(stmt)
                        
                    # Update preview to reflect the pipeline results
                    st.session_state.df_preview = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                    st.session_state.current_table = "current_data"
                    log_version_action("ETL Pipeline", "Batch Execution", f"Ran {len(statements)} sequential table operations.")
                    
                    st.markdown(action_card("Success", "Pipeline executed!", "Data updated. View results in the Clean or Visualizations tabs."), unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(anomaly_row(f"Pipeline execution failed: {e}", "danger"), unsafe_allow_html=True)

    with col2:
        if st.button("💾 Save Pipeline Template", use_container_width=True):
            st.session_state.etl_pipeline_saved = pipeline_code
            st.markdown(action_card("Saved", "Pipeline Artifact Created", "Template saved to workspace session."), unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("📋", "Pipeline Best Practices", ""), unsafe_allow_html=True)
    st.info("""
    - Use **CREATE OR REPLACE VIEW step_name AS ...** to chain transformations safely.
    - End your pipeline by recreating the **current_data** view. The application relies on this view to display and interact with the data across all other analytical tabs.
    - DuckDB syntax is fully supported for advanced window functions, Regex operations, and JSON extraction.
    """)
