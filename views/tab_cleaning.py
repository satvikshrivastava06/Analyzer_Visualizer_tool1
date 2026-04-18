import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from typing import cast
from modules.cleaning import smart_impute, drop_duplicates_secure
from modules.ai_engine import safe_ai_call
from ui_theme import page_hero, section_header, stat_card, anomaly_row, action_card, TEXT_3, SURFACE_3, BORDER

def render_tab_cleaning(conn):
    st.markdown(page_hero("✦", "Smart Clean", "Profile data quality, detect anomalies, and apply AI-driven transformations."), unsafe_allow_html=True)
    
    # ── Notification Buffer (Flash Messages) ──
    if "flash_msg" in st.session_state and st.session_state.flash_msg:
        st.markdown(st.session_state.flash_msg, unsafe_allow_html=True)
        # We don't clear immediately to allow user to see it. 
        # But we provide a clear button OR it clears on next meaningful interaction.
        if st.button("Clear Notification ✕", key="clear_flash"):
            st.session_state.flash_msg = None
            st.rerun()

    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Go to the Ingest tab to load a dataset first."), unsafe_allow_html=True)
        return

    df = cast(pd.DataFrame, st.session_state.df_preview)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🏥", "Data Health Profile", "Current dataset metrics"), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(stat_card("Rows", f"{len(df):,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_card("Columns", str(len(df.columns))), unsafe_allow_html=True)
    with col3:
        null_cnt = int(df.isnull().sum().sum())
        st.markdown(stat_card("Missing", f"{null_cnt:,}", delta="Attention" if null_cnt > 0 else "Perfect", delta_type="neg" if null_cnt > 0 else "neu"), unsafe_allow_html=True)
    with col4:
        dup_cnt = int(df.duplicated().sum())
        st.markdown(stat_card("Duplicates", f"{dup_cnt:,}", delta="Attention" if dup_cnt > 0 else "Perfect", delta_type="neg" if dup_cnt > 0 else "neu"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["📊 Profile Details", "🔍 Missing Fields"])
    with t1:
        try:
            # Robust profiling: Describe all types, then reformat for display
            pdf = df.describe(include='all')
            # If all categorical, describe() returns different indices. We fillna for visual comfort.
            pdf = pdf.fillna('-').astype(str)
            st.dataframe(pdf, width='stretch', height=450)
            st.caption("Auto-generated statistical summary across all data types.")
        except Exception as e:
            st.markdown(anomaly_row(f"Profiling Error: {e}", "warning", "Ensure column types are consistent."), unsafe_allow_html=True)
            
    with t2:
        # Check if missing values exist
        null_mask = df.isnull().sum()
        missing_cols = null_mask[null_mask > 0].index.tolist()
        
        if missing_cols:
            miss_stats = []
            for col in missing_cols:
                cnt = int(null_mask[col])
                pct = (cnt / len(df)) * 100
                miss_stats.append({"Column": col, "Count": cnt, "Percentage": f"{pct:.1f}%"})
            
            mstd = pd.DataFrame(miss_stats)
            st.dataframe(mstd, width='stretch', height=450)
        else:
            st.markdown(anomaly_row("Dataset is 100% complete. No missing fields detected.", "success"), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🔍", "Deep Profiling", "Generate interactive HTML analysis report"), unsafe_allow_html=True)
    
    if st.button("Generate Sweetviz Report 📈", use_container_width=True):
        with st.spinner("Analyzing dataset with sweetviz... This might take a moment."):
            try:
                from modules.cleaning import generate_sweetviz_report
                generate_sweetviz_report(df)
                st.session_state.show_sweetviz = True
                st.session_state.flash_msg = action_card("Profiling", "Sweetviz Report Complete", "Interactive report successfully generated.")
                st.rerun()
            except Exception as e:
                st.error(f"Profiling failed. Ensure the dataset is valid: {e}")

    if st.session_state.get('show_sweetviz'):
        import os
        if os.path.exists("profile.html"):
            st.markdown("<br>", unsafe_allow_html=True)
            with open("profile.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=700, scrolling=True)
            if st.button("Close Interactive Report ✕", use_container_width=True):
                st.session_state.show_sweetviz = False
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("⚡", "Automated Data Interventions", "One-click cleaning workflows"), unsafe_allow_html=True)
    
    col_clean1, col_clean2 = st.columns(2)
    with col_clean1:
        if st.button("Run Smart Imputation ✨", use_container_width=True):
            with st.spinner("Cleaning data..."):
                cleaned_df = smart_impute(df)
                st.session_state.df_preview = cleaned_df
                # Record the persistent flash message
                st.session_state.flash_msg = action_card("Imputation", "Missing Values Resolved", "Median for numeric, Mode for categorical")
                st.rerun()
                
        if st.button("Anonymize PII Data 🛡️", use_container_width=True, help="Privacy Best Practice"):
            with st.spinner("Detecting and masking sensitive columns..."):
                from modules.cleaning import mask_pii
                cleaned_df, masked_cols = mask_pii(df)
                if masked_cols:
                    st.session_state.df_preview = cleaned_df
                    st.session_state.flash_msg = action_card("Security", "Data Masked", f"Anonymized {len(masked_cols)} columns: {', '.join(masked_cols)}")
                    st.rerun()
                else:
                    st.markdown(anomaly_row("No common PII columns detected based on column names.", "info"), unsafe_allow_html=True)

    with col_clean2:
        if st.button("Drop Duplicates 🗑️", use_container_width=True):
            cleaned_df, dropped = drop_duplicates_secure(df)
            if dropped > 0:
                st.session_state.df_preview = cleaned_df
                st.session_state.flash_msg = action_card("Cleaning", "Duplicates Removed", f"Dropped {dropped} duplicate rows.")
                st.rerun()
            else:
                st.markdown(anomaly_row("No duplicate rows found.", "info"), unsafe_allow_html=True)
        
        if st.button("Drop Missing (NAs) ✂️", use_container_width=True):
            from modules.cleaning import drop_missing_nas
            cleaned_df, dropped = drop_missing_nas(df)
            if dropped > 0:
                st.session_state.df_preview = cleaned_df
                st.session_state.flash_msg = action_card("Cleaning", "Missing NAs Dropped", f"Dropped {dropped} rows with NA values.")
                st.rerun()
            else:
                st.markdown(anomaly_row("No missing values found.", "info"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🔬", "Audits & Discovery", "Statistical checks and anomaly detection"), unsafe_allow_html=True)

    col_audit1, col_audit2 = st.columns(2)
    with col_audit1:
        if st.button("Run Statistical Anomaly Scan 🔎", use_container_width=True):
            with st.spinner("Scanning for statistical anomalies..."):
                from modules.analysis import detect_anomalies
                anomalies_found = detect_anomalies(df)
                if anomalies_found:
                    for anomaly in anomalies_found:
                        st.markdown(anomaly_row(anomaly, "warning"), unsafe_allow_html=True)
                    st.caption("Consider reviewing these values to prevent dataset skewing.")
                else:
                    st.markdown(anomaly_row("No extreme statistical anomalies detected.", "success"), unsafe_allow_html=True)

    with col_audit2:
        if st.button("Run Logic / Quality Audit 🚦", use_container_width=True):
            with st.spinner("Auditing data integrity..."):
                issues = []
                fin_keywords = ['revenue', 'fare', 'amount', 'price', 'cost', 'sales', 'quantity']
                for col in df.select_dtypes(include=[np.number]).columns:
                    if any(key in col.lower() for key in fin_keywords):
                        neg_count = (df[col] < 0).sum()
                        if neg_count > 0:
                            issues.append(f"{col}: Found {neg_count} negative values (Logical error for financial data).")
                
                for col in df.select_dtypes(include=['object']).columns:
                    if 'email' in col.lower():
                        invalid_emails = df[df[col].astype(str).str.contains('@') == False]
                        if len(invalid_emails) > 0:
                            issues.append(f"{col}: Found {len(invalid_emails)} rows missing '@' symbol.")

                if issues:
                    st.markdown("### Quality Audit Results")
                    for issue in issues:
                        st.markdown(anomaly_row(issue, "danger"), unsafe_allow_html=True)
                else:
                    st.markdown(anomaly_row("Data Quality Audit passed! No glaring logical inconsistencies.", "success"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("📝", "Documentation", "AI-Generated context dictionary"), unsafe_allow_html=True)
    
    if st.button("📝 Generate AI Data Dictionary", use_container_width=True):
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
                from ui_theme import model_badge
                st.markdown("### 📘 AI Dictionary " + model_badge(model_used), unsafe_allow_html=True)
                st.markdown(dictionary)
                st.caption("This documentation helps with long-term project maintainability.")
            except Exception as e:
                st.error(f"Could not generate dictionary: {e}")
