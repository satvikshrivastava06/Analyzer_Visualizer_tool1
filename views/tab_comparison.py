import streamlit as st
import pandas as pd
from typing import cast
from modules.analysis import compare_datasets
import plotly.express as px
from ui_theme import page_hero, section_header, anomaly_row, apply_plotly_theme, CHART_COLORS

def render_tab_comparison():
    st.markdown(page_hero("◨", "Dataset Comparison Mode", "Compare the active dataset with another to identify shifts and structural differences (A/B testing, MoM analysis)."), unsafe_allow_html=True)
    
    if st.session_state.df_preview is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Active Data", "Load an active dataset in the 'Data Ingestion' tab first."), unsafe_allow_html=True)
        return

    df_active = cast(pd.DataFrame, st.session_state.df_preview)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("1️⃣", "Upload Comparison Dataset", "Select a CSV to compare against active data"), unsafe_allow_html=True)
    comp_file = st.file_uploader("Upload CSV for comparison:", type=['csv'], key="comparison_uploader")
    
    if comp_file:
        try:
            df_comp = pd.read_csv(comp_file)
            st.markdown(anomaly_row(f"Successfully loaded comparison dataset: {comp_file.name} ({len(df_comp)} rows)", "success"), unsafe_allow_html=True)
            
            # Perform Comparison
            with st.spinner("Calculating deltas and structural shifts..."):
                results = compare_datasets(df_active, df_comp)
            
            st.divider()
            
            # --- Structural Summary ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("📐", "Structural Delta", "Row and schema differences"), unsafe_allow_html=True)
            
            from ui_theme import stat_card
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(stat_card("Active Rows", f"{results['active_rows']:,}"), unsafe_allow_html=True)
            with c2:
                row_delta_val = results['row_delta']
                delta_type = "pos" if row_delta_val > 0 else "neg"
                st.markdown(stat_card(
                    "Comparison Rows", 
                    f"{results['comparison_rows']:,}", 
                    delta=f"{row_delta_val:+,} ({results['row_delta_pct']:+.1f}%)",
                    delta_type=delta_type
                ), unsafe_allow_html=True)
            with c3:
                st.markdown(stat_card("Shared Columns", f"{len(results['shared_columns'])}"), unsafe_allow_html=True)

            # --- Column Overlap ---
            with st.expander("🛠️ Schema Overlap", expanded=False):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.write("**Only in Active Dataset:**")
                    st.write(results['only_in_active'] if results['only_in_active'] else "_None_")
                with col_c2:
                    st.write("**Only in Comparison Dataset:**")
                    st.write(results['only_in_comparison'] if results['only_in_comparison'] else "_None_")

            # --- Numeric Shifts ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("📊", "Statistical Shift", "Numeric Analysis"), unsafe_allow_html=True)
            if results['numeric_shifts']:
                shift_df = pd.DataFrame(results['numeric_shifts']).T.reset_index()
                shift_df.columns = ['Metric', 'Active Mean', 'Comparison Mean', 'Delta', 'Delta %']
                
                # Styled dataframe for deltas
                def color_delta(val):
                    color = '#34d399' if val > 0 else '#f87171' # DAVT v2 colors
                    return f'color: {color}'
                
                st.dataframe(shift_df.style.format({
                    'Active Mean': '{:,.2f}',
                    'Comparison Mean': '{:,.2f}',
                    'Delta': '{:+,.2f}',
                    'Delta %': '{:+.1f}%'
                }).applymap(color_delta, subset=['Delta %']), use_container_width=True)
                
                # Visualizing shifts
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(section_header("🎯", "Visual Drift Explorer", "Distribution comparison"), unsafe_allow_html=True)
                selected_metric = st.selectbox("Select metric to visualize drift:", shift_df['Metric'].tolist())
                
                # Distribution Comparison
                fig = px.histogram(pd.concat([
                    pd.DataFrame({selected_metric: df_active[selected_metric], "Source": "Active"}),
                    pd.DataFrame({selected_metric: df_comp[selected_metric], "Source": "Comparison"})
                ]), x=selected_metric, color="Source", barmode="overlay", title=f"Distribution Drift: {selected_metric}", color_discrete_sequence=[CHART_COLORS[0], CHART_COLORS[1]])
                
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.markdown(anomaly_row("No shared numerical columns found for statistical comparison.", "info"), unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(anomaly_row(f"Failed to process comparison dataset: {e}", "warning"), unsafe_allow_html=True)
    else:
        st.caption("Upload a file above to begin the comparison analysis.")
