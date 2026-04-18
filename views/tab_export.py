import streamlit as st
from modules.export import export_dashboard_pdf
from ui_theme import page_hero, section_header, anomaly_row

def render_tab_export():
    st.markdown(page_hero("⎋", "Reproducible Export", "Export this exact session logic as a standalone Python script or PDF."), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("💻", "Code Generation", "Standalone script"), unsafe_allow_html=True)
    
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
    st.download_button("Download Script (reproducible_analysis.py)", script_template, file_name="reproducible_analysis.py", use_container_width=True)
    
    req_txt = "streamlit\npandas\nduckdb==1.4.4\nplotly\ngoogle-generativeai\nnumpy\npygwalker\ngroq\naltair\nstreamlit-aggrid\nstreamlit-echarts\nstreamlit-extras\ndbt-duckdb\nrequests\nscikit-learn"
    st.download_button("Download dependencies (requirements.txt)", req_txt, file_name="session_requirements.txt", help="Download the Python requirements needed to run the exported pipeline.", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("📄", "One-Click PDF Report", "Export visualizations"), unsafe_allow_html=True)
    
    if 'dashboard_figs' not in st.session_state or not st.session_state.dashboard_figs:
        st.markdown(anomaly_row("No charts found. Please visit the **Visual Dashboard** tab first to generate visualizations.", "warning"), unsafe_allow_html=True)
    else:
        if st.button("🛠️ Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF... this may take a few seconds."):
                try:
                    pdf_bytes = export_dashboard_pdf(st.session_state.dashboard_figs)
                    st.download_button(
                        label="⬇️ Download PDF Dashboard",
                        data=pdf_bytes,
                        file_name="data_analyzer_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.markdown(anomaly_row("Dashboard report ready!", "success"), unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(anomaly_row(f"Failed to generate PDF: {e}", "warning"), unsafe_allow_html=True)
