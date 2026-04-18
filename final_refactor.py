import os

file_path = "c:/Users/Satvik Shrivastava/OneDrive/Desktop/Antigravity/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False

# We want to remove blocks from line 193 onwards, but keep the tab variable definitions
# and the new modularized calls we already added.

# Actually, it's easier to just rebuild the part from with tab1 onwards.
# Let's see where with tab1 starts.

start_idx = -1
for i, line in enumerate(lines):
    if "with tab1:" in line:
        start_idx = i
        break

if start_idx != -1:
    # Keep everything before with tab1
    new_lines = lines[:start_idx]
    
    # Add new modularized calls
    modular_calls = """
# ------------------------------------------
# Main Tabbed Interface Implementation
# ------------------------------------------

with tab1:
    from views.tab_ingestion import render_tab_ingestion
    render_tab_ingestion(conn, db_name, get_ai_missions)

with tab_etl:
    from views.tab_etl import render_tab_etl
    render_tab_etl(conn)


with tab_forecasting:
    from views.tab_forecasting import render_tab_forecasting
    render_tab_forecasting()

with tab_journalist:
    from views.tab_journalist import render_tab_journalist
    render_tab_journalist(conn)

with tab1b:
    from views.tab_summary import render_tab_summary
    render_tab_summary(conn)

with tab2:
    from views.tab_cleaning import render_tab_cleaning
    render_tab_cleaning(conn)

with tab3:
    from views.tab_assistant import render_tab_assistant
    render_tab_assistant(conn, gemini_key, groq_key)

with tab4:
    from views.tab_generative_viz import render_tab_generative_viz
    render_tab_generative_viz(conn, gemini_key, groq_key, config.BRAND_COLORS)

with tab_pyg:
    from views.tab_visualizations import render_tab_pyg
    render_tab_pyg()

with tab_altair:
    from views.tab_visualizations import render_tab_altair
    render_tab_altair(config.BRAND_COLORS)

with tab_echarts:
    from views.tab_visualizations import render_tab_echarts
    render_tab_echarts()

with tab_dashboard:
    from views.tab_visualizations import render_tab_dashboard
    render_tab_dashboard()

with tab5:
    from views.tab_export import render_tab_export
    render_tab_export()

# Quality of Life: Back to Top
add_vertical_space(5)
st.button("⬆️ Back to Top", on_click=lambda: st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True))
"""
    new_lines.append(modular_calls)

    # Also need to remove apply_538_style definition from the top
    final_lines = []
    skip_style = False
    for line in new_lines:
        if "def apply_538_style(fig):" in line:
            skip_style = True
        if skip_style:
            if line.strip() == "" or (not line.startswith(" ") and not line.startswith("def") and not line.startswith("#") and "return fig" in line):
                 # This is tricky without a real parser, but let's just find the end
                 if "return fig" in line:
                     skip_style = False
                 continue
        final_lines.append(line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    print("Successfully refactored app.py")
else:
    print("Could not find start of tabs")
