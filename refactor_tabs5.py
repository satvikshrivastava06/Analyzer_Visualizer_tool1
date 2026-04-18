import sys

file_path = r"c:\Users\Satvik Shrivastava\OneDrive\Desktop\Antigravity\app.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_content = """# ------------------------------------------
# Tab PyGWalker: Advanced Drag and Drop BI
# ------------------------------------------
with tab_pyg:
    from views.tab_visualizations import render_tab_pyg
    render_tab_pyg()

# ------------------------------------------
# Tab Altair: Declarative Statistical Viz
# ------------------------------------------
with tab_altair:
    from views.tab_visualizations import render_tab_altair
    render_tab_altair(config.BRAND_COLORS)

# ------------------------------------------
# Tab ECharts: Premium Visualizations
# ------------------------------------------
with tab_echarts:
    from views.tab_visualizations import render_tab_echarts
    render_tab_echarts()

# ------------------------------------------
# Tab Dashboard: Premium Visual Dashboard
# ------------------------------------------
with tab_dashboard:
    from views.tab_visualizations import render_tab_dashboard
    render_tab_dashboard()

"""

tab_pyg_idx = -1
tab5_idx = -1

for i, line in enumerate(lines):
    if line.strip() == "# Tab PyGWalker: Advanced Drag and Drop BI":
        tab_pyg_idx = i - 1
    if line.strip() == "# Tab 5: Export / Distribution / Save State":
        tab5_idx = i - 1

if tab_pyg_idx != -1 and tab5_idx != -1:
    del lines[tab_pyg_idx:tab5_idx]
    lines.insert(tab_pyg_idx, new_content)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Refactoring visual tabs successful!")
else:
    print(f"Failed to cleanly find tab indices. tab_pyg: {tab_pyg_idx}, tab5: {tab5_idx}")
