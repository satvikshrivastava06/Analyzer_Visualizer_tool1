import sys

file_path = r"c:\Users\Satvik Shrivastava\OneDrive\Desktop\Antigravity\app.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_content = """# ------------------------------------------
# Tab 4: Generative Viz (LLM recommended)
# ------------------------------------------
with tab4:
    from views.tab_generative_viz import render_tab_generative_viz
    render_tab_generative_viz(conn, gemini_key, groq_key, BRAND_COLORS)

"""

tab4_idx = -1
tab_pyg_idx = -1

for i, line in enumerate(lines):
    if line.strip() == "# Tab 4: Generative Viz (LLM recommended)":
        tab4_idx = i - 1
    if line.strip() == "# Tab PyGWalker: Advanced Drag and Drop BI":
        tab_pyg_idx = i - 1

if tab4_idx != -1 and tab_pyg_idx != -1:
    del lines[tab4_idx:tab_pyg_idx]
    lines.insert(tab4_idx, new_content)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Refactoring Tab 4 successful!")
else:
    print(f"Failed to cleanly find tab indices. tab4: {tab4_idx}, tab_pyg: {tab_pyg_idx}")
