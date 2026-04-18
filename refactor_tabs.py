import sys

file_path = r"c:\Users\Satvik Shrivastava\OneDrive\Desktop\Antigravity\app.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_content = """# ------------------------------------------
# Tab 1: Data Ingestion (Scalable loading)
# ------------------------------------------
with tab1:
    from views.tab_ingestion import render_tab_ingestion
    render_tab_ingestion(conn, db_name, get_ai_missions)

# ------------------------------------------
# Tab 2: Smart Clean (Automated Profiling)
# ------------------------------------------
with tab2:
    from views.tab_cleaning import render_tab_cleaning
    render_tab_cleaning(conn)

"""

# We know we want to replace lines 194 to 608 (1-indexed) -> 193 to 608 (0-indexed slice)
# Let's verify by finding the exact indices of "with tab1:" and "with tab3:"
tab1_idx = -1
tab3_idx = -1

for i, line in enumerate(lines):
    if line.strip() == "# Tab 1: Data Ingestion (Scalable loading)":
        tab1_idx = i - 1  # Get the line before it as well if it's empty
    if line.strip() == "# Tab 3: AI Assistant (Natural Language)":
        tab3_idx = i - 1

if tab1_idx != -1 and tab3_idx != -1:
    del lines[tab1_idx:tab3_idx]
    lines.insert(tab1_idx, new_content)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Refactoring Tab 1 & 2 successful!")
else:
    print(f"Failed to cleanly find tab indices. tab1: {tab1_idx}, tab3: {tab3_idx}")
