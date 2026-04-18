import sys

file_path = r"c:\Users\Satvik Shrivastava\OneDrive\Desktop\Antigravity\app.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_content = """# ------------------------------------------
# Tab 3: AI Assistant (Natural Language)
# ------------------------------------------
with tab3:
    from views.tab_assistant import render_tab_assistant
    render_tab_assistant(conn, gemini_key, groq_key)

"""

tab3_idx = -1
tab4_idx = -1

for i, line in enumerate(lines):
    if line.strip() == "# Tab 3: AI Assistant (Natural Language)":
        tab3_idx = i - 1
    if line.strip() == "# Tab 4: Generative Viz (LLM recommended)":
        tab4_idx = i - 1

if tab3_idx != -1 and tab4_idx != -1:
    del lines[tab3_idx:tab4_idx]
    lines.insert(tab3_idx, new_content)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Refactoring Tab 3 successful!")
else:
    print(f"Failed to cleanly find tab indices. tab3: {tab3_idx}, tab4: {tab4_idx}")
