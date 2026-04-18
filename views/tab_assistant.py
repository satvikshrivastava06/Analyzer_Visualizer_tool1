import streamlit as st
import pandas as pd
from typing import cast
from modules.ai_engine import safe_ai_call, validate_sql_query, log_version_action
from ui_theme import page_hero, section_header, model_badge, action_card, anomaly_row, TEXT_3, SURFACE_3, BORDER

def render_tab_assistant(conn, gemini_key, groq_key):
    st.markdown(page_hero("◆", "AI Data Assistant", "Chat with your data, run complex agentic transformations, and analyze SQL schemas."), unsafe_allow_html=True)
    
    if not gemini_key and not groq_key:
        from ui_theme import empty_state
        st.markdown(empty_state("🔐", "API Key Missing", "Please add a Gemini or Groq API Key to your configuration to use this feature."), unsafe_allow_html=True)
        return
    elif st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load data in the Ingest tab first."), unsafe_allow_html=True)
        return

    st.markdown("<br>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    import re
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            display_text = msg["content"]
            
            think_match = re.search(r"<think>(.*?)</think>", display_text, re.DOTALL)
            if think_match:
                display_text = display_text.replace(think_match.group(0), "").strip()
                
            st.markdown(display_text)
            
            if "model" in msg:
                st.markdown(model_badge(msg["model"]), unsafe_allow_html=True)
                
            if msg["role"] == "assistant":
                match = re.search(r"\[AGENTIC_SELECT\](.*?)\[/AGENTIC_SELECT\]", msg["content"], re.DOTALL)
                if match:
                    raw_sql = match.group(1).strip()
                    # Clean markdown code blocks if AI included them
                    agentic_sql = raw_sql.replace("```sql", "").replace("```", "").replace("`", "").strip()
                    
                    if st.button("🚀 Perform Live Surgery", use_container_width=True, key=f"surgery_{i}"):
                        with st.spinner("Executing live surgery..."):
                            is_safe, keyword = validate_sql_query(agentic_sql)
                            if not is_safe:
                                st.markdown(anomaly_row(f"Security Block: Destructive keyword `{keyword}` detected.", "danger"), unsafe_allow_html=True)
                            else:
                                try:
                                    conn.execute(f"CREATE OR REPLACE VIEW current_data AS {agentic_sql}")
                                    st.session_state.df_preview = conn.execute("SELECT * FROM current_data LIMIT 1000").df()
                                    log_version_action("Agentic Data Surgery", agentic_sql, f"AI-driven refinement using {msg.get('model', 'History')}")
                                    st.session_state.flash_msg = action_card("Transformation", "Data successfully refined!", "Check the Ingest or Clean tabs to see results.")
                                    st.rerun()
                                except Exception as e:
                                    st.markdown(anomaly_row(f"Surgery failed: {e}", "danger"), unsafe_allow_html=True)

    
    with st.expander("🧠 Semantic Manifest (Context Layer)", expanded=False):
        st.markdown(
            f'<div style="font-size:13px;color:{TEXT_3};margin-bottom:12px">'
            f'Define business logic, calculated metrics, and context. The Text-to-SQL engine will strictly follow this manifest when generating queries to prevent hallucinations.'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if "semantic_manifest" not in st.session_state:
            st.session_state.semantic_manifest = """{
    "metrics": [
        {
            "name": "Revenue",
            "description": "Total revenue generated. If columns Price and Quantity exist, multiply them. Otherwise, sum the Revenue column."
        },
        {
            "name": "Net Profit",
            "description": "Revenue minus Cost."
        }
    ],
    "context": "We are analyzing e-commerce or retail data. Keep queries compatible with DuckDB."
}"""
        manifest_input = st.text_area("Edit JSON Manifest:", value=st.session_state.semantic_manifest, height=180, label_visibility="collapsed")
        if st.button("💾 Save Manifest", use_container_width=True):
            st.session_state.semantic_manifest = manifest_input
            st.markdown(action_card("Configuration", "Manifest Saved", "The AI will now use this context."), unsafe_allow_html=True)

    if st.session_state.get("ai_missions"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("🚀", "Suggested Missions", "Launch an AI-driven deep dive"), unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.ai_missions))
        for i, mission in enumerate(st.session_state.ai_missions):
            if cols[i].button(f"🚀 {mission}", key=f"mission_{i}", use_container_width=True):
                st.session_state.ai_chat_prompt = f"Conduct a detailed analysis for this mission: {mission}"
    
    prompt = st.chat_input("Ask a question or request a transformation (e.g., 'Filter for high Revenue rows')")
    
    if st.session_state.get("ai_chat_prompt") and not prompt:
        prompt = st.session_state.ai_chat_prompt
        st.session_state.ai_chat_prompt = None 
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    schema_df = conn.execute(f"DESCRIBE {st.session_state.current_table}").df()
                    schema_str = schema_df[['column_name', 'column_type']].to_string()
                except Exception:
                    df_prev = cast(pd.DataFrame, st.session_state.df_preview)
                    schema_str = df_prev.dtypes.to_string() if df_prev is not None else "Schema unavailable"
                
                manifest_content = st.session_state.get("semantic_manifest", "{}")
                
                ai_prompt = f"""
                You are a world-class Text-to-SQL engine powered by WrenAI Agentic architecture.
                
                ### DATABASE SCHEMA ###
                Table name: '{st.session_state.current_table}'
                Schema (Column Name, Type):
                {schema_str}
                
                ### BUSINESS METRICS / SEMANTIC MANIFEST ###
                {manifest_content}
                
                ### USER INSTRUCTIONS ###
                The user asked: "{prompt}"
                
                1. Provide a friendly, helpful, and concise answer explaining your approach.
                2. If the user wants to filter, transform, or refine the active dataset, generate a SELECT query representing the new desired state.
                3. For standard insights, use ```sql blocks.
                4. For 'Agentic Execution' (Filtering/Transformation), wrap the SQL query EXACTLY in [AGENTIC_SELECT]...[/AGENTIC_SELECT] tags so the engine can execute it for the user.
                5. Only suggest SELECT queries. Never suggest DROP, DELETE, or UPDATE. Use DuckDB dialaect.
                
                ### REASONING PLAN ###
                Before writing the final response or SQL, create a <think> block analyzing which columns from the schema map to the business metrics described in the manifest.
                """
                try:
                    text, used_model = safe_ai_call(ai_prompt)
                    st.session_state.messages.append({"role": "assistant", "content": text, "model": used_model})
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

