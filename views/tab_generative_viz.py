import streamlit as st
import pandas as pd
import plotly.express as px
from typing import cast
from modules.ai_engine import safe_ai_call, nl_to_chart, build_plotly_chart
from ui_theme import page_hero, section_header, apply_plotly_theme, anomaly_row, model_badge, TEXT_3, CHART_COLORS

def render_tab_generative_viz(conn, gemini_key, groq_key, BRAND_COLORS):
    st.markdown(page_hero("◈", "Generative Viz", "Instantly build beautiful, interactive charts using AI or manual controls."), unsafe_allow_html=True)
    if st.session_state.current_table is None:
        from ui_theme import empty_state
        st.markdown(empty_state("📥", "No Data Loaded", "Load data in the Ingest tab first."), unsafe_allow_html=True)
        return
    
    df = cast(pd.DataFrame, st.session_state.df_preview)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("📊", "Interactive Visualizations", "Manual configurations (Colorblind Safe)"), unsafe_allow_html=True)
    
    use_colorblind = st.toggle("Use Colorblind-Safe Palette", value=False)
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"] if use_colorblind else CHART_COLORS
    
    max_points = st.slider("Max Data Points to Plot (Prevents Browser Freeze)", min_value=100, max_value=50000, value=10000, step=100)
    
    plot_df = df.sample(n=min(len(df), max_points), random_state=42) if len(df) > max_points else df
    if len(df) > max_points:
        st.caption(f"⚠️ *Showing a random sample of {max_points} points (out of {len(df)}) to prevent data overload.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox("Chart Type", ["Bar", "Scatter", "Histogram", "Box", "Violin", "Treemap", "Sunburst"])
    with col2:
        x_ax = st.selectbox("X-Axis / Path 1", df.columns.tolist())
    with col3:
        y_ax = st.selectbox("Y-Axis / Path 2", df.columns.tolist() + ["None"])

    if st.button("Generate Chart 📊", use_container_width=True):
        try:
            fig = None
            if chart_type == "Bar" and y_ax != "None":
                fig = px.bar(plot_df, x=x_ax, y=y_ax, color_discrete_sequence=colors)
            elif chart_type == "Scatter" and y_ax != "None":
                fig = px.scatter(plot_df, x=x_ax, y=y_ax, color_discrete_sequence=colors)
            elif chart_type == "Histogram":
                fig = px.histogram(plot_df, x=x_ax, color_discrete_sequence=colors)
            elif chart_type == "Box" and y_ax != "None":
                fig = px.box(plot_df, x=x_ax, y=y_ax, color_discrete_sequence=colors)
            elif chart_type == "Violin" and y_ax != "None":
                fig = px.violin(plot_df, x=x_ax, y=y_ax, box=True, color_discrete_sequence=colors)
            elif chart_type == "Treemap" and y_ax != "None":
                fig = px.treemap(plot_df, path=[x_ax, y_ax], color_discrete_sequence=colors)
            elif chart_type == "Sunburst" and y_ax != "None":
                fig = px.sunburst(plot_df, path=[x_ax, y_ax], color_discrete_sequence=colors)
            else:
                st.warning(f"Please select a valid Y-Axis / Path 2 for a {chart_type} chart.")

            if fig:
                apply_plotly_theme(fig)
                st.plotly_chart(fig, width='stretch')

            if y_ax != "None":
                with st.expander(f"📊 Statistical Context for {x_ax} & {y_ax} (IJCDS Best Practice)", expanded=False):
                    stats_df = df[[x_ax, y_ax]].describe(include='all')
                    st.dataframe(stats_df)
                    st.caption("Review these exact numbers to confirm the visual representation is accurate and not logically misleading.")
            else:
                with st.expander(f"📊 Statistical Context for {x_ax} (IJCDS Best Practice)", expanded=False):
                    stats_df = df[[x_ax]].describe(include='all')
                    st.dataframe(stats_df)
        except Exception as e:
            st.error(f"Could not generate {chart_type} chart with those columns. ({str(e)})")
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("✍️", "Direct NL Visualization", "Ask AI to build a chart"), unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:13px;color:{TEXT_3};margin-bottom:12px">Type what you want to see, and AI will build the plot for you directly.</div>', unsafe_allow_html=True)
    nl_query = st.text_input("Describe your chart (e.g., 'Revenue trend by month')", key="nl_chart_input", label_visibility="collapsed")
    
    if nl_query:
        with st.spinner("AI is determining optimal chart parameters..."):
            chart_config = nl_to_chart(nl_query, df)
            if chart_config:
                fig = build_plotly_chart(df, chart_config)
                if fig:
                    apply_plotly_theme(fig)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.markdown(anomaly_row("AI suggested a chart, but Plotly could not render it with the current data.", "danger"), unsafe_allow_html=True)
            else:
                st.markdown(anomaly_row("AI could not translate that request into a valid chart configuration.", "danger"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🤖", "AI Recommended Charts", "Proactive data discovery"), unsafe_allow_html=True)
    
    if not gemini_key and not groq_key:
        st.markdown(anomaly_row("Add a Gemini or Groq API key to get smart chart recommendations.", "warning"), unsafe_allow_html=True)
    else:
        if st.button("Generate Recommendations 🤖", use_container_width=True):
            with st.spinner("AI is analyzing schema for optimal visualizations..."):
                schema_str = df.dtypes.to_string()
                prompt = f"""
                Given this pandas dataframe schema:
                {schema_str}
                
                Suggest 3 insightful Plotly Express charts to visualize this data. 
                Format your response ONLY as a JSON array of objects. 
                Each object MUST have these exact keys: "title", "chart_type", "x", "y", and "rationale".
                
                Possible values for "chart_type": "scatter", "bar", "histogram", "box", "violin", "treemap", "sunburst".
                Set "y" to null for histograms.
                
                CRITICAL (2026 Visualization Standards):
                1. **NVBench Methodology (Text-to-Vis):** Ensure the strict mapping of the natural language rationale directly aligns with the underlying data types.
                2. **ChartQA Logic:** Act as a data agent. Suggest the visual representation that directly answers the most profound analytical question.
                3. Avoid oversimplification: Suggest at least one advanced chart type.
                """
                try:
                    import json
                    raw_response, used_model = safe_ai_call(prompt)
                    clean_json = raw_response.strip().replace("```json", "").replace("```", "")
                    st.session_state.ai_recommendations = json.loads(clean_json)
                    st.session_state.last_rec_model = used_model
                except Exception as e:
                    st.error(f"Could not parse AI recommendations: {e}")

        if st.session_state.get('ai_recommendations'):
            from ui_theme import action_card
            st.markdown("⚠️ **AI Provenance Disclaimer:** These suggestions are generated by AI. Always validate the underlying data.")
            
            for i, rec in enumerate(st.session_state.ai_recommendations):
                with st.expander(f"Recommendation {i+1}: {rec['title']}", expanded=True):
                    st.write(f"**Rationale:** {rec['rationale']}")
                    st.caption(f"Config: {rec['chart_type']} | X: {rec['x']} | Y: {rec['y']}")
                    
                    rec_key = f"rec_chart_{i}"
                    if st.button(f"Generate '{rec['title']}'", key=f"btn_{i}", use_container_width=True):
                        st.session_state.generated_rec_charts[rec_key] = True
                    
                    if st.session_state.generated_rec_charts.get(rec_key):
                        try:
                            r_type = rec['chart_type'].lower()
                            r_x = rec['x']
                            r_y = rec['y']
                            r_title = rec['title']
                            
                            fig = None
                            if r_type == "scatter":
                                fig = px.scatter(df, x=r_x, y=r_y, title=r_title, color_discrete_sequence=CHART_COLORS)
                            elif r_type == "bar":
                                fig = px.bar(df, x=r_x, y=r_y, title=r_title, color_discrete_sequence=CHART_COLORS)
                            elif r_type == "histogram":
                                fig = px.histogram(df, x=r_x, title=r_title, color_discrete_sequence=CHART_COLORS)
                            elif r_type == "box":
                                fig = px.box(df, x=r_x, y=r_y, title=r_title, color_discrete_sequence=CHART_COLORS)
                            elif r_type == "violin":
                                fig = px.violin(df, x=r_x, y=r_y, box=True, title=r_title, color_discrete_sequence=CHART_COLORS)
                            elif r_type == "treemap":
                                fig = px.treemap(df, path=[r_x, r_y] if r_y else [r_x], title=r_title, color_discrete_sequence=CHART_COLORS)
                            elif r_type == "sunburst":
                                fig = px.sunburst(df, path=[r_x, r_y] if r_y else [r_x], title=r_title, color_discrete_sequence=CHART_COLORS)
                            
                            if fig:
                                apply_plotly_theme(fig)
                                st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.error(f"Chart generation failed: {e}")
            
            st.markdown(model_badge(st.session_state.get('last_rec_model', 'AI Model')), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("🔎", "Semantic Search (RAG Lite)", "Search data by meaning"), unsafe_allow_html=True)
    search_query = st.text_input("Describe what you are looking for", key="semantic_search", label_visibility="collapsed")
    
    if search_query:
        with st.spinner("Searching semantically..."):
            manifest_str = st.session_state.get("semantic_manifest", "{}")
            search_prompt = f"""
            You are a Semantic Search Engine for data.
            
            ### DATABASE SCHEMA ###
            Available columns and sample values: {df.head(5).to_dict()}
            
            ### BUSINESS CONTEXT (WREN AI MANIFEST) ###
            {manifest_str}
            
            ### USER REQUEST ###
            The user is looking for: "{search_query}"
            
            Write a DuckDB SQL SELECT query that finds the most relevant rows based on the semantic meaning of the request. 
            Use LIKE or filter logic that best matches the intent of "{search_query}". Focus on the most descriptive text columns. 
            ONLY return the SQL query, without markdown backticks.
            """
            try:
                suggested_sql, _ = safe_ai_call(search_prompt)
                suggested_sql = suggested_sql.replace("```sql", "").replace("```", "").strip()
                st.info(f"AI-driven Semantic Filter applied: `{suggested_sql}`")
                search_results = conn.execute(suggested_sql).df()
                st.dataframe(search_results, width='stretch')
            except Exception as e:
                st.markdown(anomaly_row(f"Semantic search failed: {e}", "danger"), unsafe_allow_html=True)
