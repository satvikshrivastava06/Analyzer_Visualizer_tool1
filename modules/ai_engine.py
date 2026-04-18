import google.generativeai as genai
import streamlit as st
import pandas as pd
from typing import Tuple, List
import json

def get_ai_missions(df_sample) -> List[str]:
    """Analyze dataset schema and suggest 3 high-value "Analysis Missions"."""
    schema = df_sample.dtypes.to_string()
    prompt = f"""
    Analyze this dataset schema and suggest 3 high-value "Analysis Missions" for a data scientist.
    Schema:
    {schema}
    
    Return ONLY a JSON array of strings, where each string is a concise mission title.
    Example: ["Analyze Survival Price Correlation", "Predict Passenger Surivival", "Segment Users by Fare Paid"]
    """
    try:
        raw, _ = safe_ai_call(prompt)
        clean = raw.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception:
        return []

def safe_ai_call(prompt: str) -> Tuple[str, str]:
    """
    Universal AI call with multi-provider fallback strategy.
    1st: Tries best free Gemini models (fastest, most capable)
    2nd: If all Gemini quota exhausted, falls back to Groq free models
         (Llama 3.3 70B, Mixtral, Gemma 2 - very fast, zero cost)
    Returns (text, model_name).
    """
    gemini_key = None
    try:
        gemini_key = st.secrets["gemini"]["api_key"]
    except (FileNotFoundError, KeyError):
        pass

    groq_key = None
    try:
        groq_key = st.secrets["groq"]["api_key"]
    except (FileNotFoundError, KeyError):
        pass

    # --- Tier 1: Gemini (Google AI Studio Free Tier) ---
    if gemini_key:
        genai.configure(api_key=gemini_key)
        gemini_candidates = [
            'models/gemini-2.5-flash',       # Best: Latest & most capable
            'models/gemini-2.0-flash',        # Good: Fast & reliable
            'models/gemini-2.0-flash-lite',   # Lighter quota usage
            'models/gemini-flash-latest',     # Alias: always latest flash
            'models/gemma-3-27b-it',          # Open: Google's Gemma 3 27B
            'models/gemma-3-12b-it',          # Open: Google's Gemma 3 12B
            'models/gemini-pro-latest',       # Classic: Gemini Pro
        ]
        for model_name in gemini_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                return resp.text, model_name
            except Exception as e:
                err = str(e).lower()
                # Quota/not-found: skip to next model silently
                if any(x in err for x in ["429", "404", "not supported", "resource_exhausted"]):
                    continue
                else:
                    raise e  # Real error - surface it

    # --- Tier 2: Groq (Free tier - Llama/Mixtral/Gemma via Groq cloud) ---
    if groq_key:
        try:
            from groq import Groq # type: ignore
            groq_client = Groq(api_key=groq_key)
            groq_candidates = [
                "llama-3.3-70b-versatile",     # Best: Llama 3.3 70B (most capable)
                "mixtral-8x7b-32768",           # Great: Mixtral 8x7B (long context)
                "gemma2-9b-it",                 # Fast: Google Gemma 2 9B via Groq
                "llama-3.1-8b-instant",         # Fastest: Llama 3.1 8B
            ]
            for model_name in groq_candidates:
                try:
                    completion = groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                    )
                    return completion.choices[0].message.content, f"groq/{model_name}"
                except Exception as e:
                    err = str(e).lower()
                    if any(x in err for x in ["429", "rate_limit", "model_not_found", "deactivated"]):
                        continue
                    else:
                        raise e
        except ImportError:
            pass  # groq package not installed

    # --- All providers exhausted ---
    raise Exception("🛑 All free AI models are currently at their quota limit. Please wait a few minutes and try again.")


def validate_sql_query(query: str) -> Tuple[bool, str | None]:
    """
    SQL Security Sandbox: Scans for destructive keywords to prevent SQL Injection
    or accidental data deletion via AI or manual SQL.
    Returns (is_safe, blocked_keyword).
    """
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'ALTER', 'TRUNCATE', 'GRANT', 'INSERT', 'REVOKE']
    query_upper = query.upper()
    for keyword in dangerous_keywords:
        # Check for whole word match to avoid false positives with column names
        if f" {keyword} " in f" {query_upper} " or query_upper.startswith(keyword):
            return False, keyword
    return True, None


def log_version_action(action_name: str, code: str, details: str = ""):
    """
    Git-Lite: Logs every action taken on the data to ensure full auditability/provenance.
    """
    import datetime
    try:
        if 'version_history' not in st.session_state:
            st.session_state.version_history = []
        
        st.session_state.version_history.append({
            "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
            "action": action_name,
            "code": code,
            "details": details
        })
    except (RuntimeError, AttributeError):
        # Fallback for non-streamlit environments (e.g. pytest or CLI)
        pass


def nl_to_chart(user_query: str, df_sample: pd.DataFrame) -> dict:
    """Uses LLM to translate natural language into a chart configuration."""
    schema = df_sample.dtypes.to_string()
    prompt = f"""
    Translate the user's data request into a Plotly chart configuration.
    User Request: "{user_query}"
    Data Schema:
    {schema}
    
    Return ONLY JSON with the following structure:
    {{
       "chart_type": "bar" | "scatter" | "line" | "pie" | "histogram",
       "x": "column_name",
       "y": "column_name" | null,
       "title": "Clear Chart Title",
       "color": "column_name" | null
    }}
    """
    try:
        raw, _ = safe_ai_call(prompt)
        clean = raw.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception:
        return {}

def build_plotly_chart(df, config: dict):
    """Factory to build a Plotly Express figure from a JSON config."""
    import plotly.express as px
    c_type = config.get("chart_type", "").lower()
    x = config.get("x")
    y = config.get("y")
    title = config.get("title", "AI Generated Chart")
    color = config.get("color")
    
    kwargs = {"title": title}
    if color:
        kwargs["color"] = color

    if c_type == "bar":
        return px.bar(df, x=x, y=y, **kwargs)
    elif c_type == "scatter":
        return px.scatter(df, x=x, y=y, **kwargs)
    elif c_type == "line":
        return px.line(df, x=x, y=y, **kwargs)
    elif c_type == "pie":
        return px.pie(df, names=x, values=y, **kwargs)
    elif c_type == "histogram":
        return px.histogram(df, x=x, **kwargs)
    return None

def explain_anomalies(df_sample, anomaly_rows, column_name):
    """Asks AI to explain why specific rows are considered anomalies."""
    summary = anomaly_rows[column_name].describe().to_string()
    sample_values = anomaly_rows[column_name].head(5).tolist()
    
    prompt = f"""
    The following rows are statistical outliers in the column '{column_name}'.
    Statistical Summary of Outliers:
    {summary}
    Example Outlier Values: {sample_values}
    
    In 2-3 concise sentences, explain what likely caused these anomalies (e.g., data entry error, seasonal peak, 
    legitimate variance) and provide a specific recommendation (e.g., investigate raw source, trim values, 
    or keep for analysis).
    """
    try:
        explanation, _ = safe_ai_call(prompt)
        return explanation
    except Exception as e:
        return f"Could not generate AI explanation: {e}"
