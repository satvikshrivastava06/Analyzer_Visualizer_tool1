import duckdb
import streamlit as st
import os

def get_db_connection():
    """
    Returns a DuckDB connection. 
    Attempts MotherDuck first; falls back to local in-memory DB.
    """
    try:
        md = st.secrets.get("motherduck", {})
        if md.get("token"):
            os.environ["MOTHERDUCK_TOKEN"] = md["token"]
            db_name = md.get("db_name", "my_db")
            conn = duckdb.connect(f'md:{db_name}')
            # Test connection
            conn.execute("SELECT 1")
            return conn, "motherduck"
    except Exception as e:
        # Fallback if connection fails, token is wrong, or offline
        pass
    
    # Generic local in memory DB
    return duckdb.connect(':memory:'), "local"
