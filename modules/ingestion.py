import pandas as pd
import streamlit as st
import io

def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all column names are unique by appending numeric suffixes."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(len(cols[cols == dup]))]
    df.columns = cols
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_from_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.DataFrame()
    return _deduplicate_columns(df)


@st.cache_data(ttl=600, show_spinner=False)
def load_from_url(url: str) -> pd.DataFrame:
    return _deduplicate_columns(pd.read_csv(url))


@st.cache_data(ttl=600, show_spinner=False)
def load_google_sheet(url: str) -> pd.DataFrame:
    """Loads a public Google Sheet by transforming the edit URL to a CSV export URL."""
    if "/edit" in url:
        csv_url = url.replace('/edit', '/export?format=csv')
        if "#gid=" in url:
            gid = url.split("#gid=")[1]
            csv_url += f"&gid={gid}"
    else:
        csv_url = url
    return _deduplicate_columns(pd.read_csv(csv_url))
