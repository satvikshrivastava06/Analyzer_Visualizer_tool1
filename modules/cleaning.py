import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from modules.ai_engine import log_version_action


def generate_sweetviz_report(df: pd.DataFrame):
    """Generates a sweetviz HTML report and saves it."""
    import sweetviz as sv
    report = sv.analyze(df)
    report.show_html("profile.html", open_browser=False)
    log_version_action("Automated EDA", "sv.analyze(df)", "Generated comprehensive sweetviz HTML report.")


def smart_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fills numeric columns with median, and object columns with mode."""
    cleaned_df = df.copy()
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        median_val = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_val)
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        mode_val = cleaned_df[col].mode()
        if not mode_val.empty:
            cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
        else:
            cleaned_df[col] = cleaned_df[col].fillna("Unknown")
    log_version_action("Heuristic Imputation", "df.fillna(median/mode)", "Filled numeric with median and categorical with mode.")
    return cleaned_df


def mask_pii(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Masks basic PII columns and returns the new DF and the list of masked columns."""
    cleaned_df = df.copy()
    pii_keywords = ['name', 'email', 'phone', 'ssn', 'address', 'ip', 'password', 'credit']
    masked_cols = []
    
    for col in cleaned_df.columns:
        if any(keyword in col.lower() for keyword in pii_keywords):
            cleaned_df[col] = "[REDACTED]"
            masked_cols.append(col)
            
    if masked_cols:
        log_version_action("PII Anonymization", f"df[{masked_cols}] = '[REDACTED]'", f"Masked {len(masked_cols)} columns.")
    return cleaned_df, masked_cols


def drop_duplicates_secure(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drops duplicates and logs the action."""
    initial_count = len(df)
    cleaned_df = df.drop_duplicates()
    final_count = len(cleaned_df)
    dropped = initial_count - final_count
    if dropped > 0:
        log_version_action("Drop Duplicates", "df.drop_duplicates()", f"Removed {dropped} rows.")
    return cleaned_df, dropped


def drop_missing_nas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drops rows with NAs and logs the action."""
    initial_count = len(df)
    cleaned_df = df.dropna()
    final_count = len(cleaned_df)
    dropped = initial_count - final_count
    if dropped > 0:
        log_version_action("Drop Missing NAs", "df.dropna()", f"Removed {dropped} rows.")
    return cleaned_df, dropped
