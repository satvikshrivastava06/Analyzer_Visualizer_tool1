import pandas as pd
import numpy as np
from modules.cleaning import smart_impute, drop_duplicates_secure
import pytest

def test_smart_impute():
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": ["cat", "dog", np.nan, "cat"]
    })
    # Median of 1, 2, 4 is 2.0
    cleaned_df = smart_impute(df)
    
    assert cleaned_df["A"].isnull().sum() == 0
    assert cleaned_df["A"][2] == 2.0  # Median
    
    assert cleaned_df["B"].isnull().sum() == 0
    assert cleaned_df["B"][2] == "cat"  # Mode

def test_drop_duplicates():
    df = pd.DataFrame({
        "A": [1, 2, 2, 4],
    })
    cleaned_df, count = drop_duplicates_secure(df)
    
    assert len(cleaned_df) == 3
    assert count == 1
