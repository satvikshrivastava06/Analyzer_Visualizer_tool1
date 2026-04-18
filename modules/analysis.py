import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def detect_anomalies(df: pd.DataFrame, threshold=3) -> list[dict]:
    """Scans numerical columns for outliers and returns structured data for diagnosis."""
    results = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            z_scores = ((df[col] - mean_val) / std_val).abs()
            outliers_idx = z_scores[z_scores > threshold].index
            if len(outliers_idx) > 0:
                results.append({
                    "column": col,
                    "count": len(outliers_idx),
                    "outliers": df.loc[outliers_idx]
                })
    return results


def train_forecasting_model(hist_df: pd.DataFrame, col_x: str, col_y: str, poly_degree: int, steps_ahead: int) -> pd.DataFrame:
    """Trains a polynomial regression model and returns predicted dataframe."""
    X = hist_df[[col_x]].values
    y = hist_df[col_y].values
    
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict future
    last_x = hist_df[col_x].max()
    min_x = hist_df[col_x].min()
    
    # Determine step size: 1 if 'year' is in col_x, otherwise average delta
    if 'year' in col_x.lower():
        step = 1
    else:
        step = (last_x - min_x) / len(hist_df) if len(hist_df) > 1 else 1

    future_Xs = np.array([last_x + i * step for i in range(1, steps_ahead + 1)]).reshape(-1, 1)
    future_X_poly = poly.transform(future_Xs)
    predictions = model.predict(future_X_poly)
    
    pred_df = pd.DataFrame({
        col_x: future_Xs.flatten(),
        "AI_Prediction": predictions
    })
    
    return pred_df

def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, name1: str = "Active", name2: str = "Comparison") -> dict:
    """
    Performs detailed delta analysis between two datasets.
    Checks for column overlap, row count skew, and statistical shifts in numeric columns.
    """
    # 1. Column Overlap
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    shared_cols = sorted(list(cols1 & cols2))
    only_in_1 = sorted(list(cols1 - cols2))
    only_in_2 = sorted(list(cols2 - cols1))

    # 2. Row Count Delta
    len1 = len(df1)
    len2 = len(df2)
    row_delta = len2 - len1
    row_delta_pct = (row_delta / len1 * 100) if len1 != 0 else 0

    # 3. Numeric Comparison
    numeric_shifts = {}
    for col in shared_cols:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            m1 = df1[col].mean()
            m2 = df2[col].mean()
            delta = m2 - m1
            delta_pct = (delta / abs(m1) * 100) if m1 != 0 else 0
            numeric_shifts[col] = {
                f"{name1}_mean": m1,
                f"{name2}_mean": m2,
                "delta": delta,
                "delta_pct": delta_pct
            }

    return {
        "shared_columns": shared_cols,
        "only_in_active": only_in_1,
        "only_in_comparison": only_in_2,
        "active_rows": len1,
        "comparison_rows": len2,
        "row_delta": row_delta,
        "row_delta_pct": row_delta_pct,
        "numeric_shifts": numeric_shifts
    }
