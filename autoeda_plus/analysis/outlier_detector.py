import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Detect outliers using IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.

    Returns:
        pd.Series: Boolean series indicating outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.
        threshold (float): Z-score threshold.

    Returns:
        pd.Series: Boolean series indicating outliers.
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = pd.Series(False, index=df.index)
    outliers.loc[df[column].notna()] = z_scores > threshold
    return outliers

def get_outlier_summary(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, int]:
    """
    Get outlier counts for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        Dict[str, int]: Outlier counts per column.
    """
    numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numerical']
    outlier_summary = {}

    for col in numerical_cols:
        if df[col].dtype in ['int64', 'float64']:
            iqr_outliers = detect_outliers_iqr(df, col).sum()
            zscore_outliers = detect_outliers_zscore(df, col).sum()
            outlier_summary[col] = {
                'iqr_outliers': int(iqr_outliers),
                'zscore_outliers': int(zscore_outliers)
            }

    return outlier_summary