import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def compute_correlation_matrix(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
    """
    Compute correlation matrix for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numerical']
    if len(numerical_cols) < 2:
        return pd.DataFrame()

    corr_matrix = df[numerical_cols].corr()
    return corr_matrix

def detect_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """
    Detect strong correlations in the matrix.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        threshold (float): Correlation threshold.

    Returns:
        List[Tuple[str, str, float]]: List of (col1, col2, corr_value) for strong correlations.
    """
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlations.append((col1, col2, corr_value))

    return sorted(correlations, key=lambda x: abs(x[2]), reverse=True)