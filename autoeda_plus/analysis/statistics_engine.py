import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

def compute_numerical_statistics(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical metrics for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        Dict[str, Dict[str, float]]: Statistics per column.
    """
    numerical_cols = [col for col, ctype in column_types.items() if ctype == 'numerical']
    stats_dict = {}

    for col in numerical_cols:
        data = df[col].dropna()
        if len(data) > 0:
            stats_dict[col] = {
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
                'iqr': data.quantile(0.75) - data.quantile(0.25)
            }

    return stats_dict

def compute_categorical_statistics(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics for categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        Dict[str, Dict[str, Any]]: Statistics per column.
    """
    categorical_cols = [col for col, ctype in column_types.items() if ctype == 'categorical']
    stats_dict = {}

    for col in categorical_cols:
        value_counts = df[col].value_counts()
        stats_dict[col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'value_counts': value_counts.head(10).to_dict()  # Top 10
        }

    return stats_dict