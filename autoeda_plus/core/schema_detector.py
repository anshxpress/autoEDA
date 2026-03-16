import pandas as pd
import numpy as np
from typing import Dict, List, Any

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detect column types based on heuristics.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, str]: Column name to type mapping.
    """
    column_types = {}

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_total = len(df)

        # Identifier detection: high cardinality, likely unique
        if n_unique == n_total and n_total > 10:
            column_types[col] = 'identifier'
        # Datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        # Numerical
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's actually categorical (few unique values)
            if n_unique <= 10 and n_total > 50:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'numerical'
        # Object type
        else:
            # Check for text vs categorical
            if df[col].astype(str).str.len().mean() > 50:  # Long strings
                column_types[col] = 'text'
            elif n_unique / n_total < 0.05:  # Low cardinality
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'

    return column_types

def detect_potential_target(df: pd.DataFrame, column_types: Dict[str, str]) -> str:
    """
    Detect potential target column for ML.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        str: Potential target column name.
    """
    # Look for binary categorical or numerical with few unique values
    candidates = []
    for col, ctype in column_types.items():
        if ctype == 'categorical' and df[col].nunique() == 2:
            candidates.append(col)
        elif ctype == 'numerical' and df[col].nunique() <= 5:
            candidates.append(col)

    # Prefer the last column if it's a candidate
    if candidates and candidates[-1] in df.columns:
        return candidates[-1]

    # Fallback to last column
    return df.columns[-1]

def get_column_summary(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
    """
    Get summary of column types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        Dict[str, Any]: Summary statistics.
    """
    summary = {}
    for ctype in ['numerical', 'categorical', 'datetime', 'text', 'identifier']:
        cols = [col for col, t in column_types.items() if t == ctype]
        summary[ctype] = {
            'count': len(cols),
            'columns': cols
        }

    return summary