import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def detect_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect and return data types for each column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary with column names as keys and data types as values.
    """
    return df.dtypes.to_dict()

def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Detect missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series with missing value counts per column.
    """
    return df.isnull().sum()

def detect_duplicates(df: pd.DataFrame) -> int:
    """
    Detect duplicate rows in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        int: Number of duplicate rows.
    """
    return df.duplicated().sum()

def get_dataset_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Get the shape of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: (rows, columns)
    """
    return df.shape