import pandas as pd
import numpy as np

def detect_data_types(df):
    """
    Detect and return data types for each column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary with column names as keys and data types as values.
    """
    return df.dtypes.to_dict()

def detect_missing_values(df):
    """
    Detect missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series with missing value counts per column.
    """
    return df.isnull().sum()

def detect_duplicates(df):
    """
    Detect duplicate rows in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        int: Number of duplicate rows.
    """
    return df.duplicated().sum()

def get_dataset_shape(df):
    """
    Get the shape of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: (rows, columns)
    """
    return df.shape