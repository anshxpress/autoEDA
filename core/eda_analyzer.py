import pandas as pd
import numpy as np
from scipy import stats

def generate_descriptive_stats(df):
    """
    Generate descriptive statistics for the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics.
    """
    return df.describe(include='all')

def analyze_numerical_features(df):
    """
    Analyze numerical features: mean, median, std, skewness.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary with stats for numerical columns.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict = {}
    for col in numerical_cols:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': df[col].skew()
        }
    return stats_dict

def analyze_categorical_features(df):
    """
    Analyze categorical features: unique counts, frequencies.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary with stats for categorical columns.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    stats_dict = {}
    for col in categorical_cols:
        stats_dict[col] = {
            'unique_count': df[col].nunique(),
            'value_counts': df[col].value_counts().to_dict()
        }
    return stats_dict

def detect_datetime_features(df):
    """
    Detect and convert datetime columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of datetime column names.
    """
    datetime_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            datetime_cols.append(col)
        except:
            pass
    return datetime_cols

def calculate_correlations(df):
    """
    Calculate correlation matrix for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    return numerical_df.corr()

def detect_outliers_iqr(df, column):
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

def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.
        threshold (float): Z-score threshold.

    Returns:
        pd.Series: Boolean series indicating outliers.
    """
    z_scores = np.abs(stats.zscore(df[column]))
    return z_scores > threshold