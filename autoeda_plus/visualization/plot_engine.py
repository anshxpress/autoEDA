import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List

def generate_histogram(df: pd.DataFrame, column: str) -> str:
    """
    Generate histogram code for a numerical column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.

    Returns:
        str: Python code for histogram.
    """
    code = f"""
plt.figure(figsize=(10, 6))
sns.histplot(data={df.name or 'df'}, x='{column}', kde=True)
plt.title('Distribution of {column}')
plt.xlabel('{column}')
plt.ylabel('Frequency')
plt.show()
"""
    return code

def generate_boxplot(df: pd.DataFrame, column: str) -> str:
    """
    Generate boxplot code for a numerical column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.

    Returns:
        str: Python code for boxplot.
    """
    code = f"""
plt.figure(figsize=(8, 6))
sns.boxplot(data={df.name or 'df'}, y='{column}')
plt.title('Boxplot of {column}')
plt.ylabel('{column}')
plt.show()
"""
    return code

def generate_countplot(df: pd.DataFrame, column: str) -> str:
    """
    Generate countplot for a categorical column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.

    Returns:
        str: Python code for countplot.
    """
    code = f"""
plt.figure(figsize=(10, 6))
sns.countplot(data={df.name or 'df'}, x='{column}', order={df.name or 'df'}['{column}'].value_counts().index)
plt.title('Count of {column}')
plt.xlabel('{column}')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
"""
    return code

def generate_correlation_heatmap(corr_matrix: pd.DataFrame) -> str:
    """
    Generate correlation heatmap code.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.

    Returns:
        str: Python code for heatmap.
    """
    code = f"""
plt.figure(figsize=(12, 8))
sns.heatmap({corr_matrix.name or 'corr_matrix'}, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
"""
    return code

def generate_scatterplot(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """
    Generate scatterplot for two numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): X-axis column.
        y_col (str): Y-axis column.

    Returns:
        str: Python code for scatterplot.
    """
    code = f"""
plt.figure(figsize=(10, 6))
sns.scatterplot(data={df.name or 'df'}, x='{x_col}', y='{y_col}')
plt.title('{y_col} vs {x_col}')
plt.xlabel('{x_col}')
plt.ylabel('{y_col}')
plt.show()
"""
    return code

def generate_missing_values_plot(df: pd.DataFrame) -> str:
    """
    Generate missing values heatmap.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        str: Python code for missing values plot.
    """
    code = f"""
plt.figure(figsize=(12, 8))
sns.heatmap({df.name or 'df'}.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
"""
    return code