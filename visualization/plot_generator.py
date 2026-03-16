import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
sns.histplot(data=df, x='{column}', kde=True)
plt.title('Histogram of {column}')
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
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='{column}')
plt.title('Boxplot of {column}')
plt.show()
"""
    return code

def generate_bar_chart(df: pd.DataFrame, column: str) -> str:
    """
    Generate bar chart code for a categorical column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name.

    Returns:
        str: Python code for bar chart.
    """
    code = f"""
plt.figure(figsize=(10, 6))
df['{column}'].value_counts().plot(kind='bar')
plt.title('Bar Chart of {column}')
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
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
"""
    return code

def generate_pairplot(df: pd.DataFrame) -> str:
    """
    Generate pairplot code for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        str: Python code for pairplot.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        code = f"""
sns.pairplot(df[{numerical_cols}])
plt.show()
"""
    else:
        code = "# Not enough numerical columns for pairplot"
    return code