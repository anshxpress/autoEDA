import pandas as pd
from typing import List, Dict, Any
from ..analysis.statistics_engine import compute_numerical_statistics
from ..analysis.correlation_engine import detect_strong_correlations
from ..analysis.outlier_detector import get_outlier_summary

def generate_insights(df: pd.DataFrame, column_types: Dict[str, str],
                     corr_matrix: pd.DataFrame, outlier_summary: Dict[str, Any]) -> List[str]:
    """
    Generate human-readable insights from the analysis.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.
        corr_matrix (pd.DataFrame): Correlation matrix.
        outlier_summary (Dict[str, Any]): Outlier summary.

    Returns:
        List[str]: List of insights.
    """
    insights = []

    # Dataset overview
    insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

    # Missing values
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        missing_pct = total_missing / (df.shape[0] * df.shape[1]) * 100
        insights.append(f"Dataset has {missing_pct:.1f}% missing values overall")

    # Column types
    num_cols = sum(1 for t in column_types.values() if t == 'numerical')
    cat_cols = sum(1 for t in column_types.values() if t == 'categorical')
    insights.append(f"Found {num_cols} numerical and {cat_cols} categorical columns")

    # Statistical insights
    num_stats = compute_numerical_statistics(df, column_types)
    for col, stats in num_stats.items():
        if abs(stats['skewness']) > 1:
            direction = "right" if stats['skewness'] > 0 else "left"
            insights.append(f"Column '{col}' is highly {direction}-skewed (skewness: {stats['skewness']:.2f})")

    # Correlation insights
    strong_corrs = detect_strong_correlations(corr_matrix, 0.7)
    for col1, col2, corr in strong_corrs[:3]:  # Top 3
        strength = "strong" if abs(corr) > 0.8 else "moderate"
        direction = "positive" if corr > 0 else "negative"
        insights.append(f"{strength} {direction} correlation ({corr:.2f}) between '{col1}' and '{col2}'")

    # Outlier insights
    for col, counts in outlier_summary.items():
        total_outliers = max(counts['iqr_outliers'], counts['zscore_outliers'])
        if total_outliers > 0:
            pct = total_outliers / len(df) * 100
            insights.append(f"Column '{col}' has {total_outliers} outliers ({pct:.1f}%)")

    # Categorical insights
    from ..analysis.statistics_engine import compute_categorical_statistics
    cat_stats = compute_categorical_statistics(df, column_types)
    for col, stats in cat_stats.items():
        if stats['unique_count'] == 2:
            insights.append(f"Column '{col}' is binary with values: {list(stats['value_counts'].keys())}")

    return insights

def generate_feature_engineering_suggestions(column_types: Dict[str, str],
                                           num_stats: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Generate feature engineering suggestions.

    Args:
        column_types (Dict[str, str]): Column types.
        num_stats (Dict[str, Dict[str, float]]): Numerical statistics.

    Returns:
        List[str]: Feature engineering suggestions.
    """
    suggestions = []

    # Skewed features
    for col, stats in num_stats.items():
        if abs(stats['skewness']) > 1:
            if stats['skewness'] > 0:
                suggestions.append(f"Consider log transformation for right-skewed '{col}'")
            else:
                suggestions.append(f"Consider square transformation for left-skewed '{col}'")

    # Categorical encoding
    cat_cols = [col for col, t in column_types.items() if t == 'categorical']
    if cat_cols:
        suggestions.append(f"Consider one-hot encoding for categorical columns: {cat_cols[:3]}")

    # Datetime features
    datetime_cols = [col for col, t in column_types.items() if t == 'datetime']
    if datetime_cols:
        suggestions.append(f"Extract features from datetime columns: {datetime_cols}")

    # Scaling
    num_cols = [col for col, t in column_types.items() if t == 'numerical']
    if len(num_cols) > 1:
        suggestions.append("Consider standardizing numerical features for ML models")

    return suggestions