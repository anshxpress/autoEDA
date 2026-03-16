import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .schema_detector import detect_column_types

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive dataset profile.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, Any]: Dataset profile.
    """
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'column_types': detect_column_types(df)
    }

    # Missing values
    profile['missing_values'] = df.isnull().sum().to_dict()
    profile['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()

    # Duplicates
    profile['duplicate_rows'] = df.duplicated().sum()
    profile['duplicate_percentage'] = profile['duplicate_rows'] / len(df) * 100

    return profile

def detect_data_quality_issues(df: pd.DataFrame, column_types: Dict[str, str]) -> List[str]:
    """
    Detect common data quality issues.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (Dict[str, str]): Column types.

    Returns:
        List[str]: List of issues found.
    """
    issues = []

    # Missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    for col, pct in missing_pct.items():
        if pct > 50:
            issues.append(f"⚠ Column '{col}' has {pct:.1f}% missing values")
        elif pct > 0:
            issues.append(f"ℹ Column '{col}' has {pct:.1f}% missing values")

    # Duplicates
    dup_pct = df.duplicated().sum() / len(df) * 100
    if dup_pct > 5:
        issues.append(f"⚠ Dataset has {dup_pct:.1f}% duplicate rows")

    # Numerical issues
    for col, ctype in column_types.items():
        if ctype == 'numerical':
            if (df[col] < 0).any() and col.lower() in ['age', 'height', 'weight']:
                issues.append(f"⚠ Column '{col}' contains negative values")
            if df[col].nunique() == 1:
                issues.append(f"⚠ Column '{col}' has constant values")

        elif ctype == 'categorical':
            if df[col].nunique() > 50:
                issues.append(f"⚠ Column '{col}' has high cardinality ({df[col].nunique()} unique values)")

    return issues