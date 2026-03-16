import pandas as pd
from typing import Tuple, Optional
from ..core.schema_detector import detect_column_types, detect_potential_target

def determine_problem_type(df: pd.DataFrame, target_col: str, column_types: Dict[str, str]) -> str:
    """
    Determine if it's a classification or regression problem.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column name.
        column_types (Dict[str, str]): Column types.

    Returns:
        str: 'classification' or 'regression'
    """
    if column_types.get(target_col) == 'categorical':
        return 'classification'
    elif column_types.get(target_col) == 'numerical':
        # Check if target has few unique values (ordinal classification)
        if df[target_col].nunique() <= 10:
            return 'classification'
        else:
            return 'regression'
    else:
        # Default to regression
        return 'regression'

def prepare_features_and_target(df: pd.DataFrame, target_col: str, column_types: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column.
        column_types (Dict[str, str]): Column types.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target.
    """
    # Select features (exclude identifiers and target)
    feature_cols = [col for col, ctype in column_types.items()
                   if ctype not in ['identifier'] and col != target_col]

    # Simple preprocessing
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle missing values (simple imputation)
    for col in X.columns:
        if X[col].isnull().any():
            if column_types[col] == 'numerical':
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')

    # Encode categorical features
    for col in X.columns:
        if column_types[col] == 'categorical':
            X[col] = X[col].astype('category').cat.codes

    return X, y

def select_baseline_model(problem_type: str) -> str:
    """
    Select appropriate baseline model.

    Args:
        problem_type (str): 'classification' or 'regression'

    Returns:
        str: Model name
    """
    if problem_type == 'classification':
        return 'RandomForestClassifier'
    else:
        return 'RandomForestRegressor'