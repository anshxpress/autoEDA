import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from typing import Dict, Any
from .model_selector import determine_problem_type, prepare_features_and_target, select_baseline_model

def train_baseline_model(df: pd.DataFrame, target_col: str, column_types: Dict[str, str]) -> Dict[str, Any]:
    """
    Train a baseline ML model and return performance metrics.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column.
        column_types (Dict[str, str]): Column types.

    Returns:
        Dict[str, Any]: Model results.
    """
    try:
        problem_type = determine_problem_type(df, target_col, column_types)
        X, y = prepare_features_and_target(df, target_col, column_types)

        # Check if we have enough data
        if len(X) < 10 or X.shape[1] < 1:
            return {
                'success': False,
                'error': 'Insufficient data for modeling'
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select and train model
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'r2': r2_score(y_test, y_pred)
            }

        return {
            'success': True,
            'problem_type': problem_type,
            'model': select_baseline_model(problem_type),
            'metrics': metrics,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }