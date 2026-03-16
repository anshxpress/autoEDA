import pandas as pd
from typing import Tuple

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with automatic type inference.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        # Try to infer dates and handle encoding
        df = pd.read_csv(file_path, parse_dates=True, low_memory=False)
        return df
    except UnicodeDecodeError:
        # Fallback to latin1 encoding
        df = pd.read_csv(file_path, encoding='latin1', parse_dates=True, low_memory=False)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the loaded dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if df.empty:
        return False, "Dataset is empty"

    if df.shape[1] < 2:
        return False, "Dataset must have at least 2 columns"

    return True, "Dataset loaded successfully"