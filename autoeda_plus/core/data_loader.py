import pandas as pd
from typing import Tuple

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with automatic encoding detection.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, parse_dates=True, low_memory=False)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If all encodings fail, try without encoding (pandas default)
    try:
        df = pd.read_csv(file_path, parse_dates=True, low_memory=False)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file with any encoding: {e}")

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