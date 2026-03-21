import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


class InsufficientDataError(Exception):
    """Raised when cleaned data is insufficient for downstream processing."""
    pass


def clean_data(df: pd.DataFrame, report: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Basic data cleaning routine adapted from user-provided code.

    Returns cleaned DataFrame and a list of log messages.
    May raise ValueError("insufficient data") when the cleaned output is unusable.
    """
    log: List[str] = []

    # Work on a copy
    df = df.copy()

    # 1. Basic Cleaning
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    before = len(df)
    df = df.drop_duplicates()
    log.append(f"Removed {before - len(df)} duplicate rows")

    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Handle Missing Values
    for col in list(df.columns):
        missing_ratio = df[col].isnull().mean()

        if missing_ratio > 0:
            # numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    median = df[col].median()
                    df[col].fillna(median, inplace=True)
                    log.append(f"{col}: filled missing with median")
                except Exception:
                    pass
            else:
                try:
                    mode = df[col].mode()
                    if not mode.empty:
                        df[col].fillna(mode[0], inplace=True)
                        log.append(f"{col}: filled missing with mode")
                except Exception:
                    pass

        # Drop high missing columns
        if missing_ratio > 0.4 and col in df.columns:
            df.drop(columns=[col], inplace=True)
            log.append(f"{col}: dropped due to high missing values")

    # 3. Fix Data Types
    for col in list(df.columns):
        # try numeric
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except Exception:
            pass
        # try datetime
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass

    # 4. Clean Text Columns
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace("?", np.nan)
        except Exception:
            pass

    # 5. Handle Infinite Values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 6. Remove Constant Columns
    for col in list(df.columns):
        try:
            if df[col].nunique(dropna=False) == 1:
                df.drop(columns=[col], inplace=True)
                log.append(f"{col}: dropped constant column")
        except Exception:
            pass

    # 7. Remove Identifier Columns
    for col in list(df.columns):
        try:
            if df[col].nunique(dropna=False) == len(df):
                df.drop(columns=[col], inplace=True)
                log.append(f"{col}: dropped identifier column")
        except Exception:
            pass

    # 8. Outlier Handling (IQR)
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            after = len(df)

            if before != after:
                log.append(f"{col}: removed {before - after} outliers")
        except Exception:
            pass

    # 9. Downcast Numeric Types
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='float')
        except Exception:
            pass

    # 10. Reset Index
    df = df.reset_index(drop=True)

    # Final checks
    if df is None or len(df) == 0 or len(df) < 2:
        log.append("insufficient data")
        # Provide the custom error message as requested
        raise InsufficientDataError("insufficient data")

    # Output
    if report:
        print("\n=== DATA CLEANING REPORT ===")
        for item in log:
            print("-", item)

    return df, log


def clean_dataset(df: pd.DataFrame,
                  column_types: Optional[dict] = None,
                  apply_outlier_capping: bool = False,
                  mode: str = 'aggressive') -> Tuple[pd.DataFrame, List[str]]:
    """Compatibility wrapper used by the pipeline.

    Parameters mirror the previous project API. `apply_outlier_capping` and
    `mode` are accepted for compatibility but the current implementation
    follows the `clean_data` logic provided by the user.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataset expects a pandas DataFrame")

    # Delegate to the cleaning routine
    cleaned_df, report = clean_data(df, report=True)

    return cleaned_df, report
