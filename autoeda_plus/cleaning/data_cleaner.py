import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df2 = df.drop_duplicates()
    removed = before - len(df2)
    return df2, int(removed)


def handle_missing_values(df: pd.DataFrame, column_types: Dict[str, str], drop_threshold: float = 0.4) -> Tuple[pd.DataFrame, List[str]]:
    """Fill missing values by heuristic; drop columns with > drop_threshold missing fraction."""
    report = []
    df2 = df.copy()
    n = len(df2)
    for col in df2.columns:
        miss = df2[col].isna().sum()
        if miss == 0:
            continue
        frac = miss / n
        if frac > drop_threshold:
            df2.drop(columns=[col], inplace=True)
            report.append(f"Dropped column {col} due to high missing fraction ({frac:.2%})")
            continue

        ctype = column_types.get(col, 'unknown')
        if ctype == 'numerical':
            fill = df2[col].median()
        else:
            mode = df2[col].mode()
            fill = mode.iloc[0] if not mode.empty else 'Unknown'
        df2[col] = df2[col].fillna(fill)
        report.append(f"Filled missing in {col} with {fill}")

    return df2, report


def correct_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    report = []
    df2 = df.copy()
    for col in df2.columns:
        # try numeric
        if df2[col].dtype == object:
            try:
                conv = pd.to_numeric(df2[col], errors='raise')
                df2[col] = conv
                report.append(f"Converted column {col} to numeric")
                continue
            except Exception:
                pass
            # try datetime
            try:
                conv = pd.to_datetime(df2[col], errors='raise')
                df2[col] = conv
                report.append(f"Converted column {col} to datetime")
            except Exception:
                pass

    return df2, report


def normalize_categoricals(df: pd.DataFrame, column_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    report = []
    df2 = df.copy()
    for col, ctype in column_types.items():
        if ctype == 'categorical' and col in df2.columns:
            if df2[col].dtype == object:
                df2[col] = df2[col].astype(str).str.strip().str.lower()
                report.append(f"Standardized categorical column {col}")
    return df2, report


def detect_identifier_columns(df: pd.DataFrame, unique_ratio_threshold: float = 0.98) -> List[str]:
    ids = []
    n = len(df)
    for col in df.columns:
        if n == 0:
            continue
        uniq = df[col].nunique(dropna=False)
        if uniq / n >= unique_ratio_threshold:
            ids.append(col)
    return ids


def remove_identifier_columns(df: pd.DataFrame, ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df2 = df.copy()
    removed = []
    for col in ids:
        if col in df2.columns:
            df2.drop(columns=[col], inplace=True)
            removed.append(col)
    return df2, removed


def cap_outliers_iqr(df: pd.DataFrame, numerical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df2 = df.copy()
    report = []
    for col in numerical_cols:
        if col not in df2.columns:
            continue
        try:
            Q1 = df2[col].quantile(0.25)
            Q3 = df2[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = df2[col].copy()
            df2[col] = df2[col].clip(lower, upper)
            report.append(f"Capped outliers in {col} to [{lower:.3f}, {upper:.3f}]")
        except Exception:
            continue
    return df2, report


def clean_dataset(df: pd.DataFrame, column_types: Dict[str, str], apply_outlier_capping: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Run the full cleaning pipeline and return cleaned df plus a report list."""
    report: List[str] = []
    df2, removed_dup = remove_duplicates(df)
    if removed_dup:
        report.append(f"Removed {removed_dup} duplicate rows")

    ids = detect_identifier_columns(df2)
    if ids:
        df2, removed_ids = remove_identifier_columns(df2, ids)
        report.append(f"Removed identifier columns: {removed_ids}")

    df2, miss_report = handle_missing_values(df2, column_types)
    report.extend(miss_report)

    df2, dtype_report = correct_dtypes(df2)
    report.extend(dtype_report)

    df2, cat_report = normalize_categoricals(df2, column_types)
    report.extend(cat_report)

    if apply_outlier_capping:
        num_cols = [c for c, t in column_types.items() if t == 'numerical' and c in df2.columns]
        df2, out_report = cap_outliers_iqr(df2, num_cols)
        report.extend(out_report)

    return df2, report
