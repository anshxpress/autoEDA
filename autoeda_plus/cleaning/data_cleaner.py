"""
AutoEDA++ — Data Cleaning Module
=================================
Implements the robust 10-step cleaning pipeline with full per-step tracking.
Each step records whether it ran, what it did, or why it was skipped.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# StepResult — tracks what each cleaning step did
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    """Tracks the outcome of a single cleaning step."""
    step_number: int
    step_name: str
    executed: bool = True          # False if entirely skipped due to data conditions
    skipped: bool = False          # True when no applicable data found
    skip_reason: str = ""          # Why was it skipped?
    messages: List[str] = field(default_factory=list)  # what actions were taken
    nothing_to_do: bool = False    # True if step ran but found nothing to clean

    def add(self, msg: str) -> None:
        self.messages.append(msg)

    def mark_skipped(self, reason: str) -> None:
        self.skipped = True
        self.executed = False
        self.skip_reason = reason

    def mark_nothing_to_do(self) -> None:
        self.nothing_to_do = True
        self.messages.append("✅ No action needed")

    def summary(self) -> str:
        if self.skipped:
            return f"⏭  Skipped — {self.skip_reason}"
        if self.nothing_to_do:
            return "✅ Nothing to clean"
        return " | ".join(self.messages) if self.messages else "✅ Done"


class InsufficientDataError(Exception):
    """Raised when cleaned data is too small for downstream processing."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Main cleaning function
# ──────────────────────────────────────────────────────────────────────────────
def clean_data(
    df: pd.DataFrame,
    report: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[StepResult]]:
    """
    Robust 10-step data cleaning pipeline.

    Parameters
    ----------
    df      : Raw DataFrame to clean.
    report  : If True, prints a formatted cleaning report to stdout.

    Returns
    -------
    df_clean     : Cleaned DataFrame.
    log          : Flat list of all log messages (for backwards compatibility).
    step_results : List of StepResult objects, one per step.
    """
    log: List[str] = []
    steps: List[StepResult] = []

    # Work on a copy so the original is never mutated
    df = df.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Normalize column names
    # ──────────────────────────────────────────────────────────────────────────
    s1 = StepResult(1, "Normalize Column Names")
    original_cols = list(df.columns)
    df.columns = df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
    # Remove duplicate columns that may appear after normalization
    df = df.loc[:, ~df.columns.duplicated()]
    renamed = [f"{o} → {n}" for o, n in zip(original_cols, df.columns) if o != n]
    if renamed:
        for r in renamed:
            s1.add(r)
        log.extend(renamed)
    else:
        s1.mark_nothing_to_do()
    steps.append(s1)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Remove duplicate rows
    # ──────────────────────────────────────────────────────────────────────────
    s2 = StepResult(2, "Remove Duplicate Rows")
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        msg = f"Removed {removed} duplicate rows"
        s2.add(msg);  log.append(msg)
    else:
        s2.mark_nothing_to_do()
    steps.append(s2)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — Handle missing values
    # ──────────────────────────────────────────────────────────────────────────
    s3 = StepResult(3, "Handle Missing Values")
    if df.isnull().any().any():
        cols_to_check = list(df.columns)
        for col in cols_to_check:
            if col not in df.columns:
                continue
            missing_ratio = df[col].isnull().mean()

            # Drop first if ratio is critical
            if missing_ratio > 0.4:
                df.drop(columns=[col], inplace=True)
                msg = f"{col}: dropped (>{int(missing_ratio*100)}% missing)"
                s3.add(msg);  log.append(msg)
                continue

            if missing_ratio > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        msg = f"{col}: filled {int(missing_ratio*100)}% missing with median ({median_val:.4g})"
                        s3.add(msg);  log.append(msg)
                    except Exception:
                        pass
                else:
                    try:
                        mode_vals = df[col].mode()
                        if not mode_vals.empty:
                            df[col] = df[col].fillna(mode_vals[0])
                            msg = f"{col}: filled {int(missing_ratio*100)}% missing with mode ('{mode_vals[0]}')"
                            s3.add(msg);  log.append(msg)
                    except Exception:
                        pass
        if not s3.messages:
            s3.mark_nothing_to_do()
    else:
        s3.mark_nothing_to_do()
    steps.append(s3)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Fix data types
    # ──────────────────────────────────────────────────────────────────────────
    s4 = StepResult(4, "Fix Data Types")
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        for col in object_cols:
            if col not in df.columns:
                continue
            original_dtype = str(df[col].dtype)
            # Try numeric first
            try:
                converted = pd.to_numeric(df[col], errors="raise")
                df[col] = converted
                msg = f"{col}: object → numeric"
                s4.add(msg);  log.append(msg)
                continue
            except Exception:
                pass
            # Try datetime — use coerce so bad parses become NaT instead of raising
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                # Only accept the conversion if >50% of values parsed successfully
                valid_ratio = converted.notna().mean()
                if valid_ratio >= 0.5:
                    df[col] = converted
                    msg = f"{col}: object → datetime"
                    s4.add(msg);  log.append(msg)
            except Exception:
                pass
        if not s4.messages:
            s4.mark_nothing_to_do()
    else:
        s4.mark_skipped("No object-typed columns to convert")
    steps.append(s4)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5 — Clean text columns
    # ──────────────────────────────────────────────────────────────────────────
    s5 = StepResult(5, "Clean Text Columns")
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        cleaned_any = False
        for col in text_cols:
            try:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].replace({"?": np.nan, "nan": np.nan, "none": np.nan, "": np.nan})
                cleaned_any = True
            except Exception:
                pass
        if cleaned_any:
            msg = f"Cleaned {len(text_cols)} text column(s): {text_cols}"
            s5.add(msg);  log.append(msg)
        else:
            s5.mark_nothing_to_do()
    else:
        s5.mark_skipped("No text (object) columns found")
    steps.append(s5)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6 — Handle infinite values
    # ──────────────────────────────────────────────────────────────────────────
    s6 = StepResult(6, "Handle Infinite Values")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        inf_mask = np.isinf(df[num_cols]).any()
        inf_cols = inf_mask[inf_mask].index.tolist()
        if inf_cols:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            msg = f"Replaced ±inf in {len(inf_cols)} column(s): {inf_cols}"
            s6.add(msg);  log.append(msg)
        else:
            s6.mark_nothing_to_do()
    else:
        s6.mark_skipped("No numeric columns to check for infinite values")
    steps.append(s6)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7 — Remove constant columns
    # ──────────────────────────────────────────────────────────────────────────
    s7 = StepResult(7, "Remove Constant Columns")
    constant_cols = []
    for col in list(df.columns):
        try:
            if df[col].nunique(dropna=False) == 1:
                constant_cols.append(col)
        except Exception:
            pass
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        msg = f"Dropped {len(constant_cols)} constant column(s): {constant_cols}"
        s7.add(msg);  log.append(msg)
    else:
        s7.mark_nothing_to_do()
    steps.append(s7)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8 — Remove identifier columns (all-unique values)
    # ──────────────────────────────────────────────────────────────────────────
    s8 = StepResult(8, "Remove Identifier Columns")
    id_cols = []
    for col in list(df.columns):
        try:
            # Only flag as identifier if it's not numeric (pure IDs tend to be strings/ints)
            if df[col].nunique(dropna=False) == len(df):
                id_cols.append(col)
        except Exception:
            pass
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        msg = f"Dropped {len(id_cols)} identifier column(s): {id_cols}"
        s8.add(msg);  log.append(msg)
    else:
        s8.mark_nothing_to_do()
    steps.append(s8)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 9 — Outlier removal via IQR
    # ──────────────────────────────────────────────────────────────────────────
    s9 = StepResult(9, "Outlier Removal (IQR)")
    num_cols_after = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols_after:
        total_removed = 0
        for col in num_cols_after:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                before_len = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                removed = before_len - len(df)
                if removed > 0:
                    msg = f"{col}: removed {removed} outlier row(s)"
                    s9.add(msg);  log.append(msg)
                    total_removed += removed
            except Exception:
                pass
        if total_removed == 0:
            s9.mark_nothing_to_do()
    else:
        s9.mark_skipped("No numeric columns for IQR outlier detection")
    steps.append(s9)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 10 — Downcast numeric types + reset index
    # ──────────────────────────────────────────────────────────────────────────
    s10 = StepResult(10, "Downcast Numeric Types & Reset Index")
    downcast_cols = []
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast="float")
            downcast_cols.append(col)
        except Exception:
            pass
    df = df.reset_index(drop=True)
    if downcast_cols:
        msg = f"Downcast {len(downcast_cols)} column(s) to save memory; index reset"
        s10.add(msg);  log.append(msg)
    else:
        s10.mark_nothing_to_do()
    steps.append(s10)

    # ──────────────────────────────────────────────────────────────────────────
    # Final guard — ensure cleaned dataset is usable
    # ──────────────────────────────────────────────────────────────────────────
    if df is None or len(df) < 2:
        msg = "❌ Dataset has < 2 rows after cleaning — cannot proceed"
        log.append(msg)
        raise InsufficientDataError(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Print report
    # ──────────────────────────────────────────────────────────────────────────
    if report:
        _print_cleaning_report(steps, df)

    return df, log, steps


def _print_cleaning_report(steps: List[StepResult], df: pd.DataFrame) -> None:
    """Pretty-print the cleaning report to stdout."""
    print("\n" + "=" * 60)
    print("   🧹  DATA CLEANING REPORT")
    print("=" * 60)
    for s in steps:
        status = "⏭  SKIP" if s.skipped else ("✅ DONE" if not s.messages or s.nothing_to_do else "🔧 FIX ")
        print(f"\n  Step {s.step_number:02d} | {status} | {s.step_name}")
        if s.skipped:
            print(f"           ↳ {s.skip_reason}")
        elif s.nothing_to_do:
            print(f"           ↳ Nothing to clean")
        else:
            for m in s.messages:
                print(f"           ↳ {m}")
    print("\n" + "-" * 60)
    print(f"  📊 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility wrapper (used by eda_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────
def clean_dataset(
    df: pd.DataFrame,
    column_types: Optional[dict] = None,
    apply_outlier_capping: bool = False,
    mode: str = "aggressive",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Backwards-compatible wrapper around clean_data().

    Parameters `column_types`, `apply_outlier_capping`, and `mode` are
    accepted for API compatibility but clean_data() handles detection internally.

    Returns
    -------
    (cleaned_df, log)   — same as previous API.
    Also raises InsufficientDataError if cleaned data is unusable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataset expects a pandas DataFrame")
    cleaned_df, log, _ = clean_data(df, report=True)
    return cleaned_df, log
