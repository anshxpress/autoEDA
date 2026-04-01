"""
AutoEDA++ — Data Loader
=======================
Supports CSV, Excel, JSON, and Parquet with automatic encoding detection.
Also supports multi-file loading with vertical concatenation.
"""
import os
import pandas as pd
from typing import Tuple, List


SUPPORTED_FORMATS = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel (legacy)",
    ".json": "JSON",
    ".parquet": "Parquet",
}


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a data file into a DataFrame.

    Supports CSV (auto-encoding), Excel (.xlsx/.xls), JSON, and Parquet.

    Parameters
    ----------
    file_path : Path to the data file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")

    ext = os.path.splitext(file_path)[1].lower()
    fmt = SUPPORTED_FORMATS.get(ext)

    if fmt is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: {list(SUPPORTED_FORMATS)}"
        )

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    print(f"\n[Step 1] 📂 Loading {fmt} file: {os.path.basename(file_path)!r}  "
          f"({file_size_mb:.2f} MB)")

    if ext == ".csv":
        df = _load_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")

    _validate(df, file_path)
    print(f"         ✅ Loaded successfully — {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def _load_csv(file_path: str) -> pd.DataFrame:
    """Try multiple encodings for CSV files."""
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Final fallback
    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Could not load CSV with any known encoding: {e}")


def _validate(df: pd.DataFrame, file_path: str) -> None:
    if df.empty:
        raise ValueError(f"The file '{file_path}' loaded an empty DataFrame.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns.")


# ── Backwards-compatible alias used by legacy and notebook code ────────────────
def load_csv(file_path: str) -> pd.DataFrame:
    """Backwards-compatible alias for load_data(). Accepts CSV only path."""
    return load_data(file_path)


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate a loaded DataFrame. Returns (is_valid, message)."""
    if df.empty:
        return False, "Dataset is empty"
    if df.shape[1] < 2:
        return False, "Dataset must have at least 2 columns"
    return True, "Dataset loaded successfully"


# ── Multi-file loader ─────────────────────────────────────────────────────────
def load_multiple_files(
    file_paths: List[str],
    add_source_col: bool = True,
) -> pd.DataFrame:
    """
    Load multiple data files and concatenate them vertically (row-wise).

    Each file is loaded with full format support (CSV, Excel, JSON, Parquet).
    A 'source_file' column is added to track provenance.

    Parameters
    ----------
    file_paths     : List of paths to data files.
    add_source_col : If True, adds a 'source_file' column (default True).

    Returns
    -------
    pd.DataFrame — merged dataset.
    """
    if not file_paths:
        raise ValueError("No file paths provided to load_multiple_files()")

    frames: List[pd.DataFrame] = []
    for path in file_paths:
        df = load_data(path)
        if add_source_col:
            df["source_file"] = os.path.basename(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    print(f"\n[Loader] 🔗 Merged {len(file_paths)} file(s) → "
          f"{merged.shape[0]:,} rows × {merged.shape[1]} columns")
    if add_source_col:
        counts = merged["source_file"].value_counts()
        for fname, cnt in counts.items():
            print(f"          ↳ {fname}: {cnt:,} rows")

    return merged