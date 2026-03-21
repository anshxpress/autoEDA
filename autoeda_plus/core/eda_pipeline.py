"""
AutoEDA++ — EDA Pipeline Orchestrator
======================================
Single entrypoint for the full 4-step pipeline:
  Step 1 — Data Loading
  Step 2 — Data Cleaning
  Step 3 — EDA & Visualization analysis
  Step 4 — Notebook Generation
"""
from typing import Dict, Any, Optional
import os
from pprint import pformat

import pandas as pd

from autoeda_plus.core.data_loader import load_data, validate_dataset
from autoeda_plus.core.schema_detector import detect_column_types, detect_potential_target
from autoeda_plus.cleaning.data_cleaner import clean_data, clean_dataset, InsufficientDataError
from autoeda_plus.core.data_profiler import profile_dataset, detect_data_quality_issues
from autoeda_plus.analysis.statistics_engine import (
    compute_numerical_statistics, compute_categorical_statistics,
)
from autoeda_plus.analysis.correlation_engine import compute_correlation_matrix
from autoeda_plus.analysis.outlier_detector import get_outlier_summary
from autoeda_plus.insights.insight_generator import (
    generate_insights, generate_feature_engineering_suggestions,
)
from autoeda_plus.ml.baseline_models import train_baseline_model
from autoeda_plus.notebook.notebook_builder import build_comprehensive_eda_notebook


def run_pipeline(
    file_path: str,
    output_path: Optional[str] = None,
    *,
    clean: bool = True,          # Cleaning is ON by default
    no_plots: bool = False,
    summary_only: bool = False,
    cap_outliers: bool = False,
) -> Dict[str, Any]:
    """
    Run the full AutoEDA++ pipeline.

    Parameters
    ----------
    file_path   : Path to the input data file (CSV, Excel, JSON, Parquet).
    output_path : Where to write the notebook. Defaults to output/EDA_<name>.ipynb.
    clean       : Run the cleaning module (default True). Pass False to skip.
    no_plots    : Generate notebook without plots.
    summary_only: Only summary statistics, no plots.
    cap_outliers: Cap outliers instead of removing them (future use).

    Returns
    -------
    dict — full pipeline summary with profiling, analysis and ML results.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join("output", f"EDA_{base}.ipynb")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Load Data
    # ──────────────────────────────────────────────────────────────────────────
    df = load_data(file_path)
    is_valid, msg = validate_dataset(df)
    if not is_valid:
        raise ValueError(msg)
    print(f"[Step 1] ✅ Data Loaded  "
          f"({df.shape[0]:,} rows × {df.shape[1]} columns)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Data Cleaning
    # ──────────────────────────────────────────────────────────────────────────
    cleaning_log = []
    step_results = []
    cleaned_df = df

    if clean:
        print("[Step 2] 🧹 Running Data Cleaning...")
        try:
            cleaned_df, cleaning_log, step_results = clean_data(df, report=True)
            print(f"[Step 2] ✅ Data Cleaned  "
                  f"({cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns)\n")
        except InsufficientDataError as e:
            cleaning_log = [str(e)]
            print(f"[Step 2] ❌ Cleaning Error: {e}")
            print("         ↳ Proceeding with uncleaned data for EDA.\n")
            cleaned_df = df
    else:
        print("[Step 2] ⏭  Data Cleaning skipped (--no-clean flag)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — EDA & Analysis
    # ──────────────────────────────────────────────────────────────────────────
    print("[Step 3] 📊 Running EDA & Analysis...")

    column_types = detect_column_types(cleaned_df)
    profile = profile_dataset(cleaned_df)
    quality_issues = detect_data_quality_issues(cleaned_df, column_types)

    num_stats = compute_numerical_statistics(cleaned_df, column_types)
    cat_stats = compute_categorical_statistics(cleaned_df, column_types)

    corr_matrix = compute_correlation_matrix(cleaned_df, column_types)
    outlier_summary = get_outlier_summary(cleaned_df, column_types)

    target_col = detect_potential_target(cleaned_df, column_types)
    ml_results = train_baseline_model(cleaned_df, target_col, column_types)

    insights = generate_insights(cleaned_df, column_types, corr_matrix, outlier_summary)
    fe_suggestions = generate_feature_engineering_suggestions(column_types, num_stats)

    print(f"[Step 3] ✅ EDA Complete  "
          f"({len(insights)} insights | "
          f"{len(num_stats)} numeric cols | "
          f"{len(cat_stats)} categorical cols)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Notebook Generation
    # ──────────────────────────────────────────────────────────────────────────
    print(f"[Step 4] 📓 Generating Notebook → {output_path}")

    build_comprehensive_eda_notebook(
        csv_path=file_path,
        output_path=output_path,
        cleaned_df=cleaned_df,
        cleaning_log=cleaning_log,
        step_results=step_results,
        column_types=column_types,
        num_stats=num_stats,
        cat_stats=cat_stats,
        corr_matrix=corr_matrix,
        outlier_summary=outlier_summary,
        quality_issues=quality_issues,
        insights=insights,
        fe_suggestions=fe_suggestions,
        ml_results=ml_results,
        target_col=target_col,
        no_plots=no_plots,
        summary_only=summary_only,
        did_clean=clean,
    )

    print(f"[Step 4] ✅ Notebook Generated: {output_path}\n")
    print("=" * 60)
    print("   🎉  AutoEDA++ pipeline complete!")
    print("=" * 60)

    return {
        "file_path": file_path,
        "output_path": output_path,
        "cleaning_log": cleaning_log,
        "step_results": step_results,
        "profile": profile,
        "quality_issues": quality_issues,
        "num_stats": num_stats,
        "cat_stats": cat_stats,
        "corr_matrix_shape": None if corr_matrix is None else corr_matrix.shape,
        "outlier_summary": outlier_summary,
        "target_col": target_col,
        "ml_results": ml_results,
        "insights": insights,
        "fe_suggestions": fe_suggestions,
    }


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="AutoEDA++ — Run the full EDA pipeline on a data file"
    )
    parser.add_argument("file", help="Path to the data file (CSV, Excel, JSON, Parquet)")
    parser.add_argument("--output", "-o", default=None, help="Output notebook path")
    parser.add_argument("--no-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument("--no-plots", action="store_true", help="No visualizations")
    parser.add_argument("--summary-only", action="store_true", help="Summary stats only")

    args = parser.parse_args()
    run_pipeline(
        args.file,
        output_path=args.output,
        clean=not args.no_clean,
        no_plots=args.no_plots,
        summary_only=args.summary_only,
    )
