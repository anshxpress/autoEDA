"""EDA pipeline orchestrator for AutoEDA++

Provides a single entrypoint to run cleaning, profiling, analysis and notebook generation
for a CSV file. Re-uses existing modules in the project.
"""
from typing import Dict, Any
import os
from pprint import pformat

import pandas as pd

from autoeda_plus.core.data_loader import load_csv
from autoeda_plus.core.schema_detector import detect_column_types, detect_potential_target
from autoeda_plus.cleaning.data_cleaner import clean_dataset
from autoeda_plus.core.data_profiler import profile_dataset, detect_data_quality_issues
from autoeda_plus.analysis.statistics_engine import compute_numerical_statistics, compute_categorical_statistics
from autoeda_plus.analysis.correlation_engine import compute_correlation_matrix
from autoeda_plus.analysis.outlier_detector import get_outlier_summary
from autoeda_plus.insights.insight_generator import generate_insights, generate_feature_engineering_suggestions
from autoeda_plus.ml.baseline_models import train_baseline_model
from autoeda_plus.notebook.notebook_builder import build_comprehensive_eda_notebook


def run_pipeline(csv_path: str,
                 output_path: str = None,
                 *,
                 clean: bool = False,
                 no_plots: bool = False,
                 summary_only: bool = False,
                 cap_outliers: bool = False
) -> Dict[str, Any]:
    """Run the full EDA pipeline for a CSV file.

    Returns a dictionary summary with profiling, cleaning report, analysis and ML results.
    Also writes a notebook via the notebook builder.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join('output', f'EDA_{base}.ipynb')

    # Load
    df = load_csv(csv_path)

    # Schema detection
    column_types = detect_column_types(df)

    # Cleaning
    cleaning_report = []
    cleaned_df = df
    if clean:
        cleaned_df, cleaning_report = clean_dataset(df, column_types, apply_outlier_capping=cap_outliers)
        # re-detect types
        column_types = detect_column_types(cleaned_df)

    # Profiling and diagnostics
    profile = profile_dataset(cleaned_df)
    quality_issues = detect_data_quality_issues(cleaned_df, column_types)

    # Statistics
    num_stats = compute_numerical_statistics(cleaned_df, column_types)
    cat_stats = compute_categorical_statistics(cleaned_df, column_types)

    # Correlation & outliers
    corr_matrix = compute_correlation_matrix(cleaned_df, column_types)
    outlier_summary = get_outlier_summary(cleaned_df, column_types)

    # Target detection and baseline modeling
    target_col = detect_potential_target(cleaned_df, column_types)
    ml_results = train_baseline_model(cleaned_df, target_col, column_types)

    # Insights and suggestions
    insights = generate_insights(cleaned_df, column_types, corr_matrix, outlier_summary)
    fe_suggestions = generate_feature_engineering_suggestions(column_types, num_stats)

    # Ensure output dir exists and build notebook
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    build_comprehensive_eda_notebook(csv_path, output_path,
                                     clean=clean,
                                     no_plots=no_plots,
                                     summary_only=summary_only,
                                     cap_outliers=cap_outliers)

    summary = {
        'csv_path': csv_path,
        'output_path': output_path,
        'cleaning_report': cleaning_report,
        'profile': profile,
        'quality_issues': quality_issues,
        'numerical_stats_sample': {k: v for k, v in (list(num_stats.items())[:5])},
        'categorical_stats_sample': {k: v for k, v in (list(cat_stats.items())[:5])},
        'corr_matrix_shape': None if corr_matrix is None else corr_matrix.shape,
        'outlier_summary_sample': dict(list(outlier_summary.items())[:5]),
        'target_col': target_col,
        'ml_results': ml_results,
        'insights_sample': insights[:6],
        'fe_suggestions_sample': fe_suggestions[:6]
    }

    # Pretty-print a small log to console
    print("\n=== AutoEDA++ pipeline summary ===")
    print(pformat({k: (v if k in ('cleaning_report','target_col') else (str(v)[:500])) for k, v in summary.items()}))

    return summary


if __name__ == '__main__':
    # Simple CLI for local testing
    import argparse

    parser = argparse.ArgumentParser(description='Run AutoEDA++ pipeline on a CSV file')
    parser.add_argument('csv_file')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--cap-outliers', action='store_true')

    args = parser.parse_args()
    run_pipeline(args.csv_file, output_path=args.output, clean=args.clean, no_plots=args.no_plots, summary_only=args.summary_only, cap_outliers=args.cap_outliers)
