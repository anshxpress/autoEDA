#!/usr/bin/env python3
"""
AutoEDA++ Professional CLI
==========================
Advanced command-line entrypoint for robust EDA and Anomaly Detection.
Supports Parquet loading, multi-file inputs, automated submission logic, 
and professional model artifact persistence (Joblib).
"""
import argparse
import os
import sys
import logging
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoeda_plus.core.eda_pipeline import run_pipeline, run_anomaly_pipeline, setup_logging

# Configure logger early
logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoEDA++ — Production-grade Anomaly Detection & EDA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional or named files
    parser.add_argument("file", nargs="?", default=None, help="Input data file (CSV/Excel/JSON/Parquet)")

    # Core Options
    parser.add_argument("--output", "-o", default=None, help="Output notebook/artifact directory")
    parser.add_argument("--no-clean", action="store_true", help="Skip data cleaning step")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization generation")
    parser.add_argument("--summary-only", action="store_true", help="Quick summary stats only")

    # Anomaly Detection Specific Flags
    parser.add_argument("--anomaly", action="store_true", help="Enable anomaly detection mode")
    parser.add_argument("--target", default=None, help="Target column for supervised detection")
    parser.add_argument("--test-file", default=None, help="Test data file for submission predictions")
    parser.add_argument("--files", nargs="+", default=None, help="Process multiple files (vertical concat)")
    parser.add_argument("--window", type=int, default=5, help="Rolling statistics window size")
    parser.add_argument("--lag-steps", type=int, default=2, help="Lag feature count")
    parser.add_argument("--contamination", type=float, default=0.08, help="Anomaly fraction (unsupervised)")
    parser.add_argument("--no-tune", action="store_true", help="Disable GridSearchCV tuning")
    parser.add_argument("--smote", action="store_true", help="Enable SMOTE imbalance handling")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging level")

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.anomaly:
        # File resolution logic
        file_list = args.files if args.files else ([args.file] if args.file else None)
        if not file_list:
            logger.error("❌ No input files provided for anomaly detection.")
            sys.exit(1)

        try:
            run_anomaly_pipeline(
                file_paths=file_list,
                output_path=args.output,
                target_col=args.target,
                test_file=args.test_file,
                clean=not args.no_clean,
                no_plots=args.no_plots,
                window=args.window,
                lag_steps=args.lag_steps,
                contamination=args.contamination,
                tune_models=not args.no_tune,
                use_smote=args.smote,
            )
        except Exception as e:
            logger.error(f"❌ Critical Pipeline Failure: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    else:
        # Standard EDA flow
        if not args.file:
            parser.print_help()
            sys.exit(1)
        run_pipeline(args.file, output_path=args.output, clean=not args.no_clean, 
                     no_plots=args.no_plots, summary_only=args.summary_only)


if __name__ == "__main__":
    main()