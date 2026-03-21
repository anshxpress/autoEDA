#!/usr/bin/env python3
"""
AutoEDA++ CLI
==============
Usage:
  python -m autoeda_plus.cli.autoeda data.csv
  python -m autoeda_plus.cli.autoeda data.csv --output report.ipynb
  python -m autoeda_plus.cli.autoeda data.csv --no-clean
  python -m autoeda_plus.cli.autoeda data.csv --no-plots
  python -m autoeda_plus.cli.autoeda data.csv --summary-only
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path when run directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoeda_plus.core.eda_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoEDA++ — Automated Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline steps:
  [Step 1] Data Loading    — CSV / Excel / JSON / Parquet
  [Step 2] Data Cleaning   — 10-step robust cleaning (default ON)
  [Step 3] EDA & Analysis  — stats, correlations, outliers, insights
  [Step 4] Notebook Gen    — Jupyter notebook with all results

Examples:
  autoeda data.csv
  autoeda data.csv -o my_report.ipynb
  autoeda data.csv --no-clean
  autoeda data.csv --summary-only --no-plots
        """,
    )

    parser.add_argument(
        "file",
        help="Path to the input data file (CSV, Excel .xlsx, JSON, Parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output notebook path (default: output/EDA_<dataset>.ipynb)"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the data cleaning step"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate notebook without visualization cells"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only include summary statistics (no distribution plots)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: '{args.file}'")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_path = os.path.join("output", f"EDA_{base_name}.ipynb")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("\n" + "=" * 60)
    print("   🚀  AutoEDA++ — Automated EDA Pipeline")
    print("=" * 60)
    print(f"  Input  : {args.file}")
    print(f"  Output : {output_path}")
    print(f"  Clean  : {'OFF (--no-clean)' if args.no_clean else 'ON'}")
    print("=" * 60)

    try:
        run_pipeline(
            args.file,
            output_path=output_path,
            clean=not args.no_clean,
            no_plots=args.no_plots,
            summary_only=args.summary_only,
        )
        print(f"\n📓 Open your notebook: {output_path}")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()