#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoeda_plus.notebook.notebook_builder import build_comprehensive_eda_notebook

def main() -> None:
    parser = argparse.ArgumentParser(
        description='AutoEDA++: Automated Intelligent Exploratory Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autoeda data.csv
  autoeda data.csv --output my_report.ipynb
  autoeda data.csv --summary-only
        """
    )
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output notebook filename (default: EDA_<dataset>.ipynb)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Generate only summary statistics without plots')

    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"EDA_{base_name}.ipynb"

    output_path = os.path.join('output', output_path)

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    try:
        print(f"Analyzing {csv_path}...")
        build_comprehensive_eda_notebook(csv_path, output_path)
        print(f"✅ EDA report generated: {output_path}")
        print("📊 Analysis complete! Open the notebook to explore your data insights.")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()