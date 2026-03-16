#!/usr/bin/env python3
import argparse
import os
from nb_builder.notebook_builder import build_eda_notebook

def main() -> None:
    parser = argparse.ArgumentParser(description='AutoEDA: Automated Exploratory Data Analysis Notebook Generator')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--output', '-o', default=None, help='Output notebook filename (default: EDA_<csv_filename>.ipynb)')

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

    try:
        build_eda_notebook(csv_path, output_path)
        print(f"EDA notebook generated: {output_path}")
    except Exception as e:
        print(f"Error generating notebook: {e}")

if __name__ == '__main__':
    main()