# AutoEDA++

Automated Intelligent Exploratory Data Analysis

## Installation

1. Clone or download the repository.
2. Install dependencies: `pip install -r requirements.txt`

Or install as a package:

```bash
pip install -e .
```

## Usage

Run the tool with a CSV file:

```bash
autoeda sample.csv
```

Or with custom output:

```bash
autoeda sample.csv --output my_analysis.ipynb
```

Options:

- `--output filename`: Specify output notebook name (default: EDA_dataset.ipynb)
- `--summary-only`: Generate only summary statistics without plots
- `--clean`: Apply automatic data cleaning (optional)
- `--no-plots`: Skip plot generation (keep summaries only)
- `--cap-outliers`: When used with `--clean`, cap outliers using IQR

## Features

- **Automatic dataset profiling**: Column type inference, missing value detection, duplicate analysis
- **Statistical summaries**: Comprehensive metrics for numerical and categorical features
- **Data quality diagnostics**: Detection of common dataset issues and inconsistencies
- **Visualization system**: Histograms, boxplots, count plots, correlation heatmaps, scatter plots
- **Correlation analysis**: Matrix computation and strong relationship detection
- **Outlier detection**: IQR and Z-score methods with visual confirmation
- **Feature engineering suggestions**: Automated recommendations for data preprocessing
- **Baseline machine learning models**: Automatic training and evaluation of simple models
- **Structured notebook generation**: Comprehensive Jupyter reports with all analyses

## Requirements

- Python 3.9+
- Libraries: pandas, numpy, matplotlib, seaborn, nbformat, scipy, scikit-learn

## Development

## Quick start (3 steps)

1. Download the repo

```powershell
git clone https://github.com/anshxpress/autoEDA.git
cd autoEDA\autoeda
```

2. Install dependencies (one-time)

```powershell
python -m pip install -r requirements.txt
```

3. Run AutoEDA on a CSV (examples below)

Notes:

- Place your CSV anywhere and pass its path to the command.
- Output notebook is written to the `output/` folder by default.

## Simple run examples

- Run with the local runner (recommended):

```powershell
python ..\autoeda_runner.py "D:\path\to\your_dataset.csv"
```

- Run the bundled batch wrapper (Windows):

```powershell
.\autoeda.bat "D:\path\to\your_dataset.csv"
```

- Install a global launcher (one-line) and run `autoeda` directly:

```powershell
powershell -ExecutionPolicy RemoteSigned -File .\install_autoeda.ps1
# then in a new shell
autoeda "D:\path\to\your_dataset.csv"
```

## Useful flags

- `--clean` : apply automatic data cleaning (drops high-missing columns, imputes missing, normalizes categories)
- `--cap-outliers` : when used with `--clean` caps numeric outliers using IQR
- `--no-plots` : skip plot generation (summary-only)
- `--summary-only` : generate only summaries; no extensive plots
- `--output <file.ipynb>` : write notebook to a custom path

Example with cleaning and plots disabled:

```powershell
autoeda "D:\path\to\your_dataset.csv" --clean --no-plots --output my_report.ipynb
```

## How to add a CSV

- Copy your CSV into any folder. You can keep it outside the repo. Examples:
  - `D:\mydata\sales.csv`
  - `C:\Users\you\Downloads\data.csv`
- Pass the full path (recommended) or a relative path to the runner/batch command above.

## Output

- The tool generates a Jupyter notebook at `output/EDA_<dataset_basename>.ipynb`.
- Open it with VS Code or Jupyter: `code output\EDA_<name>.ipynb` or `jupyter notebook output`.

## Troubleshooting

- If `autoeda` command is not found after installing, open a new PowerShell window so PATH updates take effect.
- If PowerShell blocks the installer, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

- If Python cannot import modules, ensure you installed `requirements.txt` in the same Python environment you run the commands with.
- If CSV encoding fails, the loader tries `utf-8`, `latin1`, `cp1252`, then fallback without encoding — re-save CSV as UTF-8 if needed.

## Where to look in the code

- Cleaning logic: [autoeda_plus/cleaning/data_cleaner.py](autoeda_plus/cleaning/data_cleaner.py)
- Runner: [autoeda_runner.py](autoeda_runner.py)
- Batch wrapper: [autoeda.bat](autoeda.bat)
- CLI entrypoint: [autoeda_plus/cli/autoeda.py](autoeda_plus/cli/autoeda.py)
- Notebook builder: [autoeda_plus/notebook/notebook_builder.py](autoeda_plus/notebook/notebook_builder.py)

---

If you want, I can add a one-line PowerShell snippet to your `README` that automatically places the `autoeda` launcher into your user PATH (requires confirmation).
