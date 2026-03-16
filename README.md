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

To run tests:

```bash
pip install pytest
pytest
```

To lint:

```bash
pip install ruff
ruff check .
```

To type check:

```bash
pip install mypy
mypy .
```
