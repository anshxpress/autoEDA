# AutoEDA

Automated Exploratory Data Analysis Notebook Generator

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

## Features

- Automatic dataset loading and type detection
- Missing value and duplicate analysis
- Statistical summaries for numerical and categorical features
- Visualizations: histograms, boxplots, bar charts, correlation heatmaps, pairplots
- Outlier detection using IQR method
- Generated Jupyter Notebook with all analysis

## Requirements

- Python 3.9+
- Libraries: pandas, numpy, matplotlib, seaborn, nbformat, scipy

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
