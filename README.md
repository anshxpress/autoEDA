# AutoEDA

Automated Exploratory Data Analysis Notebook Generator

## Installation

1. Clone or download the repository.
2. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the tool with a CSV file:

```bash
python  $env:PYTHONPATH = "d:\autoEDA\autoeda"; python cli\autoeda.py'location.csv'                                                         
```

Options:

- `--output filename`: Specify output notebook name (default: EDA_dataset.ipynb)

## Features

- Automatic dataset loading and type detection
- Missing value and duplicate analysis
- Statistical summaries for numerical and categorical features
- Visualizations: histograms, boxplots, bar charts, correlation heatmaps
- Outlier detection
- Generated Jupyter Notebook with all analysis

## Requirements

- Python 3.9+
- Libraries: pandas, numpy, matplotlib, seaborn, nbformat, scipy
