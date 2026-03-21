# CLI AutoEDA

Automated Intelligent Exploratory Data Analysis

---

## 🚀 Quick Start (Working Setup – Verified)

### 1. Clone the Repository

```powershell
git clone https://github.com/anshxpress/autoEDA.git
cd autoEDA
```

---

### 2. Create Virtual Environment

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

---

### 3. Install Dependencies

```powershell
cd autoEDA
pip install -r requirements.txt
```

---

### 4. Run AutoEDA++

#### ▶ Run with sample dataset:

```powershell
python -m autoeda_plus.cli.autoeda "sample.csv"
```

#### ▶ Run with custom dataset:

```powershell
python -m autoeda_plus.cli.autoeda "D:\autoEDA\autoEDA\retail_store_sales.csv"
```

---

## 📂 Output

- Output notebook is generated automatically inside:

```bash
autoEDA/output/
```

Example:

```bash
EDA_retail_store_sales.ipynb
```

Open with:

```powershell
code output\EDA_retail_store_sales.ipynb
```

OR

```powershell
jupyter notebook output
```

---

## ⚙️ CLI Options

```bash
--output <file.ipynb>     Custom output notebook name
--summary-only           Generate only summary statistics
--clean                  Apply automatic data cleaning
--no-plots               Skip visualization generation
--cap-outliers           Cap outliers using IQR (with --clean)
```

---

## 🧪 Example Commands

### ✔ Full analysis

```powershell
python -m autoeda_plus.cli.autoeda "data.csv"
```

### ✔ With cleaning

```powershell
python -m autoeda_plus.cli.autoeda "data.csv" --clean
```

### ✔ Fast mode (no plots)

```powershell
python -m autoeda_plus.cli.autoeda "data.csv" --summary-only
```

### ✔ Clean + optimize

```powershell
python -m autoeda_plus.cli.autoeda "data.csv" --clean --cap-outliers --no-plots
```

---

## 🔥 Features

- Automatic dataset profiling
- Missing value & duplicate detection
- Statistical summaries (numerical + categorical)
- Data quality diagnostics
- Visualization system (histograms, boxplots, heatmaps)
- Correlation analysis
- Outlier detection (IQR + Z-score)
- Feature engineering suggestions
- Baseline ML models
- Structured Jupyter notebook generation

---

## 🧠 Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- nbformat

---

## ⚠️ Custom Commands

# From d:\autoeda\autoEDA
python -m autoeda_plus.cli.autoeda "D:\file.csv"

# Custom output
python -m autoeda_plus.cli.autoeda "D:\file.csv" -o output\my_report.ipynb

# Skip cleaning
python -m autoeda_plus.cli.autoeda "D:\file.csv" --no-clean


## 📌 Project Structure

```
autoEDA/
│
├── autoeda_plus/
│   ├── cli/
│   ├── cleaning/
│   ├── notebook/
│
├── output/
├── autoeda_runner.py
├── requirements.txt
```

---

## 💡 Notes

- Always run commands from inside:

```
autoEDA/autoEDA
```

- Use full file paths to avoid path issues
- Works fully offline

---

## 🚀 Future Improvements (Optional)

- Multi-CSV relational EDA
- LLM-based insights generation
- Streamlit dashboard integration
- Automated feature store

---

## ⭐ Contribute

Feel free to fork, improve, and raise PRs!

