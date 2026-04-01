# AutoEDA++ — Anomaly Detection Pipeline

**End-to-End Sensor Data Analysis & Anomaly Prediction**

> Automated pipeline that transforms raw sensor data into actionable anomaly predictions using statistical analysis and machine learning.

---

## 🚀 Quick Start

### 1. Clone & Setup

```powershell
git clone https://github.com/anshxpress/autoEDA.git
cd autoEDA
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Two Modes

### Mode 1 — Standard EDA
Profile any dataset (non-sensor), generate cleaning report, EDA, and baseline ML model.

```powershell
python -m autoeda_plus.cli.autoeda "data.csv"
python -m autoeda_plus.cli.autoeda "data.csv" --clean --no-plots
```

### Mode 2 — Anomaly Detection Pipeline ⚡ NEW
Full 9-section anomaly notebook with sensor feature engineering, multi-model training, and evaluation.

```powershell
# Supervised (labels exist)
python -m autoeda_plus.cli.autoeda "sensor.csv" --anomaly --target anomaly

# Unsupervised (no labels — IsolationForest + OneClassSVM)
python -m autoeda_plus.cli.autoeda "sensor.csv" --anomaly

# Multi-file (vertical concat)
python -m autoeda_plus.cli.autoeda --anomaly --files sensor1.csv sensor2.csv --target anomaly
```

---

## 📂 Output

Notebooks are auto-generated in `output/`:

```
output/
  EDA_<dataset>.ipynb          ← Standard EDA notebook
  Anomaly_<dataset>.ipynb      ← Anomaly Detection notebook (9 sections)
```

Open with:
```powershell
jupyter notebook output\Anomaly_sample_sensor.ipynb
# or
code output\Anomaly_sample_sensor.ipynb
```

---

## ⚙️ Full CLI Reference

### Standard EDA

```
python -m autoeda_plus.cli.autoeda <file> [options]

Options:
  --output, -o <path>    Custom output notebook path
  --no-clean             Skip data cleaning step
  --no-plots             Skip visualization cells
  --summary-only         Summary statistics only
```

### Anomaly Detection

```
python -m autoeda_plus.cli.autoeda [<file>] --anomaly [options]

Options:
  --target <col>         Target/label column (supervised mode)
  --files <f1> <f2> ...  Multiple files (vertical concat)
  --window <int>         Rolling window size (default: 5)
  --lag-steps <int>      Lag features per column (default: 2)
  --contamination <f>    Expected anomaly fraction (default: 0.08)
  --no-tune              Skip GridSearchCV (faster)
  --no-clean             Skip cleaning
  --no-plots             Skip visualization cells
  --output, -o <path>    Custom output notebook path
```

---

## 🧪 Example Commands

```powershell
# Generate demo sensor datasets
python generate_sensor_data.py

# Supervised anomaly detection
python -m autoeda_plus.cli.autoeda "sample_sensor.csv" --anomaly --target anomaly

# Supervised with GridSearch tuning
python -m autoeda_plus.cli.autoeda "sample_sensor.csv" --anomaly --target anomaly --window 7 --lag-steps 3

# Unsupervised
python -m autoeda_plus.cli.autoeda "sample_sensor.csv" --anomaly

# Multi-file supervised
python -m autoeda_plus.cli.autoeda --anomaly --files sample_sensor.csv sample_sensor2.csv --target anomaly --no-tune

# Skip tuning for speed
python -m autoeda_plus.cli.autoeda "sample_sensor.csv" --anomaly --target anomaly --no-tune
```

---

## 🔥 Features

### Standard EDA
- Automatic dataset profiling
- Missing value & duplicate detection
- Statistical summaries (numerical + categorical)
- Visualization system (histograms, boxplots, heatmaps)
- Correlation analysis & outlier detection (IQR + Z-score)
- Feature engineering suggestions
- Baseline ML models (RF classifier/regressor)
- Structured Jupyter notebook generation

### Anomaly Detection ⚡ NEW
- **Multi-file loading** — merge multiple CSVs vertically with `source_file` tracking
- **Sensor Feature Engineering** — rolling mean/std, lag features (t-1, t-2), abs change, rate-of-change, z-score
- **8 ML Models** — LogReg, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **GridSearchCV** hyperparameter tuning for top models
- **Unsupervised fallback** — IsolationForest + OneClassSVM with consensus voting
- **Rigorous Evaluation** — confusion matrix, ROC curve, classification report, 5-fold cross-validation
- **9-Section Structured Notebook** — problem statement through insights & conclusion

---

## 📌 Project Structure

```
autoEDA/
│
├── autoeda_plus/
│   ├── cli/
│   │   └── autoeda.py              ← CLI entrypoint (--anomaly, --target, --files)
│   ├── core/
│   │   ├── data_loader.py          ← Single & multi-file loader
│   │   ├── eda_pipeline.py         ← run_pipeline() + run_anomaly_pipeline()
│   │   ├── data_profiler.py
│   │   └── schema_detector.py
│   ├── cleaning/
│   │   └── data_cleaner.py         ← 10-step robust cleaning
│   ├── analysis/
│   │   ├── feature_engineering.py  ← Sensor rolling/lag/zscore features
│   │   ├── statistics_engine.py
│   │   ├── correlation_engine.py
│   │   └── outlier_detector.py
│   ├── ml/
│   │   ├── model_trainer.py        ← Supervised: LR/SVM/KNN/DT/RF/GB/XGB/LGBM
│   │   ├── anomaly_detector.py     ← Unsupervised: IsolationForest + OneClassSVM
│   │   ├── model_evaluator.py      ← CV, confusion matrix, comparison table
│   │   └── baseline_models.py
│   ├── notebook/
│   │   ├── anomaly_notebook_builder.py  ← 9-section anomaly notebook
│   │   └── notebook_builder.py          ← Standard EDA notebook
│   ├── insights/
│   │   └── insight_generator.py
│   └── visualization/
│
├── output/                         ← Generated notebooks
├── sample_sensor.csv               ← Demo sensor dataset (500 rows, 40 anomalies)
├── sample_sensor2.csv              ← Demo sensor dataset 2 (300 rows, 20 anomalies)
├── generate_sensor_data.py         ← Script to regenerate demo data
├── sample.csv
└── requirements.txt
```

---

## 🧠 Requirements

- Python 3.9+
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- nbformat
- xgboost *(optional but recommended)*
- lightgbm *(optional but recommended)*

---

## 📊 Anomaly Notebook — 9 Sections

| # | Section | Content |
|---|---------|---------|
| 1 | 📋 Problem Statement | Mode (supervised/unsupervised), dataset card, strategy |
| 2 | 📂 Data Overview | Shape, types, missing values, class distribution |
| 3 | 🧹 Data Cleaning | Embedded 10-step clean_data(), cleaning report table |
| 4 | 📊 EDA | Distributions, boxplots, correlation heatmap, violin by anomaly |
| 5 | ⚙️ Feature Engineering | Rolling, lag, change, z-score feature creation |
| 6 | 🤖 Model Training | All models trained + GridSearchCV tuning |
| 7 | 📈 Comparison | Ranked metrics table + grouped bar chart |
| 8 | 🎯 Evaluation | Confusion matrix, ROC curve, CV, feature importance |
| 9 | 💡 Insights | Findings, recommendations, next steps |

---

## ⚠️ Notes

- Run all commands from inside `d:\autoeda\autoEDA`
- Use full file paths to avoid path issues
- Works fully **offline** (no internet required after install)

---

## 🚀 Future Enhancements

- SHAP explainable AI integration
- Real-time streaming anomaly detection
- MCP-based interactive analysis
- Automated feature selection (RFECV / Boruta)
- Multi-sensor correlation anomaly detection
