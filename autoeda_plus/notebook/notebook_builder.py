"""
AutoEDA++ — Notebook Builder
=============================
Generates a Jupyter notebook that mirrors the 4-step pipeline:
  Section 1 — Data Loading
  Section 2 — Data Cleaning  (with embedded clean_data function + report table)
  Section 3 — EDA & Visualization
  Section 4 — Insights, Feature Engineering Suggestions & ML Baseline
"""
import os
import textwrap
from typing import Dict, Any, List, Optional

import pandas as pd
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from autoeda_plus.core.data_loader import load_data, validate_dataset
from autoeda_plus.core.schema_detector import detect_column_types, detect_potential_target, get_column_summary
from autoeda_plus.core.data_profiler import profile_dataset, detect_data_quality_issues
from autoeda_plus.analysis.statistics_engine import compute_numerical_statistics, compute_categorical_statistics
from autoeda_plus.analysis.correlation_engine import compute_correlation_matrix, detect_strong_correlations
from autoeda_plus.analysis.outlier_detector import get_outlier_summary
from autoeda_plus.visualization.plot_engine import (
    generate_histogram, generate_boxplot, generate_countplot,
    generate_correlation_heatmap, generate_scatterplot, generate_missing_values_plot,
)
from autoeda_plus.ml.baseline_models import train_baseline_model
from autoeda_plus.insights.insight_generator import generate_insights, generate_feature_engineering_suggestions


# ──────────────────────────────────────────────────────────────────────────────
# Helper — code cell with nice leading newline stripped
# ──────────────────────────────────────────────────────────────────────────────
def _code(src: str) -> nbf.NotebookNode:
    return new_code_cell(textwrap.dedent(src).strip())


def _md(src: str) -> nbf.NotebookNode:
    return new_markdown_cell(textwrap.dedent(src).strip())


# ──────────────────────────────────────────────────────────────────────────────
# Main builder — accepts pre-computed results from the pipeline
# ──────────────────────────────────────────────────────────────────────────────
def build_comprehensive_eda_notebook(
    csv_path: str,
    output_path: str = "EDA_report.ipynb",
    *,
    # Pre-computed from pipeline (optional — will re-compute if not supplied)
    cleaned_df: Optional[pd.DataFrame] = None,
    cleaning_log: Optional[List[str]] = None,
    step_results: Optional[List[Any]] = None,
    column_types: Optional[Dict[str, str]] = None,
    num_stats: Optional[Dict] = None,
    cat_stats: Optional[Dict] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
    outlier_summary: Optional[Dict] = None,
    quality_issues: Optional[List[str]] = None,
    insights: Optional[List[str]] = None,
    fe_suggestions: Optional[List[str]] = None,
    ml_results: Optional[Dict] = None,
    target_col: Optional[str] = None,
    no_plots: bool = False,
    summary_only: bool = False,
    did_clean: bool = True,
    # Legacy kwargs accepted for backwards compatibility
    **kwargs,
) -> None:
    """
    Build a comprehensive EDA notebook.

    If pre-computed results are supplied (from run_pipeline), they are embedded
    directly. Otherwise the data is loaded and analysed from scratch.
    """
    csv_path_fwd = csv_path.replace("\\", "/")
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    # ── Load / validate data ──────────────────────────────────────────────────
    if cleaned_df is None:
        raw_df = load_data(csv_path)
        is_valid, msg = validate_dataset(raw_df)
        if not is_valid:
            raise ValueError(msg)
        cleaned_df = raw_df

    if column_types is None:
        column_types = detect_column_types(cleaned_df)
    if num_stats is None:
        num_stats = compute_numerical_statistics(cleaned_df, column_types)
    if cat_stats is None:
        cat_stats = compute_categorical_statistics(cleaned_df, column_types)
    if corr_matrix is None:
        corr_matrix = compute_correlation_matrix(cleaned_df, column_types)
    if outlier_summary is None:
        outlier_summary = get_outlier_summary(cleaned_df, column_types)
    if quality_issues is None:
        quality_issues = detect_data_quality_issues(cleaned_df, column_types)
    if target_col is None:
        target_col = detect_potential_target(cleaned_df, column_types)
    if ml_results is None:
        ml_results = train_baseline_model(cleaned_df, target_col, column_types)
    if insights is None:
        insights = generate_insights(cleaned_df, column_types, corr_matrix, outlier_summary)
    if fe_suggestions is None:
        fe_suggestions = generate_feature_engineering_suggestions(column_types, num_stats)

    cleaning_log = cleaning_log or []
    step_results = step_results or []

    num_cols = [c for c, t in column_types.items() if t == "numerical"]
    cat_cols = [c for c, t in column_types.items() if t == "categorical"]
    dt_cols  = [c for c, t in column_types.items() if t == "datetime"]

    nb = new_notebook()

    # ══════════════════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════════════════
    nb.cells.append(_md(f"""
        # 🔍 AutoEDA++ — Automated Exploratory Data Analysis
        ### Dataset: `{dataset_name}`

        > This notebook was **automatically generated** by AutoEDA++.
        > It follows the pipeline: **Load → Clean → EDA → Insights**.

        ---
        """))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — DATA LOADING
    # ══════════════════════════════════════════════════════════════════════════
    nb.cells.append(_md("""
        ## 📂 Section 1 — Data Loading

        Load the dataset with automatic encoding detection and format support.
        """))

    nb.cells.append(_code(f"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        warnings.filterwarnings('ignore')

        # ── Load dataset ──────────────────────────────────────────────────────
        DATASET_PATH = '{csv_path_fwd}'

        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(DATASET_PATH, encoding=enc, low_memory=False)
                print(f'✅ Loaded with encoding: {{enc}}')
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if df is None:
            df = pd.read_csv(DATASET_PATH, low_memory=False)
            print('✅ Loaded with default encoding')

        print(f'Shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns')
        df.head()
        """))

    nb.cells.append(_code("""
        # Dataset info
        print("── Data Types ──────────────────────────────────────────────────")
        print(df.dtypes.to_string())
        print(f"\\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        """))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — DATA CLEANING
    # ══════════════════════════════════════════════════════════════════════════
    nb.cells.append(_md("""
        ---

        ## 🧹 Section 2 — Data Cleaning

        The robust 10-step cleaning pipeline is applied to the raw dataset.
        Each step is tracked and reported below.
        """))

    # Embed the full clean_data() function so the notebook is self-contained
    nb.cells.append(_code(r"""
        # ── Embedded clean_data() — 10-step robust cleaning pipeline ────────
        from dataclasses import dataclass, field as dc_field
        from typing import Tuple, List

        @dataclass
        class StepResult:
            step_number: int
            step_name: str
            executed: bool = True
            skipped: bool = False
            skip_reason: str = ""
            messages: List[str] = dc_field(default_factory=list)
            nothing_to_do: bool = False

            def add(self, msg):
                self.messages.append(msg)
            def mark_skipped(self, reason):
                self.skipped = True
                self.executed = False
                self.skip_reason = reason
            def mark_nothing_to_do(self):
                self.nothing_to_do = True
                self.messages.append("✅ No action needed")


        def clean_data(df, report=True):
            import numpy as np
            import pandas as pd

            log, steps = [], []
            df = df.copy()

            # Step 1 — Normalize column names
            s1 = StepResult(1, "Normalize Column Names")
            original = list(df.columns)
            df.columns = df.columns.str.lower().str.strip().str.replace(r'\s+', '_', regex=True)
            df = df.loc[:, ~df.columns.duplicated()]
            renamed = [f"{o} → {n}" for o, n in zip(original, df.columns) if o != n]
            [s1.add(r) for r in renamed] if renamed else s1.mark_nothing_to_do()
            log.extend(renamed); steps.append(s1)

            # Step 2 — Remove duplicates
            s2 = StepResult(2, "Remove Duplicate Rows")
            before = len(df); df = df.drop_duplicates(); removed = before - len(df)
            s2.add(f"Removed {removed} duplicate rows") if removed else s2.mark_nothing_to_do()
            if removed: log.append(s2.messages[0])
            steps.append(s2)

            # Step 3 — Handle missing values
            s3 = StepResult(3, "Handle Missing Values")
            if df.isnull().any().any():
                for col in list(df.columns):
                    if col not in df.columns: continue
                    mr = df[col].isnull().mean()
                    if mr > 0.4:
                        df.drop(columns=[col], inplace=True)
                        m = f"{col}: dropped (>{int(mr*100)}% missing)"
                        s3.add(m); log.append(m); continue
                    if mr > 0:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            med = df[col].median(); df[col] = df[col].fillna(med)
                            m = f"{col}: filled with median ({med:.4g})"
                        else:
                            mode = df[col].mode()
                            if not mode.empty:
                                df[col] = df[col].fillna(mode[0])
                                m = f"{col}: filled with mode ('{mode[0]}')"
                            else: continue
                        s3.add(m); log.append(m)
                if not s3.messages: s3.mark_nothing_to_do()
            else: s3.mark_nothing_to_do()
            steps.append(s3)

            # Step 4 — Fix data types
            s4 = StepResult(4, "Fix Data Types")
            obj_cols = df.select_dtypes(include="object").columns.tolist()
            if obj_cols:
                for col in obj_cols:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="raise")
                        s4.add(f"{col}: → numeric"); log.append(s4.messages[-1]); continue
                    except: pass
                    try:
                        converted = pd.to_datetime(df[col], errors="coerce")
                        if converted.notna().mean() >= 0.5:
                            df[col] = converted
                            s4.add(f"{col}: → datetime"); log.append(s4.messages[-1])
                    except: pass
                if not s4.messages: s4.mark_nothing_to_do()
            else: s4.mark_skipped("No object columns to convert")
            steps.append(s4)

            # Step 5 — Clean text columns
            s5 = StepResult(5, "Clean Text Columns")
            txt = df.select_dtypes(include="object").columns.tolist()
            if txt:
                for col in txt:
                    df[col] = df[col].astype(str).str.lower().str.strip()
                    df[col] = df[col].replace({"?": np.nan, "nan": np.nan, "": np.nan})
                s5.add(f"Cleaned {len(txt)} text column(s): {txt}"); log.append(s5.messages[0])
            else: s5.mark_skipped("No text columns found")
            steps.append(s5)

            # Step 6 — Infinite values
            s6 = StepResult(6, "Handle Infinite Values")
            nc = df.select_dtypes(include=[np.number]).columns.tolist()
            if nc:
                inf_cols = [c for c in nc if np.isinf(df[c]).any()]
                if inf_cols:
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    s6.add(f"Replaced ±inf in {inf_cols}"); log.append(s6.messages[0])
                else: s6.mark_nothing_to_do()
            else: s6.mark_skipped("No numeric columns")
            steps.append(s6)

            # Step 7 — Constant columns
            s7 = StepResult(7, "Remove Constant Columns")
            cc = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
            if cc:
                df.drop(columns=cc, inplace=True)
                s7.add(f"Dropped {cc}"); log.append(s7.messages[0])
            else: s7.mark_nothing_to_do()
            steps.append(s7)

            # Step 8 — Identifier columns
            s8 = StepResult(8, "Remove Identifier Columns")
            id_cols = [c for c in df.columns if df[c].nunique(dropna=False) == len(df)]
            if id_cols:
                df.drop(columns=id_cols, inplace=True)
                s8.add(f"Dropped {id_cols}"); log.append(s8.messages[0])
            else: s8.mark_nothing_to_do()
            steps.append(s8)

            # Step 9 — IQR outlier removal
            s9 = StepResult(9, "Outlier Removal (IQR)")
            nm = df.select_dtypes(include=[np.number]).columns.tolist()
            if nm:
                total = 0
                for col in nm:
                    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR == 0: continue
                    b = len(df)
                    df = df[(df[col] >= Q1-1.5*IQR) & (df[col] <= Q3+1.5*IQR)]
                    r = b - len(df)
                    if r:
                        s9.add(f"{col}: removed {r} outlier(s)"); log.append(s9.messages[-1]); total += r
                if not total: s9.mark_nothing_to_do()
            else: s9.mark_skipped("No numeric columns for IQR")
            steps.append(s9)

            # Step 10 — Downcast + reset index
            s10 = StepResult(10, "Downcast Numeric Types & Reset Index")
            dc_cols = []
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                try: df[col] = pd.to_numeric(df[col], downcast="float"); dc_cols.append(col)
                except: pass
            df = df.reset_index(drop=True)
            if dc_cols: s10.add(f"Downcast {len(dc_cols)} col(s); index reset"); log.append(s10.messages[0])
            else: s10.mark_nothing_to_do()
            steps.append(s10)

            if report:
                print("\n" + "="*60)
                print("   🧹  DATA CLEANING REPORT")
                print("="*60)
                for s in steps:
                    icon = "⏭ SKIP" if s.skipped else ("✅ DONE" if s.nothing_to_do else "🔧 FIX ")
                    print(f"\n  Step {s.step_number:02d} | {icon} | {s.step_name}")
                    if s.skipped: print(f"           ↳ {s.skip_reason}")
                    elif s.nothing_to_do: print(f"           ↳ Nothing to clean")
                    else: [print(f"           ↳ {m}") for m in s.messages]
                print(f"\n  📊 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print("="*60)

            return df, log, steps

        # ── Run cleaning ──────────────────────────────────────────────────────
        df_clean, cleaning_log, step_results = clean_data(df, report=True)
        print(f"\n✅ Cleaning done. Shape: {df_clean.shape}")
        """))

    # Cleaning report markdown table (pre-computed from pipeline run)
    if step_results:
        table_rows = []
        for s in step_results:
            if s.skipped:
                status = "⏭ Skipped"
                action = s.skip_reason
            elif s.nothing_to_do:
                status = "✅ Nothing to clean"
                action = "—"
            else:
                status = "🔧 Fixed"
                action = "; ".join(s.messages)
            table_rows.append(f"| {s.step_number} | {s.step_name} | {status} | {action} |")

        table_md = (
            "### 📋 Cleaning Report Summary\n\n"
            "| # | Step | Status | Action Taken |\n"
            "|---|------|--------|-------------|\n"
            + "\n".join(table_rows)
        )
        nb.cells.append(_md(table_md))
    elif did_clean:
        nb.cells.append(_md("> ℹ️ Cleaning ran — see console output above for details."))
    else:
        nb.cells.append(_md("> ⏭ **Cleaning was skipped** (`--no-clean` flag was set)."))

    nb.cells.append(_code("""
        # Compare raw vs cleaned
        print(f"Raw      shape: {df.shape}")
        print(f"Cleaned  shape: {df_clean.shape}")
        print(f"\\nCleaned Dataset — First 5 rows:")
        df_clean.head()
        """))

    nb.cells.append(_code("""
        # Missing values after cleaning
        print("Missing values per column (cleaned dataset):")
        print(df_clean.isnull().sum().to_string())
        """))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — EDA & VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    nb.cells.append(_md("""
        ---

        ## 📊 Section 3 — EDA & Visualization

        Exploratory analysis of the cleaned dataset.
        """))

    # 3a — Descriptive stats
    nb.cells.append(_md("### 3.1 Descriptive Statistics"))
    nb.cells.append(_code("df_clean.describe(include='all')"))

    # 3b — Missing value heatmap
    nb.cells.append(_md("### 3.2 Missing Values Heatmap"))
    if not summary_only and not no_plots:
        nb.cells.append(_code(generate_missing_values_plot(cleaned_df)))

    # 3c — Quality issues
    if quality_issues:
        issues_md = "### 3.3 Data Quality Issues\n\n" + "\n".join(f"- {i}" for i in quality_issues)
        nb.cells.append(_md(issues_md))

    if not summary_only and not no_plots:
        # 3d — Numerical features
        if num_cols:
            nb.cells.append(_md("### 3.4 Numerical Feature Distributions"))
            for col in num_cols:
                nb.cells.append(_md(f"#### `{col}`"))
                nb.cells.append(_code(generate_histogram(cleaned_df, col)))
                nb.cells.append(_code(generate_boxplot(cleaned_df, col)))

        # 3e — Categorical features
        if cat_cols:
            nb.cells.append(_md("### 3.5 Categorical Feature Distributions"))
            for col in cat_cols:
                nb.cells.append(_md(f"#### `{col}`"))
                nb.cells.append(_code(generate_countplot(cleaned_df, col)))

        # 3f — Datetime features
        if dt_cols:
            nb.cells.append(_md("### 3.6 Datetime Feature Analysis"))
            for col in dt_cols:
                nb.cells.append(_code(
                    f"df_clean['{col}'] = pd.to_datetime(df_clean['{col}'], errors='coerce')\n"
                    f"df_clean['{col}'].dt.year.value_counts().sort_index().plot(\n"
                    f"    kind='bar', title='Year Distribution — {col}')\n"
                    f"plt.tight_layout(); plt.show()"
                ))

        # 3g — Correlation analysis
        if corr_matrix is not None and not corr_matrix.empty and len(num_cols) > 1:
            nb.cells.append(_md("### 3.7 Correlation Analysis"))
            nb.cells.append(_code(
                "corr_matrix = df_clean.select_dtypes(include='number').corr()\n"
                "corr_matrix"
            ))
            nb.cells.append(_code(generate_correlation_heatmap(corr_matrix)))

            strong_corrs = detect_strong_correlations(corr_matrix, 0.7)
            if strong_corrs:
                lines = [f"- **{c1}** ↔ **{c2}**: {r:.2f}" for c1, c2, r in strong_corrs]
                nb.cells.append(_md("#### Strong Correlations (|ρ| ≥ 0.7)\n\n" + "\n".join(lines)))

        # 3h — Pairplot for top numeric cols
        if len(num_cols) > 1:
            top_num = num_cols[:5]  # limit 5
            nb.cells.append(_md("### 3.8 Pairplot (top numeric columns)"))
            nb.cells.append(_code(
                f"import seaborn as sns\n"
                f"g = sns.pairplot(df_clean[{top_num}].dropna())\n"
                f"g.fig.suptitle('Pairplot', y=1.02)\n"
                f"plt.show()"
            ))

    # 3i — Outlier summary
    nb.cells.append(_md("### 3.9 Outlier Summary"))
    outlier_lines = []
    for col, counts in (outlier_summary or {}).items():
        iqr_n = counts.get("iqr_outliers", 0)
        z_n   = counts.get("zscore_outliers", 0)
        if iqr_n > 0 or z_n > 0:
            outlier_lines.append(f"- **{col}**: IQR={iqr_n} outlier(s), Z-score={z_n} outlier(s)")
    if outlier_lines:
        nb.cells.append(_md("\n".join(outlier_lines)))
    else:
        nb.cells.append(_md("✅ No significant outliers detected in the cleaned dataset."))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — INSIGHTS, FE SUGGESTIONS & ML BASELINE
    # ══════════════════════════════════════════════════════════════════════════
    nb.cells.append(_md("""
        ---

        ## 💡 Section 4 — Insights, Feature Engineering & Baseline ML
        """))

    # 4a — Insights
    nb.cells.append(_md("### 4.1 Key Insights"))
    if insights:
        nb.cells.append(_md("\n".join(f"- {i}" for i in insights)))
    else:
        nb.cells.append(_md("No specific insights generated."))

    # 4b — Feature engineering suggestions
    nb.cells.append(_md("### 4.2 Feature Engineering Suggestions"))
    if fe_suggestions:
        nb.cells.append(_md("\n".join(f"- 💡 {s}" for s in fe_suggestions)))
    else:
        nb.cells.append(_md("No specific suggestions at this time."))

    # 4c — ML Baseline
    if ml_results and ml_results.get("success"):
        nb.cells.append(_md("### 4.3 Baseline Machine Learning Model"))
        problem_type = ml_results.get("problem_type", "unknown")
        model_name   = ml_results.get("model", "N/A")
        metrics      = ml_results.get("metrics", {})
        fi           = ml_results.get("feature_importance", {})

        metrics_md = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
        nb.cells.append(_md(
            f"| Property | Value |\n|---|---|\n"
            f"| Problem Type | {problem_type.title()} |\n"
            f"| Model | {model_name} |\n"
            f"| Target Column | `{target_col}` |\n\n"
            f"**Metrics**\n\n| Metric | Score |\n|---|---|\n{metrics_md}"
        ))

        if fi:
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
            fi_md = "\n".join(f"| `{col}` | {imp:.4f} |" for col, imp in fi_sorted)
            nb.cells.append(_md(
                f"**Top Feature Importances**\n\n| Feature | Importance |\n|---|---|\n{fi_md}"
            ))

    # Footer
    nb.cells.append(_md(
        "---\n"
        "> 🤖 *This notebook was automatically generated by **AutoEDA++**.*  \n"
        "> Pipeline: Data Loading → Data Cleaning → EDA → Insights"
    ))

    # ── Write notebook ────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)