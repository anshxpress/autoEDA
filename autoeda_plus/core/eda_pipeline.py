"""
AutoEDA++ — EDA Pipeline Orchestrator
======================================
Single entrypoint for the full 4-step pipeline:
  Step 1 — Data Loading
  Step 2 — Data Cleaning
  Step 3 — EDA & Visualization analysis
  Step 4 — Notebook Generation
"""
import logging
import sys
import os
from pprint import pformat
from typing import Dict, Any, Optional, List

import pandas as pd
import joblib

from autoeda_plus.core.data_loader import load_data, validate_dataset, load_multiple_files
from autoeda_plus.core.schema_detector import detect_column_types, detect_potential_target
from autoeda_plus.cleaning.data_cleaner import clean_data, clean_dataset, InsufficientDataError
from autoeda_plus.core.data_profiler import profile_dataset, detect_data_quality_issues
from autoeda_plus.analysis.statistics_engine import (
    compute_numerical_statistics, compute_categorical_statistics,
)
from autoeda_plus.analysis.correlation_engine import compute_correlation_matrix
from autoeda_plus.analysis.outlier_detector import get_outlier_summary
from autoeda_plus.analysis.feature_engineering import (
    engineer_sensor_features, get_feature_engineering_summary,
)
from autoeda_plus.insights.insight_generator import (
    generate_insights, generate_feature_engineering_suggestions,
)
from autoeda_plus.ml.baseline_models import train_baseline_model
from autoeda_plus.ml.model_trainer import run_supervised_training
from autoeda_plus.ml.anomaly_detector import run_unsupervised_detection
from autoeda_plus.ml.model_evaluator import build_comparison_table, cross_validate_model
from autoeda_plus.notebook.notebook_builder import build_comprehensive_eda_notebook
from autoeda_plus.notebook.anomaly_notebook_builder import build_anomaly_notebook


# ── Configuration ─────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures the unified AutoEDA++ logger."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    
    _logger = logging.getLogger("AutoEDA++")
    if not _logger.handlers:
        _logger.addHandler(handler)
        _logger.setLevel(level)
    return _logger


logger = setup_logging()


def validate_submission(df: pd.DataFrame, df_test: pd.DataFrame, target_col: str) -> bool:
    """Ensures submission file matches test data structure perfectly."""
    if df.isnull().values.any():
        logger.warning("Submission contains NaNs!")
        return False
    if len(df) != len(df_test):
        logger.warning(f"Submission row count ({len(df)}) != Test count ({len(df_test)})")
        return False
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' missing from submission")
        return False
    return True




def run_pipeline(
    file_path: str,
    output_path: Optional[str] = None,
    *,
    clean: bool = True,          # Cleaning is ON by default
    no_plots: bool = False,
    summary_only: bool = False,
    cap_outliers: bool = False,
) -> Dict[str, Any]:
    """
    Run the full AutoEDA++ pipeline.

    Parameters
    ----------
    file_path   : Path to the input data file (CSV, Excel, JSON, Parquet).
    output_path : Where to write the notebook. Defaults to output/EDA_<name>.ipynb.
    clean       : Run the cleaning module (default True). Pass False to skip.
    no_plots    : Generate notebook without plots.
    summary_only: Only summary statistics, no plots.
    cap_outliers: Cap outliers instead of removing them (future use).

    Returns
    -------
    dict — full pipeline summary with profiling, analysis and ML results.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join("output", f"EDA_{base}.ipynb")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Load Data
    # ──────────────────────────────────────────────────────────────────────────
    df = load_data(file_path)
    is_valid, msg = validate_dataset(df)
    if not is_valid:
        raise ValueError(msg)
    print(f"[Step 1] ✅ Data Loaded  "
          f"({df.shape[0]:,} rows × {df.shape[1]} columns)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Data Cleaning
    # ──────────────────────────────────────────────────────────────────────────
    cleaning_log = []
    step_results = []
    cleaned_df = df

    if clean:
        print("[Step 2] 🧹 Running Data Cleaning...")
        try:
            cleaned_df, cleaning_log, step_results = clean_data(df, report=True)
            print(f"[Step 2] ✅ Data Cleaned  "
                  f"({cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns)\n")
        except InsufficientDataError as e:
            cleaning_log = [str(e)]
            print(f"[Step 2] ❌ Cleaning Error: {e}")
            print("         ↳ Proceeding with uncleaned data for EDA.\n")
            cleaned_df = df
    else:
        print("[Step 2] ⏭  Data Cleaning skipped (--no-clean flag)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — EDA & Analysis
    # ──────────────────────────────────────────────────────────────────────────
    print("[Step 3] 📊 Running EDA & Analysis...")

    column_types = detect_column_types(cleaned_df)
    profile = profile_dataset(cleaned_df)
    quality_issues = detect_data_quality_issues(cleaned_df, column_types)

    num_stats = compute_numerical_statistics(cleaned_df, column_types)
    cat_stats = compute_categorical_statistics(cleaned_df, column_types)

    corr_matrix = compute_correlation_matrix(cleaned_df, column_types)
    outlier_summary = get_outlier_summary(cleaned_df, column_types)

    target_col = detect_potential_target(cleaned_df, column_types)
    ml_results = train_baseline_model(cleaned_df, target_col, column_types)

    insights = generate_insights(cleaned_df, column_types, corr_matrix, outlier_summary)
    fe_suggestions = generate_feature_engineering_suggestions(column_types, num_stats)

    print(f"[Step 3] ✅ EDA Complete  "
          f"({len(insights)} insights | "
          f"{len(num_stats)} numeric cols | "
          f"{len(cat_stats)} categorical cols)\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Notebook Generation
    # ──────────────────────────────────────────────────────────────────────────
    print(f"[Step 4] 📓 Generating Notebook → {output_path}")

    build_comprehensive_eda_notebook(
        csv_path=file_path,
        output_path=output_path,
        cleaned_df=cleaned_df,
        cleaning_log=cleaning_log,
        step_results=step_results,
        column_types=column_types,
        num_stats=num_stats,
        cat_stats=cat_stats,
        corr_matrix=corr_matrix,
        outlier_summary=outlier_summary,
        quality_issues=quality_issues,
        insights=insights,
        fe_suggestions=fe_suggestions,
        ml_results=ml_results,
        target_col=target_col,
        no_plots=no_plots,
        summary_only=summary_only,
        did_clean=clean,
    )

    print(f"[Step 4] ✅ Notebook Generated: {output_path}\n")
    print("=" * 60)
    print("   🎉  AutoEDA++ pipeline complete!")
    print("=" * 60)

    return {
        "file_path": file_path,
        "output_path": output_path,
        "cleaning_log": cleaning_log,
        "step_results": step_results,
        "profile": profile,
        "quality_issues": quality_issues,
        "num_stats": num_stats,
        "cat_stats": cat_stats,
        "corr_matrix_shape": None if corr_matrix is None else corr_matrix.shape,
        "outlier_summary": outlier_summary,
        "target_col": target_col,
        "ml_results": ml_results,
        "insights": insights,
        "fe_suggestions": fe_suggestions,
    }


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="AutoEDA++ — Run the full EDA pipeline on a data file"
    )
    parser.add_argument("file", help="Path to the data file (CSV, Excel, JSON, Parquet)")
    parser.add_argument("--output", "-o", default=None, help="Output notebook path")
    parser.add_argument("--no-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument("--no-plots", action="store_true", help="No visualizations")
    parser.add_argument("--summary-only", action="store_true", help="Summary stats only")

    args = parser.parse_args()
    run_pipeline(
        args.file,
        output_path=args.output,
        clean=not args.no_clean,
        no_plots=args.no_plots,
        summary_only=args.summary_only,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly Detection Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_anomaly_pipeline(
    file_paths: List[str],
    output_path: Optional[str] = None,
    *,
    target_col: Optional[str] = None,
    test_file: Optional[str] = None,
    clean: bool = True,
    no_plots: bool = False,
    window: int = 5,
    lag_steps: int = 2,
    contamination: float = 0.08,
    tune_models: bool = True,
    use_smote: bool = False,
) -> Dict[str, Any]:
    """
    Run the full AutoEDA++ Anomaly Detection pipeline with professional robustness.
    """
    if not file_paths:
        raise ValueError("At least one file path is required")

    # Output path setup
    if output_path is None:
        base = os.path.splitext(os.path.basename(file_paths[0]))[0]
        output_path = os.path.abspath(os.path.join("output", f"Anomaly_{base}.ipynb"))
    
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚨 AutoEDA++ — Advanced Anomaly Detection Pipeline")
    logger.info("=" * 60)

    # Step 1: Load
    logger.info("Step 1: Loading datasets...")
    df = load_data(file_paths[0]) if len(file_paths) == 1 else load_multiple_files(file_paths)
    
    is_valid, msg = validate_dataset(df)
    if not is_valid: raise ValueError(msg)
    logger.info(f"✅ Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Step 2: Validate mode
    mode = "supervised" if (target_col and target_col in df.columns) else "unsupervised"
    if mode == "supervised":
        logger.info(f"Step 2: Supervised Mode confirmed (Target: '{target_col}')")
    else:
        logger.info("Step 2: Unsupervised Mode active (IsolationForest + OneClassSVM)")
        target_col = None

    # Step 3: Clean
    cleaning_log, step_results_clean, cleaned_df = [], [], df
    if clean:
        logger.info("Step 3: Executing 10-step cleaning pipeline...")
        try:
            cleaned_df, cleaning_log, step_results_clean = clean_data(df, report=False)
            logger.info(f"✅ Cleaning complete ({len(cleaned_df):,} rows remaining)")
        except InsufficientDataError as e:
            logger.warning(f"⚠️ {e} — Proceeding with raw data.")
    else:
        logger.info("Step 3: Cleaning skipped")

    # Step 4: EDA
    logger.info("Step 4: Performing deep exploratory analysis...")
    column_types    = detect_column_types(cleaned_df)
    corr_matrix     = compute_correlation_matrix(cleaned_df, column_types)
    outlier_summary = get_outlier_summary(cleaned_df, column_types)
    quality_issues  = detect_data_quality_issues(cleaned_df, column_types)
    insights        = generate_insights(cleaned_df, column_types, corr_matrix, outlier_summary)
    logger.info(f"✅ EDA complete: {len(insights)} insights generated")

    # Step 5: Feature Engineering
    logger.info("Step 5: Engineering sensor-specific features (Rolling/Lag)...")
    original_cols = list(cleaned_df.columns)
    engineered_df = engineer_sensor_features(cleaned_df, window=window, lag_steps=lag_steps, verbose=False)
    fe_summary    = get_feature_engineering_summary(original_cols, engineered_df)
    logger.info(f"✅ Feature Engineering complete: {len(engineered_df.columns)} final features")

    # Step 6: ML Training
    supervised_results, unsupervised_results, comparison_df, best_model_name, cv_results = None, None, None, None, None

    if mode == "supervised":
        logger.info("Step 6: Training and Cross-validating supervised models...")
        supervised_results = run_supervised_training(
            engineered_df, target_col=target_col, tune_top_models=tune_models, 
            use_smote=use_smote, drop_cols=["source_file"], verbose=False
        )
        comparison_df   = supervised_results["comparison_df"]
        best_model_name = supervised_results["best_model_name"]
        
        # Artifact Persistence
        if best_model_name:
            logger.info(f"🏆 Best Model: {best_model_name}")
            pipeline_path = os.path.join(out_dir, "feature_pipeline.joblib")
            model_path    = os.path.join(out_dir, "best_model.joblib")
            
            supervised_results["pipeline"].save(pipeline_path)
            joblib.dump(supervised_results["results"][best_model_name]["model"], model_path)
            logger.info(f"💾 Model artifacts saved to {out_dir}")

            # Step 6b: Cross-validate best
            best_m = supervised_results["results"][best_model_name]["model"]
            cv_results = cross_validate_model(best_m, supervised_results["X_train"], supervised_results["y_train"], cv=5)
    else:
        logger.info("Step 6: Running unsupervised anomaly consensus detection...")
        unsupervised_results = run_unsupervised_detection(engineered_df, exclude_cols=["source_file"], contamination=contamination)

    # Step 7: Prediction & Submission Validation
    submission_path = None
    if test_file and mode == "supervised":
        logger.info(f"Step 7: Generating predictions for test file: {test_file}")
        try:
            test_df = load_data(test_file)
            test_eng = engineer_sensor_features(test_df, window=window, lag_steps=lag_steps, verbose=False)
            
            # Predict using persistence-safe pipeline
            X_test_final = supervised_results["pipeline"].transform(test_eng)
            preds = supervised_results["results"][best_model_name]["model"].predict(X_test_final)
            
            # Format and Validate per Kaggle Section 10
            id_col = "ID" if "ID" in test_df.columns else ("id" if "id" in test_df.columns else (test_df.index.name or "ID"))
            ids = test_df[id_col] if id_col in test_df.columns else test_df.index
            
            sub = pd.DataFrame({id_col: ids, target_col: preds})
            if validate_submission(sub, test_df, target_col):
                submission_path = os.path.join(out_dir, "submission.csv")
                sub.to_csv(submission_path, index=False)
                logger.info(f"✅ Verified submission saved: {submission_path}")
            else:
                logger.error("❌ Submission validation failed. File not saved.")
        except Exception as e:
            logger.error(f"❌ Prediction step failed: {e}")

    # Step 8: Build Sophisticated Notebook
    logger.info(f"Step 8: Building 10-section comprehensive notebook -> {os.path.basename(output_path)}")
    build_anomaly_notebook(
        file_paths=file_paths, output_path=output_path, cleaned_df=cleaned_df,
        engineered_df=engineered_df, cleaning_log=cleaning_log, step_results=step_results_clean,
        column_types=column_types, corr_matrix=corr_matrix, outlier_summary=outlier_summary,
        quality_issues=quality_issues, target_col=target_col, supervised_results=supervised_results,
        unsupervised_results=unsupervised_results, comparison_df=comparison_df,
        best_model_name=best_model_name, cv_results=cv_results, insights=insights, 
        fe_summary=fe_summary, source_files=file_paths, mode=mode, no_plots=no_plots,
        test_file=test_file, submission_path=submission_path
    )

    logger.info("=" * 60)
    logger.info("🎉 Anomaly Detection Pipeline Successfully Completed!")
    logger.info("=" * 60)

    return {
        "output_path": output_path, "mode": mode, "best_model_name": best_model_name,
        "submission_path": submission_path, "comparison_df": comparison_df,
        "artifact_dir": out_dir
    }



