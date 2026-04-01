"""
AutoEDA++ — Professional Model Evaluator
========================================
Rigorous evaluation: cross-validation, confusion matrix, ROC-AUC, and 
high-fidelity visualizations for anomaly detection performance.
"""
import logging
import textwrap
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("AutoEDA++")

# Set global aesthetics for professional charts
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "sans-serif"


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full evaluation of a single fitted model on the test set.
    """
    y_pred = model.predict(X_test)
    avg    = "binary" if y_test.nunique() == 2 else "weighted"

    metrics: Dict[str, float] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
    }

    # ROC-AUC
    if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    if verbose:
        logger.info(f"📊 {model_name} Evaluation: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")
        logger.debug(f"Classification Report:\n{report}")

    return {
        "model_name":            model_name,
        "metrics":               metrics,
        "confusion_matrix":      cm,
        "classification_report": report,
        "y_pred":                y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Stratified K-Fold cross-validation with weighted scoring.
    """
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    cv_obj  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    try:
        scores = cross_validate(model, X, y, cv=cv_obj, scoring=scoring, n_jobs=-1)
        result = {
            "cv_folds":           cv,
            "cv_accuracy_mean":   round(float(np.mean(scores["test_accuracy"])), 4),
            "cv_accuracy_std":    round(float(np.std(scores["test_accuracy"])), 4),
            "cv_f1_mean":         round(float(np.mean(scores["test_f1_weighted"])), 4),
            "cv_f1_std":          round(float(np.std(scores["test_f1_weighted"])), 4),
            "cv_precision_mean":  round(float(np.mean(scores["test_precision_weighted"])), 4),
            "cv_recall_mean":     round(float(np.mean(scores["test_recall_weighted"])), 4),
        }
        logger.info(f"🔁 CV {model_name}: {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}")
        return result
    except Exception as e:
        logger.error(f"❌ CV Failed for {model_name}: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
def build_comparison_table(
    results: Dict[str, Dict[str, Any]],
    sort_by: str = "f1",
) -> pd.DataFrame:
    """Consolidates all model results into a ranked DataFrame."""
    rows = []
    for name, res in results.items():
        if "metrics" not in res: continue
        row = {"Model": name, **res["metrics"]}
        row["Tuning"] = "Optimized" if res.get("params") else "Baseline"
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        df.index += 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Technical Chart Code Generators (for Notebook Injection)
# ─────────────────────────────────────────────────────────────────────────────
def confusion_matrix_plot_code(model_var: str, X_test_var: str, y_test_var: str) -> str:
    return textwrap.dedent(f"""\
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        y_pred = {model_var}.predict({X_test_var})
        cm = confusion_matrix({y_test_var}, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='magma', cbar=False)
        plt.title('Performance Heatmap: Confusion Matrix', fontsize=14, pad=15)
        plt.xlabel('Predicted Anomaly', fontsize=12)
        plt.ylabel('Actual Anomaly', fontsize=12)
        plt.tight_layout()
        plt.show()
    """)


def roc_curve_plot_code(model_var: str, X_test_var: str, y_test_var: str) -> str:
    return textwrap.dedent(f"""\
        from sklearn.metrics import RocCurveDisplay
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            RocCurveDisplay.from_estimator({model_var}, {X_test_var}, {y_test_var}, ax=ax, color='#e74c3c')
            ax.plot([0, 1], [0, 1], color='#2c3e50', linestyle='--', alpha=0.5, label='Random Baseline')
            ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=15)
            ax.legend(frameon=True, facecolor='white')
            plt.tight_layout()
            plt.show()
        except:
            print("ROC visualization not supported for this model type.")
    """)


def model_comparison_bar_code(comparison_var: str = "comparison_df") -> str:
    return textwrap.dedent(f"""\
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        df_melt = {comparison_var}.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette='viridis')
        plt.title('Leaderboard: Model Comparison across Core Metrics', fontsize=15, pad=20)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=25, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.show()
    """)


def anomaly_score_distribution_code(scores_var: str = "scores") -> str:
    return textwrap.dedent(f"""\
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(9, 4))
        sns.kdeplot({scores_var}, fill=True, color='#3498db', alpha=0.4, linewidth=2)
        plt.axvline(0, color='#e74c3c', linestyle='--', label='Decision Threshold')
        plt.title('Distribution of Unsupervised Anomaly Scores', fontsize=14, pad=15)
        plt.xlabel('Detection Score (Lower = More Anomalous)')
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()
    """)
