"""
AutoEDA++ — Unsupervised Anomaly Detector
==========================================
Runs IsolationForest and OneClassSVM when no target column is available.
Returns a DataFrame with anomaly scores and binary predictions.
"""
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
def _prepare_unsupervised_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for unsupervised detection.
    Returns (X_scaled, feature_cols).
    """
    exclude = set(exclude_cols or [])
    # Drop datetime, object, and excluded columns
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]", "object"]).columns.tolist()
    exclude.update(dt_cols)

    feat_cols = [c for c in df.columns if c not in exclude]
    X = df[feat_cols].copy()

    # Impute
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )
    return X_scaled, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.08,
    n_estimators: int = 200,
    exclude_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run IsolationForest anomaly detection.

    Parameters
    ----------
    df            : Feature DataFrame (no target column needed).
    contamination : Expected fraction of anomalies (default 8%).
    n_estimators  : Number of trees.

    Returns
    -------
    dict with keys: predictions (-1/1), anomaly_flags (0/1), scores, n_anomalies
    """
    X, feat_cols = _prepare_unsupervised_features(df, exclude_cols)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    raw_preds  = model.fit_predict(X)           # -1 = anomaly, 1 = normal
    scores     = model.decision_function(X)     # lower = more anomalous

    anomaly_flags = (raw_preds == -1).astype(int)
    n_anomalies   = int(anomaly_flags.sum())

    print(f"\n[AD] 🌲 IsolationForest → {n_anomalies} anomalies detected "
          f"({n_anomalies / len(df) * 100:.1f}% of dataset)")

    return {
        "model":         model,
        "model_name":    "Isolation Forest",
        "feature_cols":  feat_cols,
        "predictions":   raw_preds,       # raw sklearn -1/1
        "anomaly_flags": anomaly_flags,   # 0/1 binary
        "scores":        scores,
        "n_anomalies":   n_anomalies,
        "contamination": contamination,
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_one_class_svm(
    df: pd.DataFrame,
    nu: float = 0.08,
    kernel: str = "rbf",
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run OneClassSVM anomaly detection.

    Parameters
    ----------
    nu     : Upper bound on fraction of training errors / anomalies (default 8%).
    kernel : Kernel type (default 'rbf').
    """
    X, feat_cols = _prepare_unsupervised_features(df, exclude_cols)

    # OneClassSVM can be slow on large datasets — sample if needed
    if len(X) > 2000:
        X_fit = X.sample(n=2000, random_state=42)
    else:
        X_fit = X

    model = OneClassSVM(nu=nu, kernel=kernel)
    model.fit(X_fit)

    raw_preds     = model.predict(X)           # -1 or 1
    anomaly_flags = (raw_preds == -1).astype(int)
    scores        = model.decision_function(X) # signed distance to boundary
    n_anomalies   = int(anomaly_flags.sum())

    print(f"[AD] 🔵 OneClassSVM     → {n_anomalies} anomalies detected "
          f"({n_anomalies / len(df) * 100:.1f}% of dataset)")

    return {
        "model":         model,
        "model_name":    "One-Class SVM",
        "feature_cols":  feat_cols,
        "predictions":   raw_preds,
        "anomaly_flags": anomaly_flags,
        "scores":        scores,
        "n_anomalies":   n_anomalies,
        "nu":            nu,
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_unsupervised_detection(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    contamination: float = 0.08,
) -> Dict[str, Any]:
    """
    Run both IsolationForest and OneClassSVM and return combined results.

    Returns
    -------
    dict with keys: if_results, ocsvm_results, combined_df
    """
    print("\n[AD] 🔍 Running unsupervised anomaly detection...")

    if_results    = run_isolation_forest(df, contamination=contamination, exclude_cols=exclude_cols)
    ocsvm_results = run_one_class_svm(df, nu=contamination, exclude_cols=exclude_cols)

    # Consensus: anomaly if BOTH models agree
    consensus = (
        (if_results["anomaly_flags"] == 1) & (ocsvm_results["anomaly_flags"] == 1)
    ).astype(int)

    combined_df = df.copy()
    combined_df["if_anomaly"]        = np.asarray(if_results["anomaly_flags"])
    combined_df["ocsvm_anomaly"]     = np.asarray(ocsvm_results["anomaly_flags"])
    combined_df["consensus_anomaly"] = np.asarray(consensus)
    combined_df["if_score"]          = np.asarray(if_results["scores"])
    combined_df["ocsvm_score"]       = np.asarray(ocsvm_results["scores"])

    n_consensus = int(consensus.sum())
    print(f"[AD] 🤝 Consensus      → {n_consensus} anomalies (both models agree)")

    return {
        "if_results":    if_results,
        "ocsvm_results": ocsvm_results,
        "consensus_anomalies": n_consensus,
        "combined_df":   combined_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_unsupervised_with_labels(
    anomaly_flags: np.ndarray,
    true_labels: pd.Series,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    If ground truth labels exist, evaluate unsupervised predictions against them.
    Returns precision, recall, F1 and confusion matrix.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = true_labels.values
    y_pred = anomaly_flags

    metrics = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[AD] 📊 {model_name} vs ground truth: "
          f"P={metrics['precision']}  R={metrics['recall']}  F1={metrics['f1']}")
    return {"metrics": metrics, "confusion_matrix": cm}
