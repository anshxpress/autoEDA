"""
AutoEDA++ — Robust Multi-Model Trainer & Persistence Module
===========================================================
Trains a suite of classifiers (LR, SVM, KNN, DT, RF, GB, XGB, LGBM, CatBoost)
with automated imbalance handling (SMOTE/Weights) and full model persistence.
"""
import os
import warnings
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from autoeda_plus.analysis.feature_engineering import extract_datetime_features

warnings.filterwarnings("ignore")
logger = logging.getLogger("AutoEDA++")


# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Robust Feature Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class FeaturePipeline:
    """Manages feature transformation persistence and re-injection of unknown categories."""
    
    def __init__(self, target_col: str, drop_cols: Optional[List[str]] = None, scale: bool = True):
        self.target_col = target_col
        self.drop_cols = drop_cols or []
        self.scale = scale
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self.dt_cols: List[str] = []
        self.medians: Dict[str, float] = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fits transformers on training data and returns transformed X, y."""
        df = df.copy()
        
        # 0. Datetime Feature Extraction (Kaggle Section 4)
        df = extract_datetime_features(df)
        
        y = df[self.target_col].copy()
        
        # 1. Column Filtering
        drop = [self.target_col] + self.drop_cols
        # Detect and preserve any columns that are datetime or derived features
        self.dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        drop += self.dt_cols
        
        X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        self.feature_cols = X.columns.tolist()

        # 2. Categorical Encoding
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # 3. Robust Imputation
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isnull().any():
                val = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0
                self.medians[col] = val
                X[col] = X[col].fillna(val)

        # 4. Feature Scaling
        X_out = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index) if self.scale else X
        self.is_fitted = True
        return X_out, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform()")
        
        df = df.copy()
        # 0. Consistent Datetime Extraction
        df = extract_datetime_features(df)
        
        # Ensure exact columns
        X = df.reindex(columns=self.feature_cols)

        # 1. Categorical Encoding (with unseen category handling)
        for col, le in self.label_encoders.items():
            if col in X.columns:
                classes = set(le.classes_)
                X[col] = X[col].astype(str).apply(lambda val: val if val in classes else le.classes_[0])
                X[col] = le.transform(X[col])

        # 2. Imputation
        X = X.replace([np.inf, -np.inf], np.nan)
        for col, val in self.medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        X = X.fillna(0)

        # 3. Scaling
        X_out = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index) if self.scale else X
        return X_out

    def save(self, path: str) -> None:
        """Serialize the pipeline to disk."""
        joblib.dump(self, path)
        logger.info(f"✅ Pipeline persistent at: {path}")

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        """Load a pipeline from disk."""
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry & Imbalance Handling
# ─────────────────────────────────────────────────────────────────────────────
def _get_registry(pos_weight: float = 1.0, random_state: int = 42) -> Dict[str, Any]:
    """Returns the suite of models configured for anomaly detection."""
    registry = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=random_state, class_weight="balanced"),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=random_state, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=random_state),
    }
    
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=150, scale_pos_weight=pos_weight, use_label_encoder=False,
            eval_metric="logloss", random_state=random_state, verbosity=0
        )
    if LIGHTGBM_AVAILABLE:
        registry["LightGBM"] = LGBMClassifier(
            n_estimators=150, random_state=random_state, class_weight="balanced", verbose=-1
        )
    if CATBOOST_AVAILABLE:
        registry["CatBoost"] = CatBoostClassifier(
            iterations=150, random_state=random_state, auto_class_weights="Balanced", verbose=False
        )
        
    return registry


_PARAM_GRIDS: Dict[str, Dict] = {
    "Random Forest":     {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "Gradient Boosting": {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1]},
    "XGBoost":           {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1]},
    "LightGBM":          {"n_estimators": [100, 150], "num_leaves": [31, 63]},
    "CatBoost":          {"iterations": [100, 150], "learning_rate": [0.05, 0.1]},
}


def run_supervised_training(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    tune_top_models: bool = True,
    use_smote: bool = False,
    drop_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fits multiple models, handles imbalance via SMOTE/Weights, and returns comparison matrix.
    """
    pipeline = FeaturePipeline(target_col=target_col, drop_cols=drop_cols)
    X, y = pipeline.fit_transform(df)

    # Stratified Split (Crucial for Imbalanced Anomaly Data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale positive weight for XGB/CatBoost
    counts = y_train.value_counts()
    pos_weight = counts.iloc[0] / counts.iloc[1] if (len(counts) > 1 and counts.iloc[1] > 0) else 1.0

    # Fault-Tolerant SMOTE
    if use_smote and SMOTE_AVAILABLE:
        minority_size = int(y_train.sum())
        if minority_size > 1:
            k = min(5, minority_size - 1)
            if verbose: print(f"[ML] 🧬 SMOTE (k={k}) active on {minority_size} anomaly samples.")
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            if verbose: print("[ML] ⚠️ SMOTE skipped: Not enough anomaly samples.")

    registry = _get_registry(pos_weight=pos_weight)
    results = {}
    avg = "binary" if y_train.nunique() == 2 else "weighted"

    if verbose:
        print(f"\n[ML] 🤖 Training {len(registry)} models (Target: '{target_col}')")
        print(f"     ↳ Train size: {len(X_train)} | Test size: {len(X_test)}")

    for name, model in registry.items():
        try:
            if tune_top_models and name in _PARAM_GRIDS:
                gs = GridSearchCV(model, _PARAM_GRIDS[name], cv=3, scoring="f1_weighted", n_jobs=-1)
                gs.fit(X_train, y_train)
                best, params = gs.best_estimator_, gs.best_params_
            else:
                model.fit(X_train, y_train)
                best, params = model, {}

            y_pred = best.predict(X_test)
            metrics = {
                "accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
                "f1":        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
            }
            
            fi = dict(zip(X_train.columns, best.feature_importances_.tolist())) if hasattr(best, "feature_importances_") else {}
            
            results[name] = {"model": best, "metrics": metrics, "params": params, "fi": fi, "y_pred": y_pred}
            if verbose: print(f"     ✅ {name:<25} F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}")
        except Exception as e:
            if verbose: print(f"     ❌ {name:<25} FAILED - {e}")

    comp_rows = [{"model": n, **r["metrics"]} for n, r in results.items() if "metrics" in r]
    comparison_df = pd.DataFrame(comp_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_name = comparison_df.iloc[0]["model"] if not comparison_df.empty else None

    return {
        "pipeline": pipeline, "X_train": X_train, "X_test": X_test, "y_train": y_train, 
        "y_test": y_test, "results": results, "comparison_df": comparison_df, 
        "best_model_name": best_name
    }
