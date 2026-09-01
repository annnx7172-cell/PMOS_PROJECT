"""Shared helpers: path resolution, pickle IO and the risk banding rules.

Paths are resolved from this file's location rather than the current working
directory, so ``streamlit run app.py``, ``python -m src.pipeline.train_pipeline``
and a notebook all resolve ``artifacts/`` to the same folder.
"""

import os
import pickle
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.exception import CustomException

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebook')


def artifact_path(*parts: str) -> str:
    """Absolute path to a file inside ``artifacts/``."""
    return os.path.join(ARTIFACTS_DIR, *parts)


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


# Risk banding — the single source of truth. predict_pipeline, the diagnosis
# page and the risk dashboard all read these instead of repeating the numbers.
RISK_HIGH = 0.66
RISK_MODERATE = 0.33

COLOR_HIGH = '#c0392b'
COLOR_MODERATE = '#d68910'
COLOR_LOW = '#1e8449'


def risk_band(prob: float):
    """Map a probability to its ``(label, colour)`` pair."""
    if prob > RISK_HIGH:
        return 'High', COLOR_HIGH
    if prob > RISK_MODERATE:
        return 'Moderate', COLOR_MODERATE
    return 'Low', COLOR_LOW


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Score a fitted model on the held-out set plus 5-fold CV ROC-AUC.

    Mirrors the ``evaluate_model`` helper in Block 3 so component runs and
    notebook runs produce comparable numbers.
    """
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

        return {
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
            'PR-AUC': round(average_precision_score(y_test, y_prob), 4),
            'CV-AUC': round(float(np.mean(cv_scores)), 4),
            'CV-STD': round(float(np.std(cv_scores)), 4),
        }
    except Exception as e:
        raise CustomException(e, sys)
