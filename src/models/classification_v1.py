"""Supervised classifiers predicting whether a crime is violent.

Three classifiers trained on the same feature set: logistic regression, random
forest and SVM. Each is evaluated with accuracy, F1, precision, recall and
ROC-AUC on a held-out test split.

The feature set is intentionally simple & reproducible so the client can audit it:

* ``latitude`` / ``longitude``
* ``month_num`` / ``year`` / ``quarter`` (temporal)
* one-hot season
* ``grid_id`` frequency-encoded (high-cardinality spatial bin)

Target: ``is_violent`` (see :mod:`src.data.features`).
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config

logger = logging.getLogger(__name__)


@dataclass
class ClfResult:
    name: str
    model: object
    metrics: Dict[str, float]
    report: str


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). All numeric, no leakage from outcome_category."""
    freq = df["grid_id"].value_counts(normalize=True)
    X = pd.DataFrame({
        "latitude":   df["latitude"].to_numpy(),
        "longitude":  df["longitude"].to_numpy(),
        "month_num":  df["month_num"].to_numpy(),
        "year":       df["year"].to_numpy(),
        "quarter":    df["quarter"].to_numpy(),
        "grid_freq":  df["grid_id"].map(freq).to_numpy(),
    })
    for s in ("winter", "spring", "summer", "autumn"):
        X[f"season_{s}"] = (df["season"] == s).astype(int).to_numpy()
    y = df[config.CLASSIFIER.target].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _make_models() -> Dict[str, Pipeline]:
    rs = config.CLASSIFIER.random_state
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=rs)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=config.CLASSIFIER.rf_n_estimators,
                n_jobs=-1, random_state=rs)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=config.CLASSIFIER.svm_c, probability=True, random_state=rs)),
        ]),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
    }


def fit_all(df: pd.DataFrame) -> List[ClfResult]:
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.CLASSIFIER.test_size,
        random_state=config.CLASSIFIER.random_state,
        stratify=y,
    )

    results: List[ClfResult] = []
    for name, pipe in _make_models().items():
        logger.info("Fitting %s …", name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_pred, y_proba)
        report = classification_report(y_test, y_pred, digits=3)
        logger.info("%s → %s", name, {k: round(v, 3) for k, v in metrics.items()})
        results.append(ClfResult(name=name, model=pipe, metrics=metrics, report=report))
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save(results: List[ClfResult],
         path: Path = config.MODEL_DIR / "classifiers.pkl") -> None:
    rows = []
    models = {}
    for r in results:
        rows.append({"model": r.name, **r.metrics})
        models[r.name] = r.model
    pd.DataFrame(rows).to_csv(config.DATA_DIR / "classifier_metrics.csv", index=False)
    with open(path, "wb") as f:
        pickle.dump(models, f)
    logger.info("Saved classifiers → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> pd.DataFrame:
    results = fit_all(df)
    save(results)
    return pd.DataFrame([{"model": r.name, **r.metrics} for r in results])
