"""Supervised classifiers predicting whether a crime is violent.

Three classifiers are trained on the same feature set: logistic regression, random
forest, and SVM. Each is evaluated with accuracy, F1, precision, recall, and
ROC-AUC on a held-out test split.

Design notes (why the features look the way they do)
----------------------------------------------------
An earlier draft of this module used ``grid_freq`` — the *total* crime density
per grid cell — as a spatial feature. On Sheffield data that proved to be the
wrong signal: retail districts produce enormous overall counts (shoplifting,
theft) but a *smaller* share of violent crime, while residential and nightlife
cells have lower total counts but a *higher* share. As a result ``grid_freq``
was anti-correlated with the target and the random forest landed on ROC-AUC
below 0.5.

The replacement is a **leakage-safe violent-rate prior**: for each grid cell
we compute ``P(is_violent | grid_id)`` using only the training rows, then map
those rates onto both train and test. Cells appearing only in the test split
fall back to the global training rate. The same pattern is applied at the LSOA
level when the column is present. This gives the model a proper class-aware
spatial signal without peeking at the test labels.

Other changes against the earlier draft
---------------------------------------
* ``class_weight='balanced'`` on all three models. With ~32% positives, the
  default weighting makes LogReg and SVM collapse to the majority class.
* The SVM is fit on a random subsample of the training rows (default 15k)
  because RBF-SVC is O(N²) in samples and becomes impractical above ~20k.
* ``year`` is dropped — on a random split it leaks the training distribution
  into memorisation; on a temporal split it would be a constant within each
  fold. Neither case helps.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Smoothing strength for Bayesian priors: each cell effectively gets `k`
# pseudo-observations drawn from the global rate. This prevents cells with
# only a handful of training rows from producing extreme-valued priors that
# tree ensembles memorise and invert on.
_PRIOR_SMOOTHING_K: int = 20


def _smoothed_rate(counts: pd.Series, sums: pd.Series,
                   global_rate: float, k: int = _PRIOR_SMOOTHING_K) -> pd.Series:
    """Beta-binomial smoothed rate: (sum + k·p_global) / (count + k).

    Cells with count << k are pulled toward global_rate; cells with count >> k
    are essentially unchanged.
    """
    return (sums + k * global_rate) / (counts + k)


def _compute_priors(train_df: pd.DataFrame, target: str
                    ) -> Dict[str, pd.Series]:
    """Compute leakage-safe, Bayesian-smoothed per-cell / per-LSOA violent-rate
    priors on the training rows only.

    Returns a dict with keys ``grid_freq``, ``violent_rate_grid`` and (if the
    column exists) ``violent_rate_lsoa``, plus the scalar ``global_rate`` used
    as a fallback for unseen cells.
    """
    priors: Dict[str, pd.Series] = {}
    global_rate = float(train_df[target].mean())

    # Overall density (frequency of each grid cell)
    priors["grid_freq"] = train_df["grid_id"].value_counts(normalize=True)

    # Smoothed violent-rate prior per grid cell
    grid_group = train_df.groupby("grid_id")[target]
    priors["violent_rate_grid"] = _smoothed_rate(
        counts=grid_group.count(),
        sums=grid_group.sum(),
        global_rate=global_rate,
    )

    # Smoothed violent-rate prior per LSOA (if column present)
    if "lsoa_code" in train_df.columns:
        lsoa_group = train_df.groupby("lsoa_code")[target]
        priors["violent_rate_lsoa"] = _smoothed_rate(
            counts=lsoa_group.count(),
            sums=lsoa_group.sum(),
            global_rate=global_rate,
        )

    priors["global_rate"] = pd.Series(
        [global_rate], index=["global_rate"]
    )
    return priors


def build_features(df: pd.DataFrame,
                   train_idx: Optional[np.ndarray] = None
                   ) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y).

    If ``train_idx`` is provided, spatial priors (``grid_freq``,
    ``violent_rate_grid``, ``violent_rate_lsoa``) are estimated on those rows
    only and broadcast to the whole frame. This is the leakage-safe path used
    during training. If ``train_idx`` is ``None`` the priors are estimated on
    the whole frame — acceptable for inference on already-trained models.
    """
    target = config.CLASSIFIER.target
    prior_src = df.iloc[train_idx] if train_idx is not None else df
    priors = _compute_priors(prior_src, target=target)

    global_rate = float(priors["global_rate"].iloc[0])

    X = pd.DataFrame({
        "latitude":  df["latitude"].to_numpy(),
        "longitude": df["longitude"].to_numpy(),
        "month_num": df["month_num"].to_numpy(),
        "quarter":   df["quarter"].to_numpy(),
    })

    # Distance to Sheffield city centre (degrees → rough km multiplier).
    # Raw lat/lng already encode location, but a radial feature gives linear
    # models a cleaner axis — the centre is the highest-volume hot spot.
    c_lat, c_lng = config.SHEFFIELD_CENTRE
    X["dist_to_centre"] = np.sqrt(
        (df["latitude"].to_numpy() - c_lat) ** 2
        + (df["longitude"].to_numpy() - c_lng) ** 2
    )

    grid_freq_mapped = df["grid_id"].map(priors["grid_freq"]).fillna(0.0).to_numpy()
    X["grid_freq"] = grid_freq_mapped
    # log1p(density) linearises the heavy-tailed cell-frequency distribution
    X["log_grid_freq"] = np.log1p(grid_freq_mapped * 1e6)

    X["violent_rate_grid"] = (
        df["grid_id"].map(priors["violent_rate_grid"])
                    .fillna(global_rate)
                    .to_numpy()
    )
    if "violent_rate_lsoa" in priors:
        X["violent_rate_lsoa"] = (
            df["lsoa_code"].map(priors["violent_rate_lsoa"])
                           .fillna(global_rate)
                           .to_numpy()
        )
    for s in ("winter", "spring", "summer", "autumn"):
        X[f"season_{s}"] = (df["season"] == s).astype(int).to_numpy()

    y = df[target].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _make_models() -> Dict[str, Pipeline]:
    rs = config.CLASSIFIER.random_state
    cw = config.CLASSIFIER.class_weight
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight=cw, random_state=rs)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=config.CLASSIFIER.rf_n_estimators,
                max_depth=config.CLASSIFIER.rf_max_depth,
                min_samples_leaf=config.CLASSIFIER.rf_min_samples_leaf,
                class_weight=cw,
                n_jobs=-1, random_state=rs)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=config.CLASSIFIER.svm_c,
                        probability=True,
                        class_weight=cw,
                        random_state=rs)),
        ]),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
    }


def _subsample(X: pd.DataFrame, y: pd.Series, n: int,
               random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratified random subsample used to cap the SVM training size."""
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(random_state)
    idx_pos = np.flatnonzero(y.to_numpy() == 1)
    idx_neg = np.flatnonzero(y.to_numpy() == 0)
    share_pos = len(idx_pos) / len(y)
    n_pos = int(round(n * share_pos))
    n_neg = n - n_pos
    pick_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
    pick_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
    pick = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(pick)
    return X.iloc[pick], y.iloc[pick]


def fit_all(df: pd.DataFrame) -> List[ClfResult]:
    rs = config.CLASSIFIER.random_state
    target = config.CLASSIFIER.target

    # 1. split indices first, so priors only see training rows
    y_full = df[target].astype(int)
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=config.CLASSIFIER.test_size,
        random_state=rs,
        stratify=y_full,
    )

    # 2. build feature matrix with leakage-safe priors
    X, y = build_features(df, train_idx=train_idx)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info(
        "Split: train=%d (pos=%.1f%%)  test=%d (pos=%.1f%%)  features=%d",
        len(X_train), 100 * y_train.mean(),
        len(X_test),  100 * y_test.mean(),
        X.shape[1],
    )

    results: List[ClfResult] = []
    for name, pipe in _make_models().items():
        # SVM gets a subsample because RBF-SVC is O(N²)
        if name == "svm":
            Xt, yt = _subsample(
                X_train, y_train,
                n=config.CLASSIFIER.svm_subsample,
                random_state=rs,
            )
            logger.info("Fitting %s on %d rows (subsampled) …", name, len(Xt))
        else:
            Xt, yt = X_train, y_train
            logger.info("Fitting %s on %d rows …", name, len(Xt))

        pipe.fit(Xt, yt)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_pred, y_proba)
        report = classification_report(y_test, y_pred, digits=3,
                                        zero_division=0)
        logger.info("%s → %s", name, {k: round(v, 3) for k, v in metrics.items()})
        results.append(ClfResult(name=name, model=pipe,
                                 metrics=metrics, report=report))
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
    pd.DataFrame(rows).to_csv(config.DATA_DIR / "classifier_metrics.csv",
                              index=False)
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
