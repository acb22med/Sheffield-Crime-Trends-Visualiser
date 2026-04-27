"""Smoke-tests for model wrappers — use tiny synthetic data so runs stay <5 s.

Prophet is heavy to import in a test; we skip forecasting here and cover it in the
end-to-end runtime instead. Clustering + classifiers are light enough to test
directly on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import classification_v1, clustering


@pytest.fixture
def synth_df():
    rng = np.random.default_rng(0)
    n = 400
    # Two spatial blobs
    blob1 = np.column_stack([
        rng.normal(53.38, 0.01, n // 2),
        rng.normal(-1.47, 0.01, n // 2),
    ])
    blob2 = np.column_stack([
        rng.normal(53.42, 0.01, n // 2),
        rng.normal(-1.43, 0.01, n // 2),
    ])
    coords = np.vstack([blob1, blob2])
    df = pd.DataFrame({
        "latitude":  coords[:, 0],
        "longitude": coords[:, 1],
        "month":     rng.choice(["2024-01", "2024-02", "2024-03"], n),
        "category":  rng.choice(["burglary", "violent-crime", "robbery", "drugs"], n),
    })
    # Build temporal / target columns the way features.engineer() would
    md = pd.to_datetime(df["month"] + "-01")
    df["month_date"] = md
    df["year"] = md.dt.year
    df["month_num"] = md.dt.month
    df["quarter"] = md.dt.quarter
    df["season"] = "winter"
    df["is_violent"] = df["category"].isin({"violent-crime", "robbery"}).astype(int)
    df["grid_id"] = (df["latitude"].round(2).astype(str) + "_"
                     + df["longitude"].round(2).astype(str))
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_fit_kmeans_returns_valid_result(synth_df):
    km, sweep = clustering.fit_kmeans(synth_df, k_range=(2, 4))
    assert km.n_clusters in {2, 3, 4}
    assert len(km.labels) == len(synth_df)
    assert set(sweep.columns) >= {"k", "silhouette", "davies_bouldin", "inertia"}


def test_fit_dbscan_labels_correct_shape(synth_df):
    res = clustering.fit_dbscan(synth_df, eps_metres=2000, min_samples=10)
    assert len(res.labels) == len(synth_df)
    # at least one non-noise cluster should form on this dense synthetic data
    assert res.n_clusters >= 1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_build_features_shape_matches(synth_df):
    X, y = classification_v1.build_features(synth_df)
    assert len(X) == len(y) == len(synth_df)
    assert {"latitude", "longitude", "month_num", "year", "quarter", "grid_freq"}.issubset(X.columns)
    assert set(y.unique()).issubset({0, 1})


def test_fit_all_returns_three_models(synth_df):
    results = classification_v1.fit_all(synth_df)
    assert {r.name for r in results} == {"logistic_regression", "random_forest", "svm"}
    for r in results:
        for k in ("accuracy", "f1", "precision", "recall", "roc_auc"):
            assert k in r.metrics
            assert 0.0 <= r.metrics[k] <= 1.0
