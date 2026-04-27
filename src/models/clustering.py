"""Spatial clustering of Sheffield crime reports.

Two algorithms, both using great-circle-safe coordinates:

* **K-Means** — swept over a range of ``k`` values; the best model (by silhouette
  score) is retained along with a full diagnostic table.
* **DBSCAN** — ``eps`` is specified in metres; we convert to radians and use the
  haversine metric so clusters respect actual distance on the sphere.

Results are saved as pickled models and as a CSV of per-point labels so the Dash
app can overlay them without re-fitting.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config_v1

logger = logging.getLogger(__name__)

_EARTH_R_METRES = 6_371_000.0


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    n_clusters: int
    params: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def _prepare_xy(df: pd.DataFrame) -> np.ndarray:
    X = df[["latitude", "longitude"]].to_numpy()
    return StandardScaler().fit_transform(X)


def fit_kmeans(df: pd.DataFrame, k_range: Tuple[int, int] = config_v1.CLUSTER.kmeans_k_range
               ) -> Tuple[ClusterResult, pd.DataFrame]:
    """Sweep k in ``k_range`` (inclusive) and keep the model with highest silhouette."""
    X = _prepare_xy(df)
    rows: List[dict] = []
    best: Optional[ClusterResult] = None

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=config_v1.CLUSTER.kmeans_random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        db = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "inertia": km.inertia_})
        if best is None or (sil is not np.nan and sil > best.silhouette):
            best = ClusterResult(
                name=f"kmeans_k{k}", labels=labels, silhouette=float(sil),
                davies_bouldin=float(db), n_clusters=k, params={"model": km},
            )

    sweep = pd.DataFrame(rows)
    logger.info("K-Means sweep complete — best k=%d (silhouette=%.3f)",
                best.n_clusters, best.silhouette)
    return best, sweep


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def fit_dbscan(df: pd.DataFrame,
               eps_metres: float = config_v1.CLUSTER.dbscan_eps_metres,
               min_samples: int = config_v1.CLUSTER.dbscan_min_samples) -> ClusterResult:
    """DBSCAN on lat/lng using haversine metric. ``eps_metres`` is converted to radians."""
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_metres / _EARTH_R_METRES

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords)

    core_mask = labels != -1
    n_clusters = len(set(labels[core_mask]))
    sil = (
        silhouette_score(coords[core_mask], labels[core_mask], metric="haversine")
        if core_mask.sum() > n_clusters > 1 else None
    )
    db_score = (
        davies_bouldin_score(coords[core_mask], labels[core_mask])
        if core_mask.sum() > n_clusters > 1 else None
    )
    logger.info("DBSCAN: %d clusters, %d noise points", n_clusters, (~core_mask).sum())
    return ClusterResult(
        name=f"dbscan_eps{int(eps_metres)}_ms{min_samples}",
        labels=labels,
        silhouette=float(sil) if sil is not None else None,
        davies_bouldin=float(db_score) if db_score is not None else None,
        n_clusters=n_clusters,
        params={"model": db, "eps_metres": eps_metres, "min_samples": min_samples},
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_labels(df: pd.DataFrame, km: ClusterResult, db: ClusterResult,
                path: Path = config_v1.DATA_DIR / "cluster_labels.csv") -> None:
    out = df[["latitude", "longitude", "month", "category"]].copy()
    out["kmeans_label"] = km.labels
    out["dbscan_label"] = db.labels
    out.to_csv(path, index=False)
    logger.info("Wrote cluster labels → %s", path)


def save_models(km: ClusterResult, db: ClusterResult,
                path: Path = config_v1.MODEL_DIR / "cluster_models.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "kmeans": km.params["model"],
                "dbscan": db.params["model"],
                "kmeans_n": km.n_clusters,
                "dbscan_eps_metres": db.params["eps_metres"],
                "dbscan_min_samples": db.params["min_samples"],
            },
            f,
        )
    logger.info("Wrote cluster models → %s", path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> Tuple[ClusterResult, ClusterResult, pd.DataFrame]:
    km, sweep = fit_kmeans(df)
    db = fit_dbscan(df)
    save_labels(df, km, db)
    save_models(km, db)
    sweep.to_csv(config_v1.DATA_DIR / "kmeans_sweep.csv", index=False)
    return km, db, sweep
