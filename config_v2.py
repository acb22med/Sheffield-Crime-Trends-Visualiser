"""Central configuration for the Sheffield Crime Trends Visualiser.

All tunable paths, geography, API parameters and model hyperparameters live here so
that scripts, tests and the Dash app can stay pure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
FIGURE_DIR: Path = ROOT_DIR / "figures"
MODEL_DIR: Path = DATA_DIR / "models"
CACHE_DIR: Path = DATA_DIR / "cache"

for _d in (DATA_DIR, FIGURE_DIR, MODEL_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SQLITE_PATH: Path = DATA_DIR / "sheffield_crime.sqlite"


# ---------------------------------------------------------------------------
# UK Police API
# ---------------------------------------------------------------------------
API_BASE: str = "https://data.police.uk/api"
API_RATE_LIMIT_PER_SEC: int = 15           # data.police.uk published limit
API_TIMEOUT_SEC: int = 30
API_MAX_RETRIES: int = 4


# ---------------------------------------------------------------------------
# Sheffield geography
# ---------------------------------------------------------------------------
# Approximate polygon covering Sheffield metropolitan district.
# The UK Police API accepts polygons as "lat,lng:lat,lng:..." with up to ~10k chars.
# We keep it to 6 vertices — good enough coverage, well within the limit.
SHEFFIELD_POLYGON: List[Tuple[float, float]] = [
    (53.485, -1.635),   # NW
    (53.485, -1.340),   # NE
    (53.415, -1.300),   # E
    (53.310, -1.330),   # SE
    (53.300, -1.600),   # SW
    (53.400, -1.680),   # W
]
SHEFFIELD_CENTRE: Tuple[float, float] = (53.3811, -1.4701)  # for default map view


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
DEFAULT_MONTHS_BACK: int = 36              # how far back to pull on a fresh run
COORD_DEDUP_TOLERANCE: float = 1e-5        # ~1.1 m — treat as same point

CANONICAL_CATEGORIES: List[str] = [
    "anti-social-behaviour",
    "bicycle-theft",
    "burglary",
    "criminal-damage-arson",
    "drugs",
    "other-theft",
    "possession-of-weapons",
    "public-order",
    "robbery",
    "shoplifting",
    "theft-from-person",
    "vehicle-crime",
    "violent-crime",
    "other-crime",
]


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class ClusterConfig:
    kmeans_k_range: Tuple[int, int] = (3, 12)      # inclusive range to search
    kmeans_random_state: int = 42
    dbscan_eps_metres: float = 250.0
    dbscan_min_samples: int = 25


@dataclass
class ForecastConfig:
    test_horizon_months: int = 6
    cv_initial_months: int = 24        # raised from 18 — Prophet needs more history
    cv_step_months: int = 3
    arima_order_search: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
            (0, 1, 1), (1, 0, 1), (3, 1, 1),
        ]
    )
    prophet_seasonality_mode: str = "additive"
    prophet_yearly_seasonality: bool = True   # enable explicit yearly term


@dataclass
class ClassifierConfig:
    target: str = "is_violent"      # binary: violent vs non-violent
    test_size: float = 0.2
    random_state: int = 42
    rf_n_estimators: int = 300
    svm_c: float = 1.0
    svm_subsample: int = 15000       # cap SVM training size — RBF-SVC is O(N²)
    class_weight: str = "balanced"    # fixes majority-class collapse


CLUSTER = ClusterConfig()
FORECAST = ForecastConfig()
CLASSIFIER = ClassifierConfig()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
APP_HOST: str = "127.0.0.1"
APP_PORT: int = 8050
APP_DEBUG: bool = False
MAP_DEFAULT_ZOOM: int = 11
