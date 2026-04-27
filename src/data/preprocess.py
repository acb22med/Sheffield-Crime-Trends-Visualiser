"""Pre-processing: deduplication, coordinate validation, category normalisation.

All functions are pure over a ``pandas.DataFrame`` loaded from the SQLite ``crimes``
table (see :func:`load_raw`). This makes the functions trivially unit-testable and
keeps the pipeline deterministic.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from shapely.geometry import Point, Polygon

import config_v1

logger = logging.getLogger(__name__)

_SHEFFIELD_SHAPE = Polygon([(lng, lat) for lat, lng in config_v1.SHEFFIELD_POLYGON])


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_raw(db_path: Path = config_v1.SQLITE_PATH) -> pd.DataFrame:
    """Load everything from the ``crimes`` table into a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM crimes", conn)
    logger.info("Loaded %d raw rows from %s", len(df), db_path.name)
    return df


# ---------------------------------------------------------------------------
# Cleaning steps (pure → testable)
# ---------------------------------------------------------------------------
def validate_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing / out-of-bounds / outside-Sheffield coordinates."""
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]

    mask = df.apply(
        lambda r: _SHEFFIELD_SHAPE.covers(Point(r["longitude"], r["latitude"])),
        axis=1,
    )
    df = df[mask].copy()
    logger.info("validate_coords: %d → %d (dropped %d)", before, len(df), before - len(df))
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate crime records.

    The API sometimes re-publishes the same incident across adjacent months or with
    a new ``id``. We collapse on the tuple (category, month, rounded lat/lng,
    street_id) — this mirrors what a human would call "the same report".
    """
    before = len(df)
    round_digits = 5  # ~1.1 m precision
    key = df.assign(
        _lat=df["latitude"].round(round_digits),
        _lng=df["longitude"].round(round_digits),
    )
    df = key.drop_duplicates(
        subset=["category", "month", "_lat", "_lng", "street_id"], keep="first"
    ).drop(columns=["_lat", "_lng"])
    logger.info("deduplicate: %d → %d (dropped %d)", before, len(df), before - len(df))
    return df.reset_index(drop=True)


def normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map messy category strings to the canonical set defined in config."""
    aliases = {
        "theft-from-the-person": "theft-from-person",
        "anti-social": "anti-social-behaviour",
        "violence-and-sexual-offences": "violent-crime",
    }
    df = df.copy()
    df["category"] = (
        df["category"].str.strip().str.lower().replace(aliases)
    )
    unknown = set(df["category"]) - set(config_v1.CANONICAL_CATEGORIES)
    if unknown:
        logger.warning("Unmapped categories: %s → 'other-crime'", sorted(unknown))
        df.loc[df["category"].isin(unknown), "category"] = "other-crime"
    return df


def parse_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add a proper ``month_date`` column (first day of the month, as datetime)."""
    df = df.copy()
    df["month_date"] = pd.to_datetime(df["month"] + "-01", errors="coerce")
    before = len(df)
    df = df.dropna(subset=["month_date"])
    if len(df) < before:
        logger.warning("parse_month: dropped %d rows with unparseable month", before - len(df))
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def clean(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the clean frame."""
    if df is None:
        df = load_raw()
    df = validate_coords(df)
    # normalise BEFORE dedup so case/alias variants collapse on the same pass
    df = normalise_categories(df)
    df = deduplicate(df)
    df = parse_month(df)
    return df


def save_clean(df: pd.DataFrame, db_path: Path = config_v1.SQLITE_PATH) -> None:
    """Persist the cleaned frame back to SQLite as table ``crimes_clean``."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql("crimes_clean", conn, if_exists="replace", index=False)
        conn.commit()
    logger.info("Saved %d clean rows to crimes_clean", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    clean_df = clean()
    save_clean(clean_df)
