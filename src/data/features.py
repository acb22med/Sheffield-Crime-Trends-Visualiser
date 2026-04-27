"""Feature engineering.

Adds temporal features (year / month / quarter / season), weekly aggregates and
a binary ``is_violent`` target used by the supervised classifiers.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config_v1

logger = logging.getLogger(__name__)


_VIOLENT_CATEGORIES = {
    "violent-crime",
    "robbery",
    "possession-of-weapons",
}

_SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring", 5: "spring",
    6: "summer",  7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


# ---------------------------------------------------------------------------
# Row-level features
# ---------------------------------------------------------------------------
def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    md = pd.to_datetime(df["month_date"])
    df["year"] = md.dt.year
    df["month_num"] = md.dt.month
    df["quarter"] = md.dt.quarter
    df["season"] = df["month_num"].map(_SEASON_MAP)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_violent"] = df["category"].isin(_VIOLENT_CATEGORIES).astype(int)
    return df


def add_spatial_bins(df: pd.DataFrame, grid: int = 50) -> pd.DataFrame:
    """Discretise coordinates into an NxN grid. Useful as a categorical feature."""
    df = df.copy()
    lat_bins = np.linspace(df["latitude"].min(), df["latitude"].max(), grid + 1)
    lng_bins = np.linspace(df["longitude"].min(), df["longitude"].max(), grid + 1)
    df["grid_lat"] = np.clip(np.digitize(df["latitude"], lat_bins) - 1, 0, grid - 1)
    df["grid_lng"] = np.clip(np.digitize(df["longitude"], lng_bins) - 1, 0, grid - 1)
    df["grid_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lng"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly crime counts suitable for ARIMA / Prophet."""
    out = (
        df.groupby("month_date", as_index=False)
        .size()
        .rename(columns={"size": "crime_count"})
        .sort_values("month_date")
        .reset_index(drop=True)
    )
    return out


def monthly_totals_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month_date", "category"], as_index=False)
        .size()
        .rename(columns={"size": "crime_count"})
        .sort_values(["category", "month_date"])
        .reset_index(drop=True)
    )


def weekly_aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Spec calls for weekly aggregates; month_date is month-start so we simply
    distribute the monthly count over ISO-weeks that fall inside that month.
    Returns a (week_start, category, crime_count) frame."""
    df = df.copy()
    # expand each month to its weeks, then divide evenly (rough but fine for EDA)
    records = []
    for (m, c), grp in df.groupby(["month_date", "category"]):
        weeks = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-MON")
        if len(weeks) == 0:
            weeks = pd.DatetimeIndex([m])
        per_week = len(grp) / len(weeks)
        for w in weeks:
            records.append({"week_start": w, "category": c, "crime_count": per_week})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def engineer(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        from src.data.preprocess import load_raw, clean
        df = clean(load_raw())
    df = add_temporal(df)
    df = add_target(df)
    df = add_spatial_bins(df)
    return df


def persist_aggregates(df: pd.DataFrame, db_path: Path = config_v1.SQLITE_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        monthly_totals(df).to_sql("agg_monthly", conn, if_exists="replace", index=False)
        monthly_totals_by_category(df).to_sql(
            "agg_monthly_category", conn, if_exists="replace", index=False
        )
        weekly_aggregate_by_category(df).to_sql(
            "agg_weekly_category", conn, if_exists="replace", index=False
        )
        conn.commit()
    logger.info("Persisted aggregate tables: agg_monthly / agg_monthly_category / agg_weekly_category")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = engineer()
    persist_aggregates(df)
