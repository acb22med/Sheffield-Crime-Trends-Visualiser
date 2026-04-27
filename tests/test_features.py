"""Unit tests for src/data/features.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import features


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    rows = []
    cats = ["burglary", "violent-crime", "robbery", "drugs"]
    for i, d in enumerate(dates):
        for j, c in enumerate(cats):
            rows.append({
                "category": c,
                "month": d.strftime("%Y-%m"),
                "month_date": d,
                "latitude": 53.38 + 0.001 * (i + j),
                "longitude": -1.47 - 0.001 * (i + j),
                "street_name": f"Street {j}",
                "street_id": j,
            })
    return pd.DataFrame(rows)


def test_add_temporal(cleaned_df):
    out = features.add_temporal(cleaned_df)
    for col in ("year", "month_num", "quarter", "season"):
        assert col in out.columns
    assert set(out["season"]).issubset({"winter", "spring", "summer", "autumn"})


def test_add_target_is_binary_and_consistent(cleaned_df):
    out = features.add_target(cleaned_df)
    assert set(out["is_violent"].unique()).issubset({0, 1})
    violent_mask = out["category"].isin({"violent-crime", "robbery", "possession-of-weapons"})
    assert (out.loc[violent_mask, "is_violent"] == 1).all()
    assert (out.loc[~violent_mask, "is_violent"] == 0).all()


def test_add_spatial_bins_in_range(cleaned_df):
    out = features.add_spatial_bins(cleaned_df, grid=20)
    assert out["grid_lat"].between(0, 19).all()
    assert out["grid_lng"].between(0, 19).all()
    assert out["grid_id"].str.contains("_").all()


def test_monthly_totals_sorted(cleaned_df):
    out = features.monthly_totals(cleaned_df)
    assert out["month_date"].is_monotonic_increasing
    assert (out["crime_count"] > 0).all()


def test_monthly_totals_by_category_preserves_totals(cleaned_df):
    total = features.monthly_totals(cleaned_df)["crime_count"].sum()
    by_cat = features.monthly_totals_by_category(cleaned_df)["crime_count"].sum()
    assert total == by_cat
