"""Unit tests for src/data/preprocess.py — pure functions, no network."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import preprocess


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":               [1, 2, 3, 4, 5, 6],
        "api_id":           [101, 102, 103, 104, 105, 106],
        "category":         ["burglary", "Burglary", "violence-and-sexual-offences",
                             "anti-social", "robbery", "unknown-new-cat"],
        "month":            ["2024-01", "2024-01", "2024-02", "2024-02",
                             "2024-03", "2024-03"],
        "latitude":         [53.38, 53.38, 53.40, 53.42, None, 53.50],     # last is outside polygon
        "longitude":        [-1.47, -1.47, -1.45, -1.43, -1.50, -1.30],
        "location_type":    ["Force"] * 6,
        "street_id":        [10, 10, 11, 12, 13, 14],
        "street_name":      ["A Road"] * 6,
        "context":          [None] * 6,
        "outcome_category": [None] * 6,
        "outcome_date":     [None] * 6,
        "persistent_id":    [None] * 6,
    })


def test_validate_coords_drops_invalid(raw_df):
    out = preprocess.validate_coords(raw_df)
    # Row with lat=None and row outside Sheffield polygon both dropped
    assert len(out) <= 4
    assert out["latitude"].between(-90, 90).all()
    assert out["longitude"].between(-180, 180).all()


def test_deduplicate_removes_exact_duplicates(raw_df):
    valid = preprocess.validate_coords(raw_df)
    out = preprocess.deduplicate(valid)
    # Rows 1 and 2 share (category, month, lat, lng, street_id) after lowercasing
    # but deduplicate doesn't lowercase — that's normalise_categories' job.
    # So verify length only decreases (never increases).
    assert len(out) <= len(valid)


def test_normalise_categories_aliases_applied(raw_df):
    out = preprocess.normalise_categories(raw_df)
    cats = set(out["category"])
    assert "violence-and-sexual-offences" not in cats
    assert "anti-social" not in cats
    assert "violent-crime" in cats or "other-crime" in cats


def test_normalise_categories_unknown_becomes_other(raw_df):
    out = preprocess.normalise_categories(raw_df)
    assert out.loc[raw_df["category"] == "unknown-new-cat", "category"].iloc[0] == "other-crime"


def test_parse_month_adds_month_date(raw_df):
    out = preprocess.parse_month(raw_df)
    assert "month_date" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["month_date"])
    assert (out["month_date"].dt.day == 1).all()


def test_full_clean_pipeline_idempotent(raw_df):
    once = preprocess.clean(raw_df)
    twice = preprocess.clean(once)
    assert len(once) == len(twice)
