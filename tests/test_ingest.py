"""Unit tests for src/data/ingest.py — API client mocked, no real network calls."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from src.data import ingest


def test_months_range_returns_n_months_oldest_first():
    out = ingest.months_range(date(2024, 6, 15), 6)
    assert len(out) == 6
    assert out == sorted(out)  # ascending
    # Current month (June) should be excluded; latest should be May.
    assert out[-1] == "2024-05"


def test_polygon_to_param_format():
    poly = [(53.48, -1.63), (53.30, -1.30)]
    out = ingest.polygon_to_param(poly)
    assert out == "53.48,-1.63:53.3,-1.3"


def test_flatten_record_happy_path():
    rec = {
        "id": 123,
        "category": "burglary",
        "month": "2024-01",
        "location": {
            "latitude": "53.38",
            "longitude": "-1.47",
            "location_type": "Force",
            "street": {"id": 99, "name": "Fargate"},
        },
        "outcome_status": {"category": "Investigation complete", "date": "2024-02"},
        "persistent_id": "abc",
    }
    row = ingest._flatten_record(rec, "2024-01")
    assert row[0] == 123
    assert row[1] == "burglary"
    assert row[3] == pytest.approx(53.38)
    assert row[4] == pytest.approx(-1.47)
    assert row[7] == "Fargate"
    assert row[9] == "Investigation complete"


def test_flatten_record_missing_coords_returns_empty():
    rec = {"id": 1, "category": "burglary", "location": {}}
    assert ingest._flatten_record(rec, "2024-01") == tuple()


def test_api_client_get_uses_cache_hit(monkeypatch, tmp_path):
    # Redirect cache dir to a tmp location
    monkeypatch.setattr(ingest.config, "CACHE_DIR", tmp_path)
    payload = [{"id": 1, "category": "burglary",
                "location": {"latitude": "53.38", "longitude": "-1.47"}}]
    (tmp_path / "crimes_2024-01.json").write_text(json.dumps(payload))

    # Replace APIClient.get so any accidental network call fails the test
    client = MagicMock(spec=ingest.APIClient)
    client.get.side_effect = AssertionError("Should not touch the network")

    out = ingest.fetch_month(client, "2024-01", use_cache=True)
    assert out == payload


def test_api_client_get_writes_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.config, "CACHE_DIR", tmp_path)
    client = MagicMock()
    client.get.return_value = [{"hello": "world"}]

    out = ingest.fetch_month(client, "2024-02", use_cache=True)
    assert out == [{"hello": "world"}]
    assert (tmp_path / "crimes_2024-02.json").exists()


def test_init_db_creates_tables(tmp_path):
    db = tmp_path / "x.sqlite"
    conn = ingest.init_db(db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "crimes" in tables
    conn.close()
