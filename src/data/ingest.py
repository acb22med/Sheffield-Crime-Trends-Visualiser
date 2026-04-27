"""Ingest Sheffield street-level crime data from the UK Police API.

The API documentation is at https://data.police.uk/docs/. Street-level data for a
polygon is obtained from ``/crimes-street/all-crime?poly=...&date=YYYY-MM``.

Design notes
------------
* One HTTP call per month (polygon-wide).  A 36-month pull is therefore ~36 requests.
* Responses are cached on disk as raw JSON so re-runs skip the network.
* The public rate limit is 15 req/s; we sleep 0.08 s between calls to stay safely under.
* All rows land in a normalised ``crimes`` table in SQLite (see :func:`init_db`).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import config_v1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
@dataclass
class APIClient:
    """Tiny wrapper around `requests` with retries and rate-limit awareness."""

    base_url: str = config_v1.API_BASE
    timeout: int = config_v1.API_TIMEOUT_SEC
    max_retries: int = config_v1.API_MAX_RETRIES
    rate_limit_per_sec: int = config_v1.API_RATE_LIMIT_PER_SEC

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "sheffield-crime-visualiser/1.0 (+github.com/ab3lT)"}
        )
        self._min_interval = 1.0 / max(self.rate_limit_per_sec - 1, 1)
        self._last_call = 0.0

    def get(self, path: str, params: Optional[dict] = None, data: Optional[dict] = None) -> list:
        url = f"{self.base_url}{path}"
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            # throttle
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                # Polygon can be large → use POST with form body, which the API accepts.
                if data:
                    resp = self.session.post(url, params=params, data=data, timeout=self.timeout)
                else:
                    resp = self.session.get(url, params=params, timeout=self.timeout)
                self._last_call = time.monotonic()
            except requests.RequestException as exc:
                logger.warning("Request error %s (attempt %d/%d)", exc, attempt, self.max_retries)
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                # 404 means "no data for that month" — treat as empty.
                return []
            if resp.status_code in (429, 502, 503, 504):
                logger.warning("HTTP %s — backing off %.1fs", resp.status_code, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Exceeded retries fetching {url}")


# ---------------------------------------------------------------------------
# Month utilities
# ---------------------------------------------------------------------------
def months_range(end: date, n: int) -> List[str]:
    """Return the last *n* YYYY-MM strings, oldest first, ending at *end*.

    The API exposes the previous ~3 years. We lag the ``end`` month by one because
    the current month is typically not yet published.
    """
    anchor = (end.replace(day=1) - relativedelta(months=1))
    months = [(anchor - relativedelta(months=i)).strftime("%Y-%m") for i in range(n)]
    return list(reversed(months))


def polygon_to_param(poly: Iterable[tuple]) -> str:
    return ":".join(f"{lat},{lng}" for lat, lng in poly)


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS crimes (
    id                INTEGER PRIMARY KEY,
    api_id            INTEGER,
    category          TEXT     NOT NULL,
    month             TEXT     NOT NULL,            -- YYYY-MM
    latitude          REAL     NOT NULL,
    longitude         REAL     NOT NULL,
    location_type     TEXT,
    street_id         INTEGER,
    street_name       TEXT,
    context           TEXT,
    outcome_category  TEXT,
    outcome_date      TEXT,
    persistent_id     TEXT,
    UNIQUE (api_id, month, category, latitude, longitude)
);
CREATE INDEX IF NOT EXISTS idx_crimes_month    ON crimes(month);
CREATE INDEX IF NOT EXISTS idx_crimes_category ON crimes(category);
CREATE INDEX IF NOT EXISTS idx_crimes_coords   ON crimes(latitude, longitude);
"""


def init_db(path: Path = config_v1.SQLITE_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the project SQLite database."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(month: str) -> Path:
    return config_v1.CACHE_DIR / f"crimes_{month}.json"


def _load_cache(month: str) -> Optional[list]:
    p = _cache_path(month)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            p.unlink(missing_ok=True)
    return None


def _save_cache(month: str, payload: list) -> None:
    _cache_path(month).write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------
def fetch_month(client: APIClient, month: str, use_cache: bool = True) -> list:
    """Fetch a single month's records for the Sheffield polygon."""
    if use_cache:
        cached = _load_cache(month)
        if cached is not None:
            logger.debug("cache hit for %s (%d records)", month, len(cached))
            return cached

    poly = polygon_to_param(config_v1.SHEFFIELD_POLYGON)
    payload = client.get(
        "/crimes-street/all-crime",
        data={"poly": poly, "date": month},
    )
    _save_cache(month, payload)
    return payload


def _flatten_record(rec: dict, month: str) -> tuple:
    """Normalise one API record into a row tuple matching the ``crimes`` schema."""
    loc = rec.get("location") or {}
    street = loc.get("street") or {}
    outcome = rec.get("outcome_status") or {}
    # lat/lng come back as strings — cast up front, skip rows with missing coords.
    try:
        lat = float(loc["latitude"])
        lng = float(loc["longitude"])
    except (KeyError, TypeError, ValueError):
        return tuple()  # sentinel: caller will filter empties

    return (
        rec.get("id"),
        rec.get("category", "unknown"),
        rec.get("month") or month,
        lat,
        lng,
        loc.get("location_type"),
        street.get("id"),
        street.get("name"),
        rec.get("context") or None,
        outcome.get("category"),
        outcome.get("date"),
        rec.get("persistent_id") or None,
    )


def _insert_many(conn: sqlite3.Connection, rows: Iterable[tuple]) -> int:
    sql = """
    INSERT OR IGNORE INTO crimes
        (api_id, category, month, latitude, longitude, location_type, street_id,
         street_name, context, outcome_category, outcome_date, persistent_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = conn.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


def ingest(months: int = config_v1.DEFAULT_MONTHS_BACK, use_cache: bool = True) -> int:
    """Pull *months* months of crime data into the SQLite DB. Returns row count added."""
    client = APIClient()
    conn = init_db()
    total_inserted = 0
    month_list = months_range(date.today(), months)

    logger.info("Fetching %d months: %s -> %s", len(month_list), month_list[0], month_list[-1])
    for m in tqdm(month_list, desc="UK Police API", unit="month"):
        try:
            payload = fetch_month(client, m, use_cache=use_cache)
        except Exception as exc:  # network, JSON decode, etc. — log and continue
            logger.error("Failed month %s: %s", m, exc)
            continue

        rows = [r for r in (_flatten_record(rec, m) for rec in payload) if r]
        if not rows:
            continue
        total_inserted += _insert_many(conn, rows)

    conn.close()
    logger.info("Inserted %d new rows.", total_inserted)
    return total_inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ingest()
