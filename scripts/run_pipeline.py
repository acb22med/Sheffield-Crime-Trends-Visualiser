"""Run the full data pipeline: ingest → clean → feature-engineer → persist.

Usage
-----
    python scripts/run_pipeline.py [--months N] [--no-cache]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# make `import config` / `import src.*` work whichever directory we launch from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_v1  # noqa: E402
from src.data.features import engineer, persist_aggregates  # noqa: E402
from src.data.ingest import ingest  # noqa: E402
from src.data.preprocess import clean, load_raw, save_clean  # noqa: E402
from src.eda.explore import run_all as run_eda  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Sheffield crime pipeline end-to-end.")
    p.add_argument("--months", type=int, default=config_v1.DEFAULT_MONTHS_BACK,
                   help=f"Months of history to pull (default: {config_v1.DEFAULT_MONTHS_BACK})")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore the on-disk API cache and refetch everything")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip the API fetch and use whatever is already in SQLite")
    p.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    args = parse_args()

    if not args.skip_ingest:
        ingest(months=args.months, use_cache=not args.no_cache)

    raw = load_raw()
    if raw.empty:
        logging.error("Raw table is empty — check network / API access.")
        sys.exit(1)

    cleaned = clean(raw)
    save_clean(cleaned)

    features = engineer(cleaned)
    persist_aggregates(features)

    if not args.skip_eda:
        run_eda(features)

    logging.info("✓ Pipeline finished.  SQLite: %s", config_v1.SQLITE_PATH)


if __name__ == "__main__":
    main()
