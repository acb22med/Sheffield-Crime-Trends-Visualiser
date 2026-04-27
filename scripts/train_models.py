"""Fit clustering, forecasting and classification models and save all artefacts.

Reads the clean/engineered data from SQLite. Run after ``run_pipeline.py``.

Usage
-----
    python scripts/train_models.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                             # noqa: E402
from src.data.features import engineer, monthly_totals    # noqa: E402
from src.models import classification, clustering, forecasting  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )

    if not config.SQLITE_PATH.exists():
        logging.error("SQLite DB not found at %s — run scripts/run_pipeline.py first",
                      config.SQLITE_PATH)
        sys.exit(1)

    df = engineer()
    if df.empty:
        logging.error("Feature frame is empty.")
        sys.exit(1)

    # logging.info("Training clustering models …")
    # km, db, sweep = clustering.run(df)

    logging.info("Training forecasting models …")
    cv = forecasting.run(monthly_totals(df))

    logging.info("Training classifiers …")
    clf = classification.run(df)

    logging.info("✓ All models trained.")
    logging.info("Classifier metrics:\n%s", clf.to_string(index=False))
    logging.info("Forecast CV summary (mean):\n%s",
                 cv.groupby("model")[["rmse", "mae"]].mean().round(2))


if __name__ == "__main__":
    main()
