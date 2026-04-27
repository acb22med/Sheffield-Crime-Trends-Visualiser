"""Dash app entry point for the Sheffield Crime Trends Visualiser."""
from __future__ import annotations

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd

import config_v1
from src.app.callbacks import register_callbacks
from src.app.layout import build_layout
from src.data.features import engineer

logger = logging.getLogger(__name__)


def _load_cluster_labels() -> pd.DataFrame | None:
    path = config_v1.DATA_DIR / "cluster_labels.csv"
    if not path.exists():
        logger.warning("cluster_labels.csv not found — run scripts/train_models.py first")
        return None
    return pd.read_csv(path)


def create_app() -> dash.Dash:
    df = engineer()
    df = df.sort_values("month_date").reset_index(drop=True)
    cluster_df = _load_cluster_labels()

    months = sorted(df["month"].unique().tolist())
    start_date = pd.to_datetime(months[0] + "-01").date()
    end_date = pd.to_datetime(months[-1] + "-01").date()

    categories = sorted(df["category"].unique().tolist())
    neighbourhoods = sorted(df["street_name"].dropna().unique().tolist())[:500]

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Sheffield Crime Trends Visualiser",
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(
        categories,
        [start_date, end_date],
        neighbourhoods,
    )
    register_callbacks(app, df, cluster_df)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    app = create_app()
    logger.info("Starting app on http://%s:%d", config_v1.APP_HOST, config_v1.APP_PORT)
    app.run(host=config_v1.APP_HOST, port=config_v1.APP_PORT, debug=config_v1.APP_DEBUG)


if __name__ == "__main__":
    main()
