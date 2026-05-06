"""Random-forest forecaster on the category-month panel.

This is the headline forecasting model. Unlike the ARIMA / Prophet pair in
``forecasting.py`` — which fit one model to the city-wide aggregate series — the
random forest sits on a *panel*: one row per (category, month) pair, with the
target being that month's count of that category and the features being the
lagged counts (lag-1, lag-2, lag-3, lag-6), a 3-month rolling mean, the
calendar month and an ordinal time index, plus the encoded category.

Why this is more useful than a city-wide aggregate forecaster
-------------------------------------------------------------
* Resource planners care about category breakdowns (how much violent crime
  next month?), not the all-categories total.
* The panel framing gives the model many more rows than the 36-month aggregate
  series, which lets a tree ensemble actually learn something.
* Tree ensembles match or beat ARIMA / Prophet on short autoregressive series
  with categorical conditioning (Makridakis et al., 2018).

Outputs
-------
* ``data/models/rf_forecaster.pkl``         — pickled model + metadata
* ``data/forecast_rf_metrics.csv``          — held-out MAE, RMSE, MAE-as-%-of-mean
* ``data/forecast_rf_future.csv``           — long-format per-category forecast
* ``data/forecast_rf_future_total.csv``     — same forecast aggregated city-wide
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config_v1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
LAGS: Tuple[int, ...] = (1, 2, 3, 6)
ROLL_WINDOW: int = 3
N_ESTIMATORS: int = 300
RANDOM_STATE: int = 42
TEST_HORIZON_MONTHS: int = 3


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a (category, month) panel and derive the lag, rolling and
    calendar features the RF will train on.
    """
    panel = (
        df.groupby(["category", "month_date"])
          .size().rename("count").reset_index()
          .sort_values(["category", "month_date"])
          .reset_index(drop=True)
    )

    # Pad each category onto a continuous monthly index so the lags align.
    full_months = pd.date_range(panel["month_date"].min(),
                                panel["month_date"].max(), freq="MS")
    cats = panel["category"].unique()
    grid = pd.MultiIndex.from_product(
        [cats, full_months], names=["category", "month_date"]
    ).to_frame(index=False)
    panel = grid.merge(panel, on=["category", "month_date"], how="left").fillna({"count": 0})

    for k in LAGS:
        panel[f"lag_{k}"] = panel.groupby("category")["count"].shift(k)

    panel[f"roll{ROLL_WINDOW}"] = (
        panel.groupby("category")["count"]
             .shift(1)
             .rolling(ROLL_WINDOW, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
    )

    panel["month_n"] = panel["month_date"].dt.month
    panel["year"] = panel["month_date"].dt.year
    panel["t"] = (
        (panel["month_date"].dt.year - panel["month_date"].dt.year.min()) * 12
        + panel["month_date"].dt.month
    )

    panel = panel.dropna(subset=[f"lag_{k}" for k in LAGS]).reset_index(drop=True)
    panel["cat_enc"] = panel["category"].astype("category").cat.codes
    return panel


def feature_cols() -> List[str]:
    return [f"lag_{k}" for k in LAGS] + [f"roll{ROLL_WINDOW}", "month_n", "year", "t", "cat_enc"]


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
@dataclass
class RFForecastResult:
    model: RandomForestRegressor
    feature_cols: List[str]
    cat_codes: dict
    metrics: dict
    train_rows: int
    test_rows: int


def fit_and_evaluate(panel: pd.DataFrame,
                     test_horizon: int = TEST_HORIZON_MONTHS) -> RFForecastResult:
    cutoff = panel["month_date"].max() - pd.offsets.MonthBegin(test_horizon)
    train = panel[panel["month_date"] <= cutoff]
    test = panel[panel["month_date"] > cutoff]

    feats = feature_cols()
    X_train, y_train = train[feats], train["count"]
    X_test, y_test = test[feats], test["count"]

    logger.info("RF forecaster: train=%d rows, test=%d rows, features=%d",
                len(X_train), len(X_test), len(feats))

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_test.to_numpy() - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test.to_numpy() - y_pred) ** 2)))
    mean_y = float(y_test.mean()) if len(y_test) else 0.0
    pct = (mae / mean_y * 100) if mean_y > 0 else float("nan")

    logger.info("RF forecaster — MAE=%.2f  RMSE=%.2f  MAE%%-of-mean=%.1f%%",
                mae, rmse, pct)

    cat_codes = dict(
        panel[["category", "cat_enc"]]
          .drop_duplicates().set_index("category")["cat_enc"]
    )

    return RFForecastResult(
        model=model,
        feature_cols=feats,
        cat_codes=cat_codes,
        metrics={"mae": mae, "rmse": rmse, "mae_pct_of_mean": pct},
        train_rows=len(X_train),
        test_rows=len(X_test),
    )


# ---------------------------------------------------------------------------
# Recursive multi-step forecast
# ---------------------------------------------------------------------------
def forecast_future(panel: pd.DataFrame,
                    result: RFForecastResult,
                    horizon: int = TEST_HORIZON_MONTHS) -> pd.DataFrame:
    """Walk the model forward ``horizon`` months for every category.

    Each step appends the predicted count to the per-category history, then
    rebuilds the lag features for the next step.
    """
    last_month = panel["month_date"].max()
    feats = result.feature_cols

    history: dict[str, list[float]] = {
        cat: panel.loc[panel["category"] == cat]
                   .sort_values("month_date")["count"].tolist()
        for cat in panel["category"].unique()
    }

    rows = []
    for step in range(1, horizon + 1):
        new_month = last_month + pd.offsets.MonthBegin(step)
        for cat, code in result.cat_codes.items():
            past = history[cat]
            if len(past) < max(LAGS):
                continue
            row = {f"lag_{k}": past[-k] for k in LAGS}
            row[f"roll{ROLL_WINDOW}"] = float(np.mean(past[-ROLL_WINDOW:]))
            row["month_n"] = new_month.month
            row["year"] = new_month.year
            row["t"] = (
                (new_month.year - panel["month_date"].dt.year.min()) * 12
                + new_month.month
            )
            row["cat_enc"] = code
            X_row = pd.DataFrame([row], columns=feats)
            yhat = float(result.model.predict(X_row)[0])
            yhat = max(0.0, yhat)
            history[cat].append(yhat)
            rows.append({
                "month_date": new_month,
                "category": cat,
                "rf_forecast": yhat,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> dict:
    panel = build_panel(df)
    if panel.empty:
        logger.warning("RF forecaster: panel is empty, skipping.")
        return {}

    result = fit_and_evaluate(panel)
    future = forecast_future(panel, result)

    config_v1.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_v1.MODEL_DIR / "rf_forecaster.pkl", "wb") as f:
        pickle.dump({
            "model": result.model,
            "feature_cols": result.feature_cols,
            "cat_codes": result.cat_codes,
            "metrics": result.metrics,
        }, f)

    metrics_row = pd.DataFrame([{
        "model": "RandomForest",
        "mae": round(result.metrics["mae"], 2),
        "rmse": round(result.metrics["rmse"], 2),
        "mae_pct_of_mean": round(result.metrics["mae_pct_of_mean"], 1),
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }])
    metrics_row.to_csv(config_v1.DATA_DIR / "forecast_rf_metrics.csv", index=False)
    future.to_csv(config_v1.DATA_DIR / "forecast_rf_future.csv", index=False)

    total = (future.groupby("month_date")["rf_forecast"].sum()
                   .round(1).reset_index())
    total.to_csv(config_v1.DATA_DIR / "forecast_rf_future_total.csv", index=False)

    logger.info("RF forecaster artefacts written to %s", config_v1.DATA_DIR)
    return {"metrics": result.metrics, "future_rows": len(future)}
