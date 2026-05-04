"""Time-series forecasting for Sheffield monthly crime counts.

Two approaches:

* **ARIMA** (via ``statsmodels``) — we run an ADF stationarity test, choose ``d``
  automatically, then grid-search over a compact ``(p,q)`` list and keep the model
  with the lowest AIC.
* **Prophet** — Facebook's additive model with yearly seasonality.

Both are evaluated with a **walk-forward cross-validation** that mirrors production
forecasting: expanding training window, fixed step size, fixed horizon.
"""
from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Silence statsmodels / prophet chatter — we log the things we care about ourselves.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config_v1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Stationarity helper
# ---------------------------------------------------------------------------
def adf_test(series: pd.Series) -> Dict[str, float]:
    from statsmodels.tsa.stattools import adfuller
    stat, p, *_ = adfuller(series.dropna())
    return {"adf_stat": float(stat), "p_value": float(p)}


def choose_d(series: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """Return the smallest ``d`` (0..max_d) for which the ADF test rejects H0."""
    s = series.copy()
    for d in range(max_d + 1):
        res = adf_test(s)
        if res["p_value"] < alpha:
            return d
        s = s.diff().dropna()
    return max_d


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------
@dataclass
class ARIMAResult:
    order: Tuple[int, int, int]
    aic: float
    model: object
    forecast: pd.Series


def fit_arima(train: pd.Series,
              horizon: int,
              orders: List[Tuple[int, int, int]] = None) -> ARIMAResult:
    from statsmodels.tsa.arima.model import ARIMA

    orders = orders or config_v1.FORECAST.arima_order_search
    d = choose_d(train)
    candidates = [(p, d, q) for (p, _, q) in orders]

    best: ARIMAResult | None = None
    for order in candidates:
        try:
            model = ARIMA(train, order=order).fit()
        except Exception as exc:          # pragma: no cover - numerical edge cases
            logger.debug("ARIMA%s failed: %s", order, exc)
            continue
        if best is None or model.aic < best.aic:
            fc = model.forecast(steps=horizon)
            best = ARIMAResult(order=order, aic=float(model.aic),
                               model=model, forecast=fc)

    if best is None:
        raise RuntimeError("No ARIMA order converged on this training window")
    logger.info("Best ARIMA order %s (AIC=%.2f)", best.order, best.aic)
    return best


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
@dataclass
class ProphetResult:
    model: object
    forecast: pd.DataFrame           # prophet's native frame
    yhat: pd.Series


def fit_prophet(train: pd.Series, horizon: int) -> ProphetResult:
    from prophet import Prophet

    df = train.reset_index()
    df.columns = ["ds", "y"]

    # yearly_seasonality=True defaults to Fourier order 10, which severely
    # overfits on <2 years of monthly data. We disable it and re-add with
    # order 3 — enough to capture summer/winter patterns without oscillating.
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=config_v1.FORECAST.prophet_seasonality_mode,
        changepoint_prior_scale=0.05,
    )
    m.add_seasonality(name="yearly", period=365.25, fourier_order=3)
    m.fit(df)

    future = m.make_future_dataframe(periods=horizon, freq="MS")
    fc = m.predict(future)

    # Crime counts cannot be negative — clip the full forecast frame.
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        fc[col] = fc[col].clip(lower=0)

    yhat = fc.set_index("ds")["yhat"].iloc[-horizon:]
    return ProphetResult(model=m, forecast=fc, yhat=yhat)


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------
def walk_forward(series: pd.Series,
                 initial: int = config_v1.FORECAST.cv_initial_months,
                 step: int = config_v1.FORECAST.cv_step_months,
                 horizon: int = config_v1.FORECAST.test_horizon_months,
                 ) -> pd.DataFrame:
    """Return a CV metric table for both ARIMA and Prophet."""
    records = []
    n = len(series)
    cutoffs = list(range(initial, n - horizon + 1, step))
    logger.info("Walk-forward CV: %d folds (initial=%d, step=%d, horizon=%d)",
                len(cutoffs), initial, step, horizon)

    for idx, cutoff in enumerate(cutoffs, start=1):
        train = series.iloc[:cutoff]
        test = series.iloc[cutoff:cutoff + horizon]

        # ARIMA
        try:
            a = fit_arima(train, horizon)
            records.append({
                "fold": idx, "model": "ARIMA", "cutoff": str(train.index[-1].date()),
                "rmse": rmse(test.to_numpy(), a.forecast.to_numpy()),
                "mae":  mae(test.to_numpy(), a.forecast.to_numpy()),
                "order": str(a.order),
            })
        except Exception as exc:
            logger.warning("Fold %d ARIMA failed: %s", idx, exc)

        # Prophet
        try:
            p = fit_prophet(train, horizon)
            records.append({
                "fold": idx, "model": "Prophet", "cutoff": str(train.index[-1].date()),
                "rmse": rmse(test.to_numpy(), p.yhat.to_numpy()),
                "mae":  mae(test.to_numpy(), p.yhat.to_numpy()),
                "order": None,
            })
        except Exception as exc:
            logger.warning("Fold %d Prophet failed: %s", idx, exc)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(monthly: pd.DataFrame) -> pd.DataFrame:
    """Fit final ARIMA/Prophet on all data, save them, return CV metrics."""
    series = (
        monthly.set_index("month_date")["crime_count"]
        .astype(float)
        .sort_index()
        .asfreq("MS", fill_value=0)   # fills any missing months; also sets .freq correctly
    )

    cv = walk_forward(series)
    cv.to_csv(config_v1.DATA_DIR / "forecast_cv.csv", index=False)

    horizon = config_v1.FORECAST.test_horizon_months
    arima = fit_arima(series, horizon)
    prophet = fit_prophet(series, horizon)

    # Persist forecasts for the Dash app
    out = pd.DataFrame({
        "date": pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1),
                              periods=horizon, freq="MS"),
        "arima_forecast": arima.forecast.to_numpy(),
        "prophet_forecast": prophet.yhat.to_numpy(),
    })
    out.to_csv(config_v1.DATA_DIR / "future_forecast.csv", index=False)

    # History with Prophet CIs (handy for the app's band)
    ci = prophet.forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    ci.to_csv(config_v1.DATA_DIR / "prophet_full.csv", index=False)

    with open(config_v1.MODEL_DIR / "forecast_models.pkl", "wb") as f:
        pickle.dump({"arima_model": arima.model, "prophet_model": prophet.model,
                     "arima_order": arima.order}, f)

    summary = (cv.groupby("model")[["rmse", "mae"]].mean().round(2)
                 .reset_index().rename(columns={"rmse": "rmse_mean", "mae": "mae_mean"}))
    logger.info("Forecast CV summary:\n%s", summary.to_string(index=False))
    return cv
