"""Callbacks for the Sheffield Crime Trends Visualiser."""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

import config_v1


# ---------------------------------------------------------------------------
# Helper KPIs & figure builders
# ---------------------------------------------------------------------------
def kpi_card(title: str, value: str, subtitle: str = "") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted small"),
            html.H3(value, className="mb-0"),
            html.Small(subtitle, className="text-muted"),
        ]),
        className="shadow-sm text-center",
    )


def _empty_fig(msg: str) -> go.Figure:
    return go.Figure().update_layout(
        annotations=[{
            "text": msg, "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5, "showarrow": False,
            "font": {"size": 16, "color": "#888"},
        }],
        xaxis={"visible": False}, yaxis={"visible": False},
        paper_bgcolor="#fafafa", plot_bgcolor="#fafafa",
    )


# ---------------------------------------------------------------------------
# Map figures
# ---------------------------------------------------------------------------
def _map_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.density_mapbox(
        df, lat="latitude", lon="longitude", radius=8,
        center={"lat": config_v1.SHEFFIELD_CENTRE[0], "lon": config_v1.SHEFFIELD_CENTRE[1]},
        zoom=config_v1.MAP_DEFAULT_ZOOM, mapbox_style="open-street-map",
        hover_data={"category": True, "month": True, "latitude": False, "longitude": False},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def _map_scatter(df: pd.DataFrame, cluster_col: str | None) -> go.Figure:
    if cluster_col and cluster_col in df:
        color = df[cluster_col].astype(str)
        title = f"Points coloured by {cluster_col}"
    else:
        color = df["category"]
        title = "Points coloured by category"

    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude", color=color,
        center={"lat": config_v1.SHEFFIELD_CENTRE[0], "lon": config_v1.SHEFFIELD_CENTRE[1]},
        zoom=config_v1.MAP_DEFAULT_ZOOM, mapbox_style="open-street-map",
        opacity=0.55, hover_data={"category": True, "month": True},
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title={"text": title, "x": 0.02, "y": 0.98, "font": {"size": 12}},
        legend={"font": {"size": 10}},
    )
    return fig


def _map_choropleth(df: pd.DataFrame) -> go.Figure:
    """Lightweight 'choropleth' built from the grid feature — no external geojson."""
    if "grid_id" not in df or df.empty:
        return _empty_fig("No grid data available")
    agg = (df.groupby(["grid_id", "grid_lat", "grid_lng"])
             .size().rename("count").reset_index())
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lng_min, lng_max = df["longitude"].min(), df["longitude"].max()
    grid = 50
    lat_step = (lat_max - lat_min) / grid
    lng_step = (lng_max - lng_min) / grid
    agg["lat_c"] = lat_min + agg["grid_lat"] * lat_step + lat_step / 2
    agg["lng_c"] = lng_min + agg["grid_lng"] * lng_step + lng_step / 2

    fig = px.scatter_mapbox(
        agg, lat="lat_c", lon="lng_c", size="count", color="count",
        color_continuous_scale="YlOrRd", size_max=22,
        center={"lat": config_v1.SHEFFIELD_CENTRE[0], "lon": config_v1.SHEFFIELD_CENTRE[1]},
        zoom=config_v1.MAP_DEFAULT_ZOOM, mapbox_style="open-street-map",
        hover_data={"count": True, "grid_id": True},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


# ---------------------------------------------------------------------------
# Trend / forecast figure (existing tab — RF overlay added)
# ---------------------------------------------------------------------------
def _load_rf_future_total(active_categories: list[str] | None) -> pd.DataFrame | None:
    """Aggregate the RF per-category forecast to a city-wide total, optionally
    restricted to the categories the user has filtered to.
    """
    rf_per_cat_path = config_v1.DATA_DIR / "forecast_rf_future.csv"
    rf_total_path = config_v1.DATA_DIR / "forecast_rf_future_total.csv"

    if active_categories and rf_per_cat_path.exists():
        per_cat = pd.read_csv(rf_per_cat_path, parse_dates=["month_date"])
        per_cat = per_cat[per_cat["category"].isin(active_categories)]
        if per_cat.empty:
            return None
        return (per_cat.groupby("month_date")["rf_forecast"].sum()
                       .reset_index().rename(columns={"month_date": "date"}))

    if rf_total_path.exists():
        return pd.read_csv(rf_total_path, parse_dates=["month_date"]) \
                 .rename(columns={"month_date": "date"})
    return None


def _trend_fig(df: pd.DataFrame, active_categories: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _empty_fig("No data for current filters")

    hist = df.groupby("month_date").size().rename("count").reset_index()
    fig.add_trace(go.Scatter(
        x=hist["month_date"], y=hist["count"],
        mode="lines+markers", name="Observed", line={"color": "#1f77b4", "width": 3},
    ))

    # Overlay Prophet CI band if available
    prophet_path = config_v1.DATA_DIR / "prophet_full.csv"
    if prophet_path.exists():
        p = pd.read_csv(prophet_path, parse_dates=["date"])
        fig.add_trace(go.Scatter(
            x=p["date"], y=p["yhat_upper"], line={"width": 0}, showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=p["date"], y=p["yhat_lower"], line={"width": 0},
            fill="tonexty", fillcolor="rgba(255,127,14,0.18)",
            name="Prophet 80% CI", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=p["date"], y=p["yhat"], mode="lines",
            line={"color": "#ff7f0e", "dash": "dot"}, name="Prophet fit",
        ))

    # Future forecast — ARIMA & Prophet (city-wide aggregate models)
    fut_path = config_v1.DATA_DIR / "future_forecast.csv"
    if fut_path.exists():
        f = pd.read_csv(fut_path, parse_dates=["date"])
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["arima_forecast"],
            mode="lines+markers", line={"color": "#2ca02c", "dash": "dash"},
            name="ARIMA forecast",
        ))
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["prophet_forecast"],
            mode="lines+markers", line={"color": "#d62728", "dash": "dash"},
            name="Prophet forecast",
        ))

    # Future forecast — Random Forest (panel model summed back to city-wide total)
    rf = _load_rf_future_total(active_categories)
    if rf is not None and not rf.empty:
        fig.add_trace(go.Scatter(
            x=rf["date"], y=rf["rf_forecast"],
            mode="lines+markers",
            line={"color": "#8B2A2A", "dash": "solid", "width": 3},
            marker={"size": 9, "symbol": "diamond"},
            name="Random Forest forecast",
        ))

    fig.update_layout(
        title="Monthly crime volume — observed vs predicted",
        xaxis_title="Month", yaxis_title="Reported crimes",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 40, "r": 20, "t": 50, "b": 20},
    )
    return fig


def _category_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No data for current filters")
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(
        counts.sort_values("count"), y="category", x="count",
        orientation="h", color="count", color_continuous_scale="Blues",
    )
    fig.update_layout(
        title="Crimes by category (after filters)",
        xaxis_title="Count", yaxis_title="",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


# ---------------------------------------------------------------------------
# RF forecast tab — per-category history + forecast
# ---------------------------------------------------------------------------
def _rf_per_category_fig(df_full: pd.DataFrame, category: str) -> go.Figure:
    """History + forecast for a single category."""
    if not category:
        return _empty_fig("Select a category")

    # History: observed counts per month for this category, from the full
    # engineered frame. We don't apply the global category-filter here — the
    # RF tab is self-contained.
    hist = (df_full[df_full["category"] == category]
              .groupby("month_date").size()
              .rename("count").reset_index())

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["month_date"], y=hist["count"],
            mode="lines+markers", name="Observed",
            line={"color": "#1f77b4", "width": 2.5},
        ))

    # Forecast: load the per-category RF forecast and select this category.
    rf_path = config_v1.DATA_DIR / "forecast_rf_future.csv"
    if rf_path.exists():
        rf = pd.read_csv(rf_path, parse_dates=["month_date"])
        rf = rf[rf["category"] == category]

        if not rf.empty:
            # Connect the last observed point to the first forecast point so
            # the lines visually meet rather than leaving a one-month gap.
            if not hist.empty:
                last_obs_date = hist["month_date"].iloc[-1]
                last_obs_count = hist["count"].iloc[-1]
                connector = pd.DataFrame({
                    "month_date": [last_obs_date],
                    "rf_forecast": [last_obs_count],
                })
                rf_plot = pd.concat([connector, rf], ignore_index=True)
            else:
                rf_plot = rf

            fig.add_trace(go.Scatter(
                x=rf_plot["month_date"], y=rf_plot["rf_forecast"],
                mode="lines+markers",
                line={"color": "#8B2A2A", "dash": "dash", "width": 3},
                marker={"size": 9, "symbol": "diamond"},
                name="Random Forest forecast",
            ))

    pretty = category.replace("-", " ").title()
    fig.update_layout(
        title=f"{pretty} — observed history and Random Forest forecast",
        xaxis_title="Month", yaxis_title="Reported crimes",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 40, "r": 20, "t": 50, "b": 20},
    )
    return fig


def _rf_summary_panel(df_full: pd.DataFrame, category: str) -> html.Div:
    """Compact text panel below the chart: predicted next-3-month counts and
    overall held-out MAE."""
    if not category:
        return html.Div()

    rf_path = config_v1.DATA_DIR / "forecast_rf_future.csv"
    metrics_path = config_v1.DATA_DIR / "forecast_rf_metrics.csv"

    blocks: list = []

    if rf_path.exists():
        rf = pd.read_csv(rf_path, parse_dates=["month_date"])
        rf = rf[rf["category"] == category].sort_values("month_date")
        if not rf.empty:
            cards = []
            for _, row in rf.iterrows():
                cards.append(
                    dbc.Col(
                        kpi_card(
                            row["month_date"].strftime("%b %Y"),
                            f"{int(round(row['rf_forecast'])):,}",
                            "predicted incidents",
                        ),
                        md=4,
                    )
                )
            blocks.append(html.H6("Forecast for the next 3 months",
                                  className="text-muted mb-2"))
            blocks.append(dbc.Row(cards, className="mb-3"))

    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        if not m.empty:
            row = m.iloc[0]
            blocks.append(
                html.Small(
                    f"Model held-out MAE: {row['mae']:.1f}  ·  "
                    f"RMSE: {row['rmse']:.1f}  ·  "
                    f"MAE = {row['mae_pct_of_mean']:.1f}% of mean  ·  "
                    f"Train rows: {int(row['train_rows']):,}, "
                    f"test rows: {int(row['test_rows']):,}",
                    className="text-muted",
                )
            )

    if not blocks:
        return html.P(
            "Train the Random Forest forecaster first (scripts/train_models.py).",
            className="text-muted",
        )
    return html.Div(blocks)


# ---------------------------------------------------------------------------
# Metrics panel (HTML, not a figure)
# ---------------------------------------------------------------------------
def _metrics_panel() -> html.Div:
    cards = []

    clf_path = config_v1.DATA_DIR / "classifier_metrics.csv"
    if clf_path.exists():
        cdf = pd.read_csv(clf_path)
        cards.append(html.H4("Supervised classifiers — held-out test set"))
        cards.append(dbc.Table.from_dataframe(
            cdf.round(3), striped=True, bordered=True, hover=True, size="sm"))

    # RF forecaster — headline forecasting result
    rf_path = config_v1.DATA_DIR / "forecast_rf_metrics.csv"
    if rf_path.exists():
        rdf = pd.read_csv(rf_path).round(2)
        cards.append(html.H4("Random Forest forecaster — held-out test set",
                             className="mt-4"))
        cards.append(dbc.Table.from_dataframe(
            rdf, striped=True, bordered=True, hover=True, size="sm"))

    fc_path = config_v1.DATA_DIR / "forecast_cv.csv"
    if fc_path.exists():
        fdf = pd.read_csv(fc_path)
        summary = fdf.groupby("model")[["rmse", "mae"]].mean().round(2).reset_index()
        cards.append(html.H4("ARIMA / Prophet — walk-forward CV average",
                             className="mt-4"))
        cards.append(dbc.Table.from_dataframe(
            summary, striped=True, bordered=True, hover=True, size="sm"))

    sweep_path = config_v1.DATA_DIR / "kmeans_sweep.csv"
    if sweep_path.exists():
        kdf = pd.read_csv(sweep_path).round(3)
        cards.append(html.H4("K-Means sweep", className="mt-4"))
        cards.append(dbc.Table.from_dataframe(
            kdf, striped=True, bordered=True, hover=True, size="sm"))

    if not cards:
        cards = [html.P("Train the models first (scripts/train_models.py).",
                        className="text-muted")]
    return html.Div(cards)


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------
def register_callbacks(app, full_df: pd.DataFrame, cluster_df: pd.DataFrame | None):
    """Wire all reactive callbacks. ``full_df`` is the engineered frame."""

    def _filtered(start, end, cats, neighbourhoods) -> pd.DataFrame:
        df = full_df
        if start:
            df = df[df["month_date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["month_date"] <= pd.to_datetime(end)]
        if cats:
            df = df[df["category"].isin(cats)]
        if neighbourhoods:
            df = df[df["street_name"].isin(neighbourhoods)]
        return df

    # --- KPIs ---
    @app.callback(
        Output("kpi-row", "children"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
        Input("neighbourhood-filter", "value"),
    )
    def update_kpis(start, end, cats, neighbourhoods):
        df = _filtered(start, end, cats, neighbourhoods)
        total = len(df)
        months = df["month_date"].nunique() if total else 0
        avg = round(total / months, 1) if months else 0
        top_cat = (df["category"].value_counts().idxmax().replace("-", " ").title()
                   if total else "—")
        return [
            dbc.Col(kpi_card("Total crimes", f"{total:,}", f"{months} months"), md=3),
            dbc.Col(kpi_card("Avg / month", f"{avg}"), md=3),
            dbc.Col(kpi_card("Top category", top_cat), md=3),
            dbc.Col(kpi_card("Distinct streets",
                             f"{df['street_name'].nunique():,}" if total else "0"), md=3),
        ]

    # --- Map ---
    @app.callback(
        Output("map-graph", "figure"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
        Input("neighbourhood-filter", "value"),
        Input("cluster-toggle", "value"),
        Input("map-style", "value"),
    )
    def update_map(start, end, cats, neighbourhoods, cluster_mode, style):
        df = _filtered(start, end, cats, neighbourhoods)
        if df.empty:
            return _empty_fig("No crimes match the current filters")

        if style == "scatter" and len(df) > 8000:
            df = df.sample(8000, random_state=0)

        if cluster_mode in ("kmeans", "dbscan") and cluster_df is not None:
            merged = df.merge(
                cluster_df[["latitude", "longitude", "month",
                            f"{cluster_mode}_label"]],
                on=["latitude", "longitude", "month"], how="left",
            )
            col = f"{cluster_mode}_label"
        else:
            merged = df
            col = None

        if style == "heatmap":
            return _map_heatmap(merged)
        if style == "choropleth":
            return _map_choropleth(merged)
        return _map_scatter(merged, col)

    # --- Trend (existing time-series tab, with RF overlay) ---
    @app.callback(
        Output("trend-graph", "figure"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
        Input("neighbourhood-filter", "value"),
    )
    def update_trend(start, end, cats, neighbourhoods):
        return _trend_fig(_filtered(start, end, cats, neighbourhoods),
                          active_categories=cats)

    # --- Category breakdown ---
    @app.callback(
        Output("category-graph", "figure"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
        Input("neighbourhood-filter", "value"),
    )
    def update_category(start, end, cats, neighbourhoods):
        return _category_fig(_filtered(start, end, cats, neighbourhoods))

    # --- RF Forecast tab (new) ---
    # Self-contained: only listens to its own dropdown, ignores the side-panel
    # category filter. This keeps it predictable for the user.
    @app.callback(
        Output("rf-forecast-graph", "figure"),
        Input("rf-category-dropdown", "value"),
    )
    def update_rf_forecast(category):
        return _rf_per_category_fig(full_df, category)

    @app.callback(
        Output("rf-summary-panel", "children"),
        Input("rf-category-dropdown", "value"),
    )
    def update_rf_summary(category):
        return _rf_summary_panel(full_df, category)

    # --- Metrics panel ---
    @app.callback(Output("metrics-panel", "children"), Input("cluster-toggle", "value"))
    def _metrics(_):
        return _metrics_panel()

    # --- CSV download ---
    @app.callback(
        Output("download-csv", "data"),
        Input("btn-csv", "n_clicks"),
        State("date-range", "start_date"), State("date-range", "end_date"),
        State("category-filter", "value"), State("neighbourhood-filter", "value"),
        prevent_initial_call=True,
    )
    def download(n_clicks, start, end, cats, neighbourhoods):
        if not n_clicks:
            return no_update
        df = _filtered(start, end, cats, neighbourhoods)
        return dcc.send_data_frame(df.to_csv, "sheffield_crime_filtered.csv", index=False)
