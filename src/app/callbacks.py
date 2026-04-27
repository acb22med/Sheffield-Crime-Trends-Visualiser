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
# Trend / forecast figure
# ---------------------------------------------------------------------------
def _trend_fig(df: pd.DataFrame) -> go.Figure:
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

    # Future forecast
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

    fc_path = config_v1.DATA_DIR / "forecast_cv.csv"
    if fc_path.exists():
        fdf = pd.read_csv(fc_path)
        summary = fdf.groupby("model")[["rmse", "mae"]].mean().round(2).reset_index()
        cards.append(html.H4("Forecasting — walk-forward CV average", className="mt-4"))
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

        # Down-sample aggressively for responsiveness when style is scatter.
        if style == "scatter" and len(df) > 8000:
            df = df.sample(8000, random_state=0)

        # Merge cluster labels if requested.
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

    # --- Trend ---
    @app.callback(
        Output("trend-graph", "figure"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
        Input("neighbourhood-filter", "value"),
    )
    def update_trend(start, end, cats, neighbourhoods):
        return _trend_fig(_filtered(start, end, cats, neighbourhoods))

    # --- Category ---
    @app.callback(
        Output("category-graph", "figure"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
        Input("neighbourhood-filter", "value"),
    )
    def update_category(start, end, cats, neighbourhoods):
        return _category_fig(_filtered(start, end, cats, neighbourhoods))

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
