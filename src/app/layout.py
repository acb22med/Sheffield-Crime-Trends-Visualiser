"""Layout for the Sheffield Crime Trends Visualiser Dash app."""
from __future__ import annotations

from datetime import date

import dash_bootstrap_components as dbc
from dash import dcc, html

import config_v1


def _filter_card(cats: list, months: list, neighbourhoods: list) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H5("Filters", className="card-title mb-3"),

            html.Label("Date range", className="fw-bold"),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=months[0],
                max_date_allowed=months[-1],
                start_date=months[0],
                end_date=months[-1],
                display_format="YYYY-MM",
                className="w-100 mb-3",
            ),

            html.Label("Offence type(s)", className="fw-bold"),
            dcc.Dropdown(
                id="category-filter",
                options=[{"label": c.replace("-", " ").title(), "value": c} for c in cats],
                value=cats,
                multi=True,
                className="mb-3",
            ),

            html.Label("Neighbourhood / street", className="fw-bold"),
            dcc.Dropdown(
                id="neighbourhood-filter",
                options=[{"label": n, "value": n} for n in neighbourhoods],
                value=[],
                multi=True,
                placeholder="All neighbourhoods",
                className="mb-3",
            ),

            html.Label("Cluster overlay", className="fw-bold"),
            dcc.RadioItems(
                id="cluster-toggle",
                options=[
                    {"label": " None", "value": "none"},
                    {"label": " K-Means", "value": "kmeans"},
                    {"label": " DBSCAN", "value": "dbscan"},
                ],
                value="none",
                labelStyle={"display": "block"},
                className="mb-3",
            ),

            html.Label("Map view", className="fw-bold"),
            dcc.RadioItems(
                id="map-style",
                options=[
                    {"label": " Heatmap", "value": "heatmap"},
                    {"label": " Choropleth (grid)", "value": "choropleth"},
                    {"label": " Scatter", "value": "scatter"},
                ],
                value="heatmap",
                labelStyle={"display": "block"},
                className="mb-3",
            ),

            html.Hr(),
            html.Label("Exports", className="fw-bold"),
            dbc.Button("Download CSV", id="btn-csv", color="primary",
                       size="sm", className="me-2"),
            dcc.Download(id="download-csv"),
        ]),
        className="shadow-sm h-100",
    )


def _rf_tab_content(cats: list) -> html.Div:
    """Layout for the new Random Forest forecast tab.

    The tab is self-contained: its own category dropdown (does NOT affect the
    rest of the dashboard), a per-category history-plus-forecast chart, and a
    small results panel beneath. We default to the highest-volume category so
    the chart is never empty on first open.
    """
    default_cat = "anti-social-behaviour" if "anti-social-behaviour" in cats else cats[0]
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Crime category", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id="rf-category-dropdown",
                                options=[
                                    {"label": c.replace("-", " ").title(), "value": c}
                                    for c in cats
                                ],
                                value=default_cat,
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Small(
                                    "Random Forest regressor on the (category, month) panel. "
                                    "Features: lag-1, lag-2, lag-3, lag-6 of the per-category count, "
                                    "a 3-month rolling mean, calendar terms, and the encoded category. "
                                    "Forecast is recursive: each predicted month feeds into the lag "
                                    "features of the next.",
                                    className="text-muted",
                                ),
                            ],
                            className="pt-2",
                        ),
                        md=8,
                    ),
                ],
                className="mb-2",
            ),
            dcc.Graph(id="rf-forecast-graph", style={"height": "440px"}),
            html.Hr(),
            html.Div(id="rf-summary-panel", className="p-2"),
        ],
        className="p-3",
    )


def build_layout(cats, months, neighbourhoods) -> dbc.Container:
    return dbc.Container(
        fluid=True,
        className="py-3",
        children=[
            # Header
            dbc.Row([
                dbc.Col([
                    html.H2("Sheffield Crime Trends Visualiser", className="mb-0"),
                    html.P(
                        "Street-level crime in Sheffield · data.police.uk · "
                        "clustering, forecasting & classification",
                        className="text-muted",
                    ),
                ], md=9),
                # dbc.Col(
                #     html.Div([
                #         html.Small("Built by ", className="text-muted"),
                #         html.A("Mohammed", href="https://github.com/",
                #                target="_blank"),
                #     ], className="text-end pt-3"),
                #     md=3,
                # ),
            ], className="mb-3"),

            # KPI row
            dbc.Row(id="kpi-row", className="mb-3"),

            # Main content
            dbc.Row([
                dbc.Col(_filter_card(cats, months, neighbourhoods), md=3, className="mb-3"),
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            dcc.Graph(id="map-graph", style={"height": "620px"}),
                            label="Map",
                        ),
                        dbc.Tab(
                            dcc.Graph(id="trend-graph", style={"height": "620px"}),
                            label="Time-series & Forecast",
                        ),
                        dbc.Tab(
                            dcc.Graph(id="category-graph", style={"height": "620px"}),
                            label="Category breakdown",
                        ),
                        dbc.Tab(
                            _rf_tab_content(cats),
                            label="RF Forecast",
                        ),
                        dbc.Tab(
                            html.Div(id="metrics-panel", className="p-3"),
                            label="Model metrics",
                        ),
                    ]),
                ], md=9),
            ]),

            # Footer
            html.Hr(),
            html.Footer(
                html.Small(
                    "Data: UK Police open API · Sheffield polygon defined in config.py · "
                    "Project © Mohamed",
                    className="text-muted",
                ),
                className="text-center py-2",
            ),
        ],
    )
