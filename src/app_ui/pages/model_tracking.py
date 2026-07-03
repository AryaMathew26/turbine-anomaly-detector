"""
Model Tracking & Registry Page - MLflow Integration
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from app_data_manager.utils import read_config
from app_ui.utils import (
    create_challenger_info_content,
    create_champion_info_content,
    get_model_info_by_alias,
)

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Read configuration file (contains MLflow model name, etc.)
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Register this page with Dash - accessible at "/model-tracking"
dash.register_page(__name__, path="/model-tracking")

# MLflow configuration
# Get MLflow tracking URI from environment variable or use default
# Using 127.0.0.1 instead of localhost helps with some browser security policies
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
# In production, this should be the external server IP/domain
MLFLOW_UI_URI = os.getenv("MLFLOW_UI_URI", "http://localhost:8080")

# Page layout - Single column layout, stacked vertically, full screen
layout = dbc.Container(
    [
        # pull new info every minute
        dcc.Interval(
            id="champion-info-interval",
            interval=60000,
            n_intervals=0,
        ),
        dcc.Store(id="champion-info-trigger", data=0),
        dcc.Interval(
            id="challenger-info-interval",
            interval=60000,
            n_intervals=0,
        ),
        dcc.Store(id="challenger-info-trigger", data=0),
        dcc.Store(id="champion-status-store", data=None),
        dcc.Store(id="challenger-status-store", data=None),
        dcc.Store(id="last-update-timestamp", data=None),
        html.Div(
            [
                html.H4("MLflow Integration", className="mt-section-title"),
                html.P(
                    "This page provides access to the MLflow Tracking UI, where you can "
                    "view experiment runs and metrics, compare model performance, track hyperparameters, "
                    "manage model versions, and register and promote models. "
                    "Click the buttons below to access different sections of the MLflow Tracking UI.",
                    className="mt-intro",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Refresh model info",
                                        id="refresh-model-info-btn",
                                        color="secondary",
                                        size="lg",
                                        className="mt-btn-full mt-btn-mint",
                                    ),
                                    html.Div(
                                        id="model-info-status",
                                        className="mt-status-wrap mt-status-under-btn",
                                    ),
                                ],
                                className="mt-btn-with-status",
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Open MLflow UI",
                                href=MLFLOW_UI_URI,
                                target="_blank",
                                color="primary",
                                size="lg",
                                className="mt-btn-full",
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Experiments",
                                href=f"{MLFLOW_UI_URI}/#/experiments",
                                target="_blank",
                                color="primary",
                                size="lg",
                                className="mt-btn-full",
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Models",
                                href=f"{MLFLOW_UI_URI}/#/models",
                                target="_blank",
                                color="primary",
                                size="lg",
                                className="mt-btn-full",
                            ),
                            width=3,
                        ),
                    ],
                    className="g-3",
                ),
            ],
            className="mt-card",
        ),
        html.Div(
            [
                html.H5("Champion Model Information", className="mt-card-title"),
                html.Div(
                    id="champion-model-info",
                    children=[
                        html.Div(
                            "Loading champion model information...",
                            className="mt-loading",
                        ),
                    ],
                ),
            ],
            className="mt-card",
        ),
        html.Div(
            [
                html.H5("Challenger Model Information", className="mt-card-title"),
                html.Div(
                    id="challenger-model-info",
                    children=[
                        html.Div(
                            "Loading challenger model information...",
                            className="mt-loading",
                        ),
                    ],
                ),
            ],
            className="mt-card",
        ),
    ],
    fluid=True,
    className="mt-page-container",
)


# --- Callback: Refresh button -> trigger stores ---
# When "Refresh model info" is clicked, we bump both trigger stores so the champion
# and challenger info callbacks re-run and refetch from MLflow.
@callback(
    [
        Output("champion-info-trigger", "data"),
        Output("challenger-info-trigger", "data"),
    ],
    Input("refresh-model-info-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_model_info(n_clicks):
    """On Refresh button click, update both trigger stores so champion and challenger callbacks run."""
    if n_clicks is None:
        return dash.no_update, dash.no_update
    return n_clicks, n_clicks


# --- Callback: Champion model info ---
# Runs on page load (interval n_intervals), every 60s (interval), or when refresh trigger updates.
# Fetches model with "champion" alias from MLflow, fills the champion card and writes status + timestamp
# for the green/red status light.
@callback(
    [
        Output("champion-model-info", "children"),
        Output("champion-status-store", "data"),
        Output("last-update-timestamp", "data"),
    ],
    [
        Input("champion-info-interval", "n_intervals"),
        Input("champion-info-trigger", "data"),
    ],
)
def update_champion_info(n_intervals, trigger):
    """
    reads config and displays the model with "champion" alias info
    """
    current_config = read_config(parameters_path)
    model_name = current_config["mlflow"]["registered_model_name"]
    champion_info = get_model_info_by_alias(
        "champion", MLFLOW_TRACKING_URI, model_name=model_name
    )
    status = "ok" if champion_info else "no_model"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return create_champion_info_content(champion_info), status, ts


# --- Callback: Challenger model info ---
# Runs on page load, every 60s, or when refresh trigger updates. Fetches model with "challenger"
# alias from MLflow and fills the challenger card; writes status for the status light.
@callback(
    [
        Output("challenger-model-info", "children"),
        Output("challenger-status-store", "data"),
    ],
    [
        Input("challenger-info-interval", "n_intervals"),
        Input("challenger-info-trigger", "data"),
    ],
)
def update_challenger_info(n_intervals, trigger):
    current_config = read_config(parameters_path)
    model_name = current_config["mlflow"]["registered_model_name"]
    challenger_info = get_model_info_by_alias(
        "candidate", MLFLOW_TRACKING_URI, model_name=model_name
    )
    status = "ok" if challenger_info else "no_model"
    return create_challenger_info_content(challenger_info), status


# --- Callback: Status light under Refresh button ---
# Reads champion/challenger status and last-update timestamp; shows green dot when both models
# are present, red dot when champion or challenger is missing; displays "Updated at <time>".
@callback(
    Output("model-info-status", "children"),
    [
        Input("champion-status-store", "data"),
        Input("challenger-status-store", "data"),
        Input("last-update-timestamp", "data"),
    ],
)
def update_status_light(champion_status, challenger_status, last_ts):
    """Show green light when both models found, red when either missing. Display last update time."""
    if last_ts is None:
        return html.Span("—", className="mt-status-text")
    both_ok = champion_status == "ok" and challenger_status == "ok"
    # Red light: champion or challenger is missing in the registry; green when both are present.
    light_class = (
        "mt-status-light mt-status-light--green"
        if both_ok
        else "mt-status-light mt-status-light--red"
    )
    return html.Div(
        [
            html.Span(className=light_class),
            html.Span(f"Updated at {last_ts}", className="mt-status-text"),
        ],
        className="mt-status-inner",
    )
