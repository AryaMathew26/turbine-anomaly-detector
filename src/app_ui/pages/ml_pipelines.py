import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Register this page with Dash - accessible at "/ml-pipelines"
dash.register_page(__name__, path="/ml-pipelines")

# Kedro-Viz configuration
# Get Kedro-Viz URI from environment variable or use default
# Kedro-Viz is a separate service that visualizes Kedro pipelines
# Default: http://localhost:4141 (local Kedro-Viz server)
# Can be configured via KEDRO_VIZ_URI environment variable
KEDRO_VIZ_URI = os.getenv("KEDRO_VIZ_URI", "http://localhost:4141")


# Page layout - 2-column layout: Info panel (left) + Visualization (right)
layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left column: Pipeline Information Panel (3/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5("Pipeline Details", className="mp-panel-title"),
                                html.Div(
                                    [
                                        html.H6(
                                            "Feature Engineering Pipeline",
                                            className="mp-section-title",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Standardizes column names and removes irrelevant columns"
                                                ),
                                                html.Li(
                                                    "Detects and handles outliers using difference-based methods"
                                                ),
                                                html.Li(
                                                    "Applies signal smoothing to reduce noise"
                                                ),
                                                html.Li(
                                                    "Creates lag features to capture temporal dependencies"
                                                ),
                                                html.Li(
                                                    "Generates rolling statistical features (mean, std, min, max)"
                                                ),
                                                html.Li(
                                                    "Separates features from target variable (power output)"
                                                ),
                                                html.Li(
                                                    "Ensures consistent transformations across training and production"
                                                ),
                                            ],
                                            className="mp-section-list",
                                        ),
                                        html.H6(
                                            "Training Pipeline",
                                            className="mp-section-title",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Splits data into training and test sets"
                                                ),
                                                html.Li(
                                                    "Performs hyperparameter tuning using cross-validation"
                                                ),
                                                html.Li(
                                                    "Trains the best model on full training set"
                                                ),
                                                html.Li(
                                                    "Evaluates model performance on test set"
                                                ),
                                                html.Li(
                                                    "Logs results, metrics, and model to MLflow"
                                                ),
                                                html.Li(
                                                    "Registers model in MLflow model registry"
                                                ),
                                                html.Li(
                                                    "Validates model as challenger candidate"
                                                ),
                                            ],
                                            className="mp-section-list",
                                        ),
                                        html.H6(
                                            "Inference Pipeline",
                                            className="mp-section-title",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Loads champion model from MLflow registry"
                                                ),
                                                html.Li(
                                                    "Applies feature engineering transformations"
                                                ),
                                                html.Li(
                                                    "Generates predictions on production data"
                                                ),
                                                html.Li(
                                                    "Computes raw and rolling prediction MAPE metric"
                                                ),
                                                html.Li(
                                                    "Detects anomalies of rolling metric is above the threshold"
                                                ),
                                                html.Li(
                                                    "Saves predictions, errors and anomalies to the database"
                                                ),
                                            ],
                                            className="mp-section-list",
                                        ),
                                    ]
                                ),
                            ],
                            className="mp-panel",
                        ),
                    ],
                    width=3,
                ),
                # Right column: Kedro-Viz Visualization (9/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "ML Pipeline Visualization",
                                    className="mp-viz-title",
                                ),
                                # Kedro-Viz iframe - embeds visualization in an iframe
                                html.Iframe(
                                    src=KEDRO_VIZ_URI,
                                    className="mp-viz-iframe",
                                    allow="fullscreen",
                                ),
                            ],
                            className="mp-viz-container",
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
    className="mp-page-container",
)
