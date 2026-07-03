import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Register this page with Dash - accessible at "/architecture"
dash.register_page(__name__, path="/architecture")

# Asset path - architecture diagram is stored in the assets folder
# Dash automatically serves files from /assets/ at the root URL path
ARCHITECTURE_GIF_PATH = "/assets/app_architecture.gif"


def _li_bold_tools(description, tools_brackets):
    """Return an Li with description and parenthesized tools in bold."""
    return html.Li([description, " ", html.Strong(tools_brackets)])


# Page layout - 2-column layout: Info panel (left) + Architecture diagram (right)
layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left column: Components and tools (4/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Components and tools",
                                    className="arch-section-title",
                                ),
                                html.Ul(
                                    [
                                        _li_bold_tools(
                                            "Reproducible development environment",
                                            "(uv, uv.lock)",
                                        ),
                                        _li_bold_tools(
                                            "Dockerized multi-service architecture",
                                            "(Docker, Docker Compose)",
                                        ),
                                        _li_bold_tools(
                                            "Persistent storage via Docker volumes",
                                            "(Docker volumes)",
                                        ),
                                        _li_bold_tools(
                                            "Modular ML pipelines",
                                            "(Kedro)",
                                        ),
                                        _li_bold_tools(
                                            "Separation of feature engineering, training, inference, monitoring pipelines",
                                            "(Kedro, Prefect)",
                                        ),
                                        _li_bold_tools(
                                            "Code quality automation: pre-commit + formatting/linting",
                                            "(pre-commit, Ruff)",
                                        ),
                                        _li_bold_tools(
                                            "Automated testing and validation",
                                            "(PyTest)",
                                        ),
                                        _li_bold_tools(
                                            "CI pipeline on every commit/PR",
                                            "(GitHub Actions)",
                                        ),
                                        _li_bold_tools(
                                            "CD pipeline build + push + deploy",
                                            "(GitHub Actions, Docker)",
                                        ),
                                        _li_bold_tools(
                                            "Docker image versioning and rollback",
                                            "(Docker Registry)",
                                        ),
                                        _li_bold_tools(
                                            "Cloud production deployment",
                                            "(DigitalOcean, Docker Compose)",
                                        ),
                                        _li_bold_tools(
                                            "Orchestration server for scheduling and triggers",
                                            "(Prefect)",
                                        ),
                                        _li_bold_tools(
                                            "Streaming/batch data ingestion service",
                                            "(Python, SQLite)",
                                        ),
                                        _li_bold_tools(
                                            "SQL-backed storage for history, predictions, and errors",
                                            "(SQLite)",
                                        ),
                                        _li_bold_tools(
                                            "Experiment tracking for runs, metrics, artifacts",
                                            "(MLflow)",
                                        ),
                                        _li_bold_tools(
                                            "Central model registry with versioning and stage transitions",
                                            "(MLflow Model Registry)",
                                        ),
                                        _li_bold_tools(
                                            "Hyperparameter tuning integrated into training",
                                            "(Optuna / scikit-learn, tracked in MLflow)",
                                        ),
                                        _li_bold_tools(
                                            "Champion/Challenger model promotion workflow",
                                            "(MLflow aliases/tags)",
                                        ),
                                        _li_bold_tools(
                                            "Monitoring pipeline for model performance monitoring",
                                            "(Kedro, Prefect)",
                                        ),
                                        _li_bold_tools(
                                            "Automated re-training triggers based on monitoring thresholds",
                                            "(Kedro, Prefect, MLflow)",
                                        ),
                                    ],
                                    className="arch-list",
                                ),
                            ],
                            className="arch-panel",
                        ),
                    ],
                    width=4,
                ),
                # Right column: Architecture Diagram (8/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Architecture Diagram",
                                    className="arch-diagram-title",
                                ),
                                html.Div(
                                    id="architecture-image-container",
                                    children=[
                                        html.Img(
                                            src=ARCHITECTURE_GIF_PATH,
                                            className="arch-diagram-img",
                                            alt="Application Architecture Diagram",
                                        ),
                                    ],
                                ),
                            ],
                            className="arch-diagram-container",
                        ),
                    ],
                    width=8,
                ),
            ],
            className="arch-row",
        ),
    ],
    fluid=True,
)
