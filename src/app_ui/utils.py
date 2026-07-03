import copy
from datetime import datetime
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import html, no_update

import mlflow
from app_data_manager.data_manager import DataManager
from app_data_manager.utils import read_config
from mlflow.tracking import MlflowClient

# Configuration (same as home page: parameters.yml)
project_root = Path(__file__).resolve().parents[2]
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)


def load_prod_data(
    n_data_points: int = 100000, anomaly_error_type: str = "mape"
) -> pd.DataFrame:
    """
    Load last N rows from predictions, errors, anomalies, raw_data; merge on Timestamps.
    Returns only DB columns: Timestamps, predict_power, mape, anomaly, Power (if present).
    """
    data_manager = DataManager(config["data_manager"])
    predictions = data_manager.get_last_n_points(
        n_data_points, table_name="predictions"
    )
    errors = data_manager.get_last_n_points(n_data_points, table_name="errors")
    anomalies = data_manager.get_last_n_points(n_data_points, table_name="anomalies")
    raw_data = data_manager.get_last_n_points(n_data_points, table_name="raw_data")
    df = predictions.copy()
    df = df.merge(
        errors[
            ["Timestamps", anomaly_error_type, f"rolling_{anomaly_error_type}"]
        ].drop_duplicates(subset=["Timestamps"]),
        on="Timestamps",
        how="left",
    )
    df = df.merge(
        anomalies[["Timestamps", "anomaly"]].drop_duplicates(subset=["Timestamps"]),
        on="Timestamps",
        how="left",
    )

    df = df.merge(
        raw_data[["Timestamps", "Power"]].drop_duplicates(subset=["Timestamps"]),
        on="Timestamps",
        how="left",
    )
    df.drop_duplicates(subset=["Timestamps"], inplace=True)
    df["true_value"] = df["Power"] if "Power" in df.columns else df["predict_power"]
    df["prediction"] = df["predict_power"]
    return df.sort_values("Timestamps").reset_index(drop=True)


def create_error_plot(
    df: pd.DataFrame,
    metric_threshold: float,
    anomaly_error_type: str = "mape",
    rolling_window: int = 5,
) -> go.Figure:
    """Plot error column, rolling error, threshold line, and anomaly markers."""

    x_min, x_max = df["Timestamps"].min(), df["Timestamps"].max()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Timestamps"],
            y=df[anomaly_error_type],
            name=anomaly_error_type.upper(),
            mode="lines",
            line=dict(color="#60a5fa", width=1.5),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Timestamps"],
            y=df[f"rolling_{anomaly_error_type}"],
            name=f"{anomaly_error_type.upper()} Rolling ({rolling_window} pts)",
            mode="lines",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[metric_threshold, metric_threshold],
            name=f"Threshold: {metric_threshold:.3f}",
            mode="lines",
            line=dict(color="#111827", width=2.5, dash="dash"),
        )
    )
    anomaly_mask = df["anomaly"] == 1
    df_anom = df.loc[anomaly_mask]
    y_anom = df_anom[f"rolling_{anomaly_error_type}"]
    fig.add_trace(
        go.Scatter(
            x=df_anom["Timestamps"],
            y=y_anom,
            name="Anomaly",
            mode="markers",
            marker=dict(
                color="#dc2626",
                size=10,
                symbol="circle",
                line=dict(width=2, color="#dc2626"),
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=30, b=40),
        height=None,
        showlegend=True,
        xaxis_title="Time",
        yaxis_title="Mean Absolute Percentage Error",
        xaxis=dict(range=[x_min, x_max], gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        yaxis=dict(gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1B1D23", size=12),
    )
    return fig


def create_timeseries_plot(df: pd.DataFrame) -> go.Figure:
    """Plot DB columns only: Power (true) and predict_power. No computed columns."""
    if df.empty or "Timestamps" not in df.columns or "predict_power" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Power",
            annotations=[
                dict(
                    text="No data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return fig
    x_min, x_max = df["Timestamps"].min(), df["Timestamps"].max()
    fig = go.Figure()
    if "Power" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Timestamps"],
                y=df["Power"],
                name="Power",
                mode="lines",
                line=dict(color="#16A34A", width=2.5),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df["Timestamps"],
            y=df["predict_power"],
            name="predict_power",
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=30, b=40),
        height=None,
        showlegend=True,
        xaxis_title="Time",
        yaxis_title="Power",
        xaxis=dict(range=[x_min, x_max], gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        yaxis=dict(gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1B1D23", size=12),
    )
    return fig


def sync_xaxis(relayout_data, current_figure):
    """Copy x-axis range from relayout_data into current_figure. Returns (updated_figure, x_range) or (no_update, no_update)."""
    if not relayout_data or not current_figure:
        return no_update, no_update

    keys = ("xaxis.range[0]", "xaxis.range", "xaxis.autorange")
    if not any(k in relayout_data for k in keys):
        return no_update, no_update

    updated_figure = copy.deepcopy(current_figure)
    if "xaxis" not in updated_figure["layout"]:
        updated_figure["layout"]["xaxis"] = {}

    if "xaxis.autorange" in relayout_data:
        updated_figure["layout"]["xaxis"].pop("range", None)
        updated_figure["layout"]["xaxis"]["autorange"] = True
        return updated_figure, None

    x_range = None
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        x_range = [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
    elif "xaxis.range" in relayout_data and isinstance(
        relayout_data["xaxis.range"], list
    ):
        x_range = relayout_data["xaxis.range"]

    if x_range:
        updated_figure["layout"]["xaxis"]["range"] = x_range
        updated_figure["layout"]["xaxis"]["autorange"] = False
        return updated_figure, x_range

    return no_update, no_update


def get_model_info_by_alias(
    alias: str,
    mlflow_tracking_uri: str,
    model_name: str,
) -> dict | None:
    """Return basic info + test MAE/MAPE for a model version resolved by alias."""
    # 1) Resolve tracking URI (if not provided, uses the local mlflow server)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # 3) Look up model version by alias
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    except Exception:
        return None

    # Require timestamp and run_id; otherwise we can't show useful info
    if model_version.last_updated_timestamp is None or model_version.run_id is None:
        return None

    last_updated = datetime.fromtimestamp(model_version.last_updated_timestamp / 1000.0)

    # 4) Fetch run to read metrics
    try:
        run = client.get_run(model_version.run_id)
    except Exception:
        return None

    metrics = run.data.metrics

    # Helper: given a list of possible metric names, return the first one that
    # exists in `metrics` (or None if none of them are logged in this run).
    def _first_metric(keys: list[str]) -> float | None:
        return next((metrics[k] for k in keys if k in metrics), None)

    test_mae = _first_metric(
        [
            "test_mae",
            "test_MAE",
            "test_mae_err",
            "test_MAE_err",
            "mae_test",
            "MAE_test",
        ]
    )
    test_mape = _first_metric(
        [
            "test_mape",
            "test_MAPE",
            "test_mape_err",
            "test_MAPE_err",
            "mape_test",
            "MAPE_test",
        ]
    )

    return {
        "model_name": model_name,
        "version": model_version.version,
        "last_updated": last_updated,
        "test_mae": test_mae,
        "test_mape": test_mape,
    }


# Model tracking page: format model info for display
def format_last_updated(info):
    if info.get("last_updated") is None:
        return "N/A"
    return info["last_updated"].strftime("%Y-%m-%d %H:%M:%S")


def format_mae(info):
    if info.get("test_mae") is None:
        return "MAE: N/A"
    return f"MAE: {info['test_mae']:.4f}"


def format_mape(info):
    if info.get("test_mape") is None:
        return "MAPE: N/A"
    return f"MAPE: {info['test_mape']:.4f}%"


def create_challenger_info_content(challenger_info):
    """Build HTML for challenger model info card. Returns html.P with message if challenger_info is None."""
    if not challenger_info:
        return html.P(
            "No challenger model found in registry. Make sure MLflow is running and a model is registered with 'challenger' alias.",
            className="mt-empty",
        )
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H6("Model Name", className="mt-label"),
                    html.P(
                        challenger_info.get("model_name", "N/A"), className="mt-value"
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6("Version", className="mt-label"),
                    html.P(
                        f"v{challenger_info['version']}"
                        if challenger_info.get("version")
                        else "N/A",
                        className="mt-value",
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.H6("Last Updated", className="mt-label"),
                    html.P(format_last_updated(challenger_info), className="mt-value"),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6("Test Set Error Metrics", className="mt-label"),
                    html.Div(
                        [
                            html.P(format_mae(challenger_info), className="mt-metric"),
                            html.P(
                                format_mape(challenger_info), className="mt-metric-last"
                            ),
                        ],
                        className="mt-metrics-wrap",
                    ),
                ],
                width=4,
            ),
        ],
        className="g-4",
    )


def create_champion_info_content(champion_info):
    """Build HTML for champion model info card. Returns html.P with message if champion_info is None."""
    if not champion_info:
        return html.P(
            "No champion model found in registry. Make sure MLflow is running and a model is registered in Production stage.",
            className="mt-empty",
        )
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H6("Model Name", className="mt-label"),
                    html.P(
                        champion_info.get("model_name", "N/A"), className="mt-value"
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6("Version", className="mt-label"),
                    html.P(
                        f"v{champion_info['version']}"
                        if champion_info.get("version")
                        else "N/A",
                        className="mt-value",
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.H6("Last Updated", className="mt-label"),
                    html.P(format_last_updated(champion_info), className="mt-value"),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6("Test Set Error Metrics", className="mt-label"),
                    html.Div(
                        [
                            html.P(format_mae(champion_info), className="mt-metric"),
                            html.P(
                                format_mape(champion_info), className="mt-metric-last"
                            ),
                        ],
                        className="mt-metrics-wrap",
                    ),
                ],
                width=4,
            ),
        ],
        className="g-4",
    )
