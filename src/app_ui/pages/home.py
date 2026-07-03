import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, callback_context, dcc, html

from src.app_data_manager.utils import read_config
from src.app_ui.utils import (
    create_error_plot,
    create_timeseries_plot,
    load_prod_data,
    sync_xaxis,
)

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Register this page with Dash - makes it accessible at the root URL "/"
dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Control Panel Card
                        html.Div(
                            [
                                html.H4(
                                    "Control Panel",
                                    style={"color": "#222", "marginBottom": "16px"},
                                ),
                                html.Label(
                                    "Lookback Datapoints",
                                    style={"color": "#222", "marginBottom": "8px"},
                                ),
                                dcc.Input(
                                    id="lookback-datapoints",
                                    type="number",
                                    min=1,
                                    step=1,
                                    value=144,
                                    style={
                                        "marginBottom": "16px",
                                        "width": "100%",
                                        "padding": "8px",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #e0e0e0",
                                "marginBottom": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.H5(
                                    "ML Solution Overview",
                                    style={"color": "#222", "marginBottom": "12px"},
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            (
                                                "The ML solution monitors wind turbine health by "
                                                "predicting generated power and comparing it with "
                                                "the actual power output."
                                            ),
                                            style={"marginBottom": "16px"},
                                        ),
                                        html.Li(
                                            (
                                                "The ML model (CatBoost) is trained on data from "
                                                "normal operating conditions. When the rolling "
                                                "MAE/MAPE between predicted and actual power "
                                                "exceeds a predefined threshold continuously for "
                                                "6 hours, the turbine state is classified as an "
                                                "anomaly."
                                            ),
                                            style={"marginBottom": "16px"},
                                        ),
                                        html.Li(
                                            (
                                                "The model is designed to detect turbine anomaly "
                                                "states 2–4 weeks before a failure occurs, "
                                                "resulting in significant economic benefits "
                                                "through reduced downtime."
                                            ),
                                            style={"marginBottom": "16px"},
                                        ),
                                        html.Li(
                                            (
                                                "Model inference runs every 30 seconds for "
                                                "visualization purposes using data unseen during "
                                                "training. In practice, each data point "
                                                "represents a 10-minute measurement."
                                            ),
                                            style={"marginBottom": "16px"},
                                        ),
                                        html.Li(
                                            html.Span(
                                                (
                                                    "To explore the solution architecture, ML "
                                                    "pipelines, and model tracking and registry "
                                                    "details, use the Menu."
                                                ),
                                                style={"fontStyle": "italic"},
                                            ),
                                        ),
                                    ],
                                    style={
                                        "color": "#444",
                                        "fontSize": "14px",
                                        "lineHeight": "1.6",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #e0e0e0",
                            },
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        # automatically trigger a callback function every 5000 milliseconds
                        dcc.Interval(
                            id="auto-refresh-interval",
                            interval=5000,
                            n_intervals=0,
                        ),
                        # share the x-axis between the plots
                        dcc.Store(id="xaxis-range-store", data=None),
                        html.H5(
                            "Anomaly Indicator – ML Model Error",
                            style={"marginBottom": "10px", "color": "#222"},
                        ),
                        dcc.Graph(
                            id="error-plot",
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "8px",
                                "height": "calc(50vh - 100px)",
                                "minHeight": "400px",
                                "marginBottom": "20px",
                            },
                        ),
                        html.H5(
                            "Power: Predictions vs True Values",
                            style={"marginBottom": "10px", "color": "#222"},
                        ),
                        dcc.Graph(
                            id="time-series-plot",
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "8px",
                                "height": "calc(50vh - 100px)",
                                "minHeight": "400px",
                            },
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
    style={
        "paddingTop": "20px",
        "paddingLeft": "5px",
        "paddingRight": "5px",
        "height": "calc(100vh - 60px)",
    },
)


@callback(
    [Output("error-plot", "figure"), Output("time-series-plot", "figure")],
    [
        Input("lookback-datapoints", "value"),
        Input("auto-refresh-interval", "n_intervals"),
    ],
)
def update_plots(lookback_datapoints, n_intervals):
    if lookback_datapoints and lookback_datapoints >= 1:
        n_points = lookback_datapoints
    else:
        n_points = 144
    # read anomaly threshold from the config file
    metric_threshold = config["inference_pipeline"]["anomaly_threshold"]
    # read the predictions, errors and anomalies merged table for n_points
    df = load_prod_data(n_points)
    # create the plots
    fig1 = create_error_plot(df, metric_threshold)
    fig2 = create_timeseries_plot(df)
    return fig1, fig2


@callback(
    [
        Output("error-plot", "figure", allow_duplicate=True),
        Output("time-series-plot", "figure", allow_duplicate=True),
        Output("xaxis-range-store", "data", allow_duplicate=True),
    ],
    [
        Input("error-plot", "relayoutData"),
        Input("time-series-plot", "relayoutData"),
    ],
    [
        State("error-plot", "figure"),
        State("time-series-plot", "figure"),
    ],
    prevent_initial_call=True,
)
def sync_plots_xaxis(
    error_relayout_data, timeseries_relayout_data, error_figure, timeseries_figure
):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "error-plot":
        updated_figure, x_range = sync_xaxis(error_relayout_data, timeseries_figure)
        return dash.no_update, updated_figure, x_range
    if trigger_id == "time-series-plot":
        updated_figure, x_range = sync_xaxis(timeseries_relayout_data, error_figure)
        return updated_figure, dash.no_update, x_range
    return dash.no_update, dash.no_update, dash.no_update
