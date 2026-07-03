from typing import Any

import numpy as np
import pandas as pd

from app_data_manager.data_manager import DataManager
from turbine_anomaly_detector.common.mlflow_utils import load_model_by_alias


def load_champion_model(mlflow_params: dict[str, Any]) -> None:
    """
    Validates candidate against production and promote if better (lower MAPE)
    """
    registered_model_name = mlflow_params["registered_model_name"]
    production_alias = mlflow_params["model_aliases"]["production"]

    # load champion model and predict on current test set, compute mape

    return load_model_by_alias(registered_model_name, alias=production_alias)


def predict(
    features_data: pd.DataFrame, champion_model: Any, prediction_column_name: list[str]
) -> pd.DataFrame:
    """_summary_

    Args:
        features_data (pd.DataFrame): _description_
        champion_model (Any): _description_
        prediction_column_name (list[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    predictions = champion_model.predict(features_data)
    # print("shape of pred: ", predictions.shape)
    # print("Pred: ", predictions)
    return pd.DataFrame(predictions, columns=pd.Index(prediction_column_name))


def compute_model_errors(
    y_pred: pd.Series, y_true: pd.Series, anomaly_error_type: str
) -> pd.DataFrame:
    """
    Compute MAPE metric from predictions and target data.

    Parameters
    ----------
    y_pred : pd.Series
        Predicted values.
    target_data : pd.Series
        Ground truth target values.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'mae', 'rmse', and 'mape' metrics.
    """
    y_true = y_true.values.ravel()
    y_pred = y_pred.values.ravel()
    if anomaly_error_type == "mape":
        error = np.abs(y_true - y_pred) / (y_true + 1e-8) * 100
        error_column_name = "mape"

    # print("Error: ", error)
    # print("Mean Error: ", error.mean())
    return pd.DataFrame(error, columns=pd.Index([error_column_name]))


def compute_rolling_error(df_error: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """
    Smooth a metric using a rolling window.

    Parameters
    ----------
    df_error : pd.DataFrame
        DataFrame containing the error metric.
    rolling_window : int
        Window size for the rolling window.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the smoothed error metric.
    """
    error_name = df_error.columns[0]
    df_error[f"rolling_{error_name}"] = (
        df_error[error_name].rolling(window=rolling_window).median().bfill()
    )
    return df_error


def detect_anomaly(
    df_error: pd.DataFrame, threshold: float, anomaly_error_type: str
) -> pd.DataFrame:
    """
    Detect anomalies using a threshold.
    """
    df_anomaly = pd.DataFrame(
        {"anomaly": (df_error[f"rolling_{anomaly_error_type}"] > threshold).astype(int)}
    )
    # print("Anomaly: ", df_anomaly)
    return df_anomaly


def save_predictions_to_db(
    predictions: pd.DataFrame,
    predictions_column_names: list[str],
    db_table_name: str,
    data_timestamps: pd.DataFrame,
    data_manager_config: dict[str, Any],
) -> None:
    """"""
    # Initialise DataManager
    data_manager = DataManager(data_manager_config)
    # Convert input timestamps tp pandas datetime
    timestamps = pd.to_datetime(data_timestamps["Timestamps"])

    # Normalise timestamps to string format expected by the database
    # This works for both Series-like inputs and single Timestamp values
    timestamps_str = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    predictions["Timestamps"] = timestamps_str

    # Create predictions Dataframe
    predictions_df = pd.DataFrame(
        {
            "Timestamps": timestamps_str,
            **{
                col: predictions[col].values.ravel() for col in predictions_column_names
            },
        }
    )

    # save predictions table
    data_manager.insert_data_to_db(new_data=predictions_df, table_name=db_table_name)
