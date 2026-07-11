import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.logging import get_run_logger

from app_data_manager.data_manager import DataManager
from app_data_manager.utils import read_config

project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)

config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
config_data_manager = config["data_manager"]
config_mlflow = config["mlflow"]


@task(
    name="load_retraining_trigger_data",
)
def load_retraining_trigger_data(config_data_manager: dict[str, Any]) -> pd.DataFrame:
    """
    Load the last n points from the retraining trigger table.
    """
    data_manager = DataManager(config_data_manager)
    retraining_trigger_df = data_manager.get_last_n_points(
        n=10, table_name=config_data_manager["retraining_trigger_table_name"]
    )
    return retraining_trigger_df


@task(name="run_training_pipeline")
def run_training_pipeline(env: str = "local", pipeline_name: str = "training"):
    """Prefect task to run the Kedro training pipeline."""
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


def get_last_training_timestamp() -> datetime | None:
    """Get timestamp of the most recently updated model (champion or challenger)."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    )

    client = MlflowClient()
    model_name = config_mlflow["registered_model_name"]

    timestamps = []

    # Check champion
    try:
        champion_alias = config_mlflow["model_aliases"]["production"]  # "champion"
        model_version = client.get_model_version_by_alias(
            name=model_name, alias=champion_alias
        )
        if model_version.last_updated_timestamp is not None:
            timestamps.append(model_version.last_updated_timestamp)
    except Exception:
        pass  # No champion exists

    # Check challenger
    try:
        challenger_alias = config_mlflow["model_aliases"]["candidate"]  # "challenger"
        model_version = client.get_model_version_by_alias(
            name=model_name, alias=challenger_alias
        )
        if model_version.last_updated_timestamp is not None:
            timestamps.append(model_version.last_updated_timestamp)
    except Exception:
        pass  # No challenger exists

    if not timestamps:
        return None
    # maximum of the timestamps
    latest_timestamp = max(timestamps)
    return datetime.fromtimestamp(latest_timestamp / 1000.0)


@task
def should_retrain(retraining_trigger_df: pd.DataFrame) -> bool:
    """
    Check if the retraining trigger df is empty.
    """
    logger = get_run_logger()

    # Case 1: No previous training runs found in MLflow → run initial training
    last_training_timestamp = get_last_training_timestamp()
    if last_training_timestamp is None:
        logger.info("No previous training runs found - running initial training")
        return True

    # Case 2: Empty re-train table → first run, train model
    if retraining_trigger_df.empty:
        logger.info("No retraining triggers found - running initial training")
        return True

    # Get the LATEST row (last one after chronological ordering)
    trigger_row = retraining_trigger_df.iloc[-1]
    trigger_value = int(trigger_row["retraining_trigger"])
    trigger_timestamp = pd.to_datetime(trigger_row["Timestamps"])

    # Case 3: No drift detected in the last trigger → don't train
    if trigger_value == 0:
        logger.info("No data drift detected - skipping training")
        return False

    if trigger_value == 1:
        # Case 4: drift detected after last training → train
        if trigger_timestamp > last_training_timestamp:
            logger.info("Data drift detected - running training")
            return True
        else:
            # Case 5: drift detected older than last training → don't train
            logger.info(
                "Trigger timestamp is older than last training timestamp - skipping training"
            )
            return False

    logger.info("No retraining triggers found - skipping training")
    return False


@flow(name="training_flow", log_prints=True)
def training_flow(config_data_manager: dict[str, Any]):
    logger = get_run_logger()
    logger.info("Starting training flow")
    retraining_trigger_df = load_retraining_trigger_data(config_data_manager)
    retrain_flag = should_retrain(retraining_trigger_df)

    if retrain_flag:
        logger.info("Retraining flag is True - running training pipeline")
        run_training_pipeline(env="local", pipeline_name="training")
    else:
        logger.info("Retraining flag is False - skipping training pipeline")
        pass


if __name__ == "__main__":
    training_flow(config_data_manager=config_data_manager)
