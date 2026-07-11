import os
import sys
import tomllib
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, get_run_logger, task

# Add src directory to path before imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)

from app_data_manager.data_manager import DataManager  # noqa: E402
from app_data_manager.utils import read_config  # noqa: E402

# Read configuration
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Track last processed timestamp across flow runs
last_timestamp: str | None = None


@task(name="check_new_data", log_prints=True)
def check_new_data() -> tuple[bool, str | None]:
    """
    Check if new data is available by comparing timestamps.
    Returns (has_new_data, latest_timestamp).
    """
    global last_timestamp  # noqa: PLW0602, PLW0603
    logger = get_run_logger()
    data_manager = DataManager(config["data_manager"])

    df = data_manager.get_last_n_points(1, table_name="raw_data")
    latest_timestamp = df.iloc[-1]["Timestamps"]

    if last_timestamp is None or latest_timestamp > last_timestamp:
        logger.info(f"New data found: {latest_timestamp}")
        return True, latest_timestamp

    logger.info(f"No new data (last: {last_timestamp})")
    return False, None


@task(name="run_inference_pipeline", log_prints=True)
def run_inference_pipeline(env: str = "local") -> None:
    """Run the Kedro inference pipeline programmatically."""
    logger = get_run_logger()
    logger.info("Running inference pipeline")
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name="inference")

    logger.info("Inference pipeline completed")


@flow(name="inference_flow", log_prints=True)
def inference_flow(env: str = "local") -> None:
    """
    Check for new data and run inference pipeline if new data is available.
    Scheduling is handled by Prefect's serve with interval.
    """
    global last_timestamp  # noqa: PLW0602, PLW0603
    logger = get_run_logger()
    logger.info("Starting inference flow")

    has_new_data, latest_timestamp = check_new_data()

    if has_new_data:
        run_inference_pipeline(env=env)
        last_timestamp = latest_timestamp
        logger.info("Inference completed")
    else:
        logger.info("Skipping inference - no new data")


if __name__ == "__main__":
    inference_flow()
