import os
import sys
import time
import tomllib
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Add src directory to path before imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)

from app_data_manager.data_manager import DataManager  # noqa: E402
from app_data_manager.utils import read_config  # noqa: E402


def run_inference_pipeline(env: str = "local") -> None:
    """Run the Kedro inference pipeline programmatically."""
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name="inference")


def run_inference_real_time(env: str = "local") -> None:
    """
    Continuously check for new data and run inference pipeline when new data is available.
    Configuration is read from parameters.yml (inference_pipeline section).
    """
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config["data_manager"])
    inference_frequency = config["inference_pipeline"]["inference_frequency"]
    last_timestamp: str | None = None

    while True:
        try:
            df = data_manager.get_last_n_points(1, table_name="raw_data")
            latest_timestamp = df.iloc[-1]["Timestamps"]

            if last_timestamp is None or latest_timestamp > last_timestamp:
                run_inference_pipeline(env=env)
                last_timestamp = latest_timestamp

            time.sleep(inference_frequency)
        except Exception as e:
            time.sleep(inference_frequency)
            raise e


if __name__ == "__main__":
    run_inference_real_time()
