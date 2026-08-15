import os
import sys
from datetime import timedelta
from pathlib import Path

from app_data_manager.utils import read_config
from prefect_orchestrator.training_flow import training_flow

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))
os.chdir(project_root)


# Read configuration
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Read environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://127.0.0.1:4200/api")


if __name__ == "__main__":
    # Define how often to check for data drift
    check_frequency_minutes = config["training_pipeline"]["train_check_frequency"]
    config_data_manager = config["data_manager"]

    training_flow.serve(
        name="training-deployment",
        interval=timedelta(minutes=check_frequency_minutes),
        parameters={"config_data_manager": config_data_manager},
        limit=1,
    )
