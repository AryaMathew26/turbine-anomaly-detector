import os
import sys
from datetime import timedelta
from pathlib import Path

from app_data_manager.utils import read_config
from prefect_orchestrator.inference_flow import inference_flow

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
    # Define how often to run inference (in seconds from config)
    inference_frequency_seconds = config["inference_pipeline"]["inference_frequency"]

    inference_flow.serve(
        name="inference-deployment",
        interval=timedelta(seconds=inference_frequency_seconds),
    )
