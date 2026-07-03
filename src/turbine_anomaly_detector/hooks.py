import os

import mlflow
from kedro.framework.hooks import hook_impl


class MLFlowHook:
    """Project hooks for MLflow tracking URI setup"""

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """
        Set MLflow tracking URI before pipeline runs.
        Uses environment variable if set, otherwise falls back to localhost:5001
        for local development. In Docker, the environment variable should be set
        to http://mlflow:5001 (internal service name).
        """
        # print("Setting MLflow tracking URI")
        # go to localhost if uri not provided in settings.py
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
        mlflow.set_tracking_uri(tracking_uri)

    # @hook_impl
    # def before_pipeline_run(self, run_params, pipeline, catalog):
    #     """Optional"""
    #     pipeline_name = run_params.get("pipeline_name")
    #
    #     if pipeline_name == "training":
    #         mlflow.set_tracking_uri("http://127.0.0.1:8080")
    #         mlflow.set_experiment("training_experiment")
    #     elif pipeline_name == "inference":
    #         mlflow.set_tracking_uri("http://127.0.0.1:8080")
    #         mlflow.set_experiment("training_experiment")
    #     else:
    #         # default / __default__ pipeline
    #         mlflow.set_tracking_uri("http://127.0.0.1:8080")
