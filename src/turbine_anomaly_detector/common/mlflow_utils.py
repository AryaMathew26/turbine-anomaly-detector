from typing import Any

import mlflow
from mlflow.tracking.client import MlflowClient


def load_model_by_alias(registered_model_name: str, alias: str) -> Any:
    """
    Load a model from the MLflow model registry by alias
    Resolves the alias to a model version, then loads that version
    Args:
        registered_model_name (str): The name of the registered model in MLflow
        alias (str): Model alias (e.g., 'champion', 'challenger')
    Returns:
        Any: The loaded MLflow pyfunc model. This model can be used directly for
        predictions as it handles scaling internally (if bundled with scaler)
    """

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(registered_model_name, alias)
    model_uri = f"models:/{registered_model_name}/{model_version.version}"
    return mlflow.pyfunc.load_model(model_uri)
