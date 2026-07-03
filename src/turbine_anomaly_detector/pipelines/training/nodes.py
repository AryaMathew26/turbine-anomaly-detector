from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models.signature import infer_signature
from mlflow.tracking.client import MlflowClient
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.preprocessing import StandardScaler

from turbine_anomaly_detector.common.metrics import compute_metrics
from turbine_anomaly_detector.common.mlflow_utils import load_model_by_alias

from .utils import MLModelWrapper, eval_model, objective


def train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Explicit temporal train/ val/ test split using index boundaries
    """
    test_idx = int(features.shape[0] * (1 - params["test_fraction"]))
    x_train, x_test = features.iloc[:test_idx].copy(), features.iloc[test_idx:].copy()
    y_train, y_test = target.iloc[:test_idx].copy(), target.iloc[test_idx:].copy()
    # print('x_train', x_train)

    return x_train, y_train, x_test, y_test


# def tune_hyperparameters(
#     x_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
# ) -> dict[str, Any]:

#     seed = 42
#     # 1) Create the study and optimize CV MAE via objective()
#     study = optuna.create_study(
#         direction="minimize",
#         sampler=optuna.samplers.TPESampler(seed=seed),
#     )
#     study.optimize(
#         lambda trial: objective(trial, x_train, y_train, params),
#         n_trials=params["optuna_search"]["n_trials"],
#     )
#     # 2) Get the best parameters
#     best_params = study.best_params

#     # 3) Refit the best model params
#     model_name = params["optuna_search"]["model"]
#     cv_results = eval_model(
#         x_train=x_train,
#         y_train=y_train,
#         n_splits=params["n_folds"],
#         model_name=model_name,
#         model_params=best_params,
#         seed=seed
#     )

#     print(cv_results)
#     print(best_params)

#     return {
#         "best_params": best_params,
#         "cv_metrics": {
#             "cv_mae": cv_results["cv_mae"],
#             "cv_rmse": cv_results["cv_rmse"],
#             "cv_mape": cv_results["cv_mape"],
#         },
#     }


def tune_hyperparameters(
    x_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
) -> dict[str, Any]:
    """
    Train a model by optimizing hyperparameters using Optuna.
    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values.
    params : Dict[str, Any]
        Configuration dictionary containing:
        - 'optuna_search': dict with keys:
            - 'n_trials': int, number of optimization trials to run
            - 'model': str, model name (e.g., 'CatBoost', 'RF')
            - 'model_params': dict, model-specific parameter search spaces
        - 'n_folds': int, number of CV folds (default: 3)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'best_params': Dict[str, Any], best hyperparameters found during optimization
        - 'cv_results': Dict[str, Any], cross-validation results from best model:
            - 'cv_mae': float, average MAE over CV folds
            - 'cv_rmse': float, average RMSE over CV folds
            - 'cv_mape': float, average MAPE over CV folds
    """
    seed = 42
    # 1) Create the study and optimize CV MAPE via objective()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, params),
        n_trials=params["optuna_search"]["n_trials"],
    )

    # 2) Get best parameters
    best_params = study.best_params

    # 3) Re-evaluate best model to get full CV results
    model_name = params["optuna_search"]["model"]

    cv_results = eval_model(
        x_train=x_train,
        y_train=y_train,
        n_splits=params["n_folds"],
        model_name=model_name,
        model_params=best_params,
        seed=seed,
    )

    return {
        "best_params": best_params,
        "cv_metrics": {
            "cv_mae": cv_results["cv_mae"],
            "cv_rmse": cv_results["cv_rmse"],
            "cv_mape": cv_results["cv_mape"],
        },
    }


def fit_best_model(  # noqa: PLR0913
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
    tuning_results: dict[str, Any],
) -> dict[str, Any]:
    """ """
    model_name = params["optuna_search"]["model"]

    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_train)

    if model_name == "CatBoost":
        model = CatBoostRegressor(
            **tuning_results["best_params"], allow_writing_files=False, verbose=False
        )
    elif model_name == "RF":
        model = RF(**tuning_results["best_params"])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(x_scaled, y_train)

    y_pred_test = model.predict(x_scaler.transform(x_test))
    errors = compute_metrics(y_test, y_pred_test)

    return {
        "model": model,
        "x_scaler": x_scaler,
        "input_example": x_train.iloc[:5].copy(),
        "test_metrics": {
            "test_mae": errors["mae"],
            "test_rmse": errors["rmse"],
            "test_mape": errors["mape"],
        },
    }


# class TrainingPipeline():
#     def __init__(self, name: str):
#         super().__init__(name)

#     def run(self):
#         #  write the above functions and run the same pipeline
#         # kedro makes it standardised and neater, abstract(we don't need to parse config anytime)
#         pass


def log_to_mlflow(
    hyperparams_tuning_results: dict[str, Any],
    train_results: dict[str, Any],
    train_pipeline_params: dict[str, Any],
    mlflow_params: dict[str, Any],
) -> str:
    """_summary_
    Args:
        hyperparams_tuning_results (dict[str, Any]):
            Contains 'best_params' and 'cv_metrics' (cv_mae, cv_rmse, cv_mape)
        train_results (dict[str, Any]):
            Contains 'model', 'x_scaler', 'input_example' and 'test_metrics'
        train_pipeline_params (dict[str, Any]):
            Contains 'optuna_search' with 'model' key
        mlflow_params (dict[str, Any]):
            Contains 'prod_experiment_name'
    Returns:
        str: Model URI of the logged PyFunc model
    """
    # set MLFlow tracking URI and experiment
    mlflow.set_experiment(mlflow_params["prod_experiment_name"])

    # Get model name and determine run name
    model_name = train_pipeline_params["optuna_search"]["model"]
    # Default run name with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}_{timestamp}_candidate"

    # create temporary directory for artifacts
    temp_models_dir = Path("data/06_models")
    temp_models_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        # Log metrics and parameters
        mlflow.log_metrics(hyperparams_tuning_results["cv_metrics"])
        mlflow.log_metrics(train_results["test_metrics"])
        mlflow.log_params(hyperparams_tuning_results["best_params"])

        # Create signature from input example
        input_example = train_results["input_example"]
        x_scaler = train_results["x_scaler"]
        best_model = train_results["model"]
        y_example = best_model.predict(x_scaler.transform(input_example))
        signature = infer_signature(input_example, y_example)

        # save model and scaler artifacts depending on the model type
        if model_name.lower() == "catboost":
            model_path = temp_models_dir / "catboost_model.cbm"
            best_model.save_model(str(model_path))
        elif model_name.lower() in ["rf", "random_forest"]:
            model_path = temp_models_dir / "random_forest_model.pkl"
            joblib.dump(best_model, model_path)

        scaler_path = temp_models_dir / "x_scaler.joblib"
        joblib.dump(x_scaler, scaler_path)

        # prepare artifacts dictionary for MLModelWrapper
        artifacts = {
            "model": str(model_path),
            "scaler": str(scaler_path),
        }

        # log PyFunc model using MLModelWrapper
        # MLModelWrapper
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=MLModelWrapper(model_name=model_name),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            tags={"best_model": "true", "model_type": model_name, "run_name": run_name},
        )

        # return model URI to later register the model
        return model_info.model_uri


def register_candidate_model(model_uri: str, mlflow_params: dict[str, Any]) -> str:
    """
    Register a model in MLFlow and add candidate alias

    Parameters:
        model_uri (str): The mlflow uri to register
        mlflow_params (dict[str, Any]): MLFlow config containing 'registered_model_name' and 'model_aliases'

    Returns:
        str: The registered model version
    """
    client = MlflowClient()
    registered_model_name = mlflow_params["registered_model_name"]

    # Register model
    model_info = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    version = str(model_info.version)

    # Add alias (everytime we train it is a candidate)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=mlflow_params["model_aliases"]["candidate"],
        version=version,
    )

    return version


def validate_challenger(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    training_results: dict[str, Any],
    mlflow_params: dict[str, Any],
) -> None:
    """
    Validates candidate against production and promote if better (lower MAPE)
    """
    client = MlflowClient()
    registered_model_name = mlflow_params["registered_model_name"]
    candidate_alias = mlflow_params["model_aliases"]["candidate"]
    production_alias = mlflow_params["model_aliases"]["production"]

    # load champion model and predict on current test set, compute mape
    try:
        champion_model = load_model_by_alias(
            registered_model_name, alias=production_alias
        )
        y_pred = champion_model.predict(x_test)
        champion_mape = compute_metrics(y_test, y_pred)["mape"]
    except Exception:
        champion_mape = float("inf")  # No champion exists, promote challenger

    challenger_mape = training_results["test_metrics"]["test_mape"]

    if challenger_mape < champion_mape:
        client.delete_registered_model_alias(registered_model_name, production_alias)
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=production_alias,
            version=client.get_model_version_by_alias(
                registered_model_name, candidate_alias
            ).version,
        )
        client.delete_registered_model_alias(registered_model_name, candidate_alias)
