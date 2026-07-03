from kedro.pipeline import Pipeline, node

from .nodes import (
    fit_best_model,
    log_to_mlflow,
    register_candidate_model,
    train_test_split,
    tune_hyperparameters,
    validate_challenger,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # nodes will be added here
            node(
                func=train_test_split,
                inputs=["features_data", "target_data", "params:training_pipeline"],
                outputs=["x_train", "y_train", "x_test", "y_test"],
            ),
            node(
                func=tune_hyperparameters,
                inputs=["x_train", "y_train", "params:training_pipeline"],
                outputs="tuning_results",
            ),
            node(
                func=fit_best_model,
                inputs=[
                    "x_train",
                    "y_train",
                    "x_test",
                    "y_test",
                    "params:training_pipeline",
                    "tuning_results",
                ],
                outputs="training_results",
            ),
            node(
                func=log_to_mlflow,
                inputs=[
                    "tuning_results",
                    "training_results",
                    "params:training_pipeline",
                    "params:mlflow",
                ],
                outputs="mlflow_model_uri",
            ),
            node(
                func=register_candidate_model,
                inputs=["mlflow_model_uri", "params:mlflow"],
                outputs="model_version",
            ),
            node(
                func=validate_challenger,
                inputs=["x_test", "y_test", "training_results", "params:mlflow"],
                outputs=None,
            ),
        ]
    )
