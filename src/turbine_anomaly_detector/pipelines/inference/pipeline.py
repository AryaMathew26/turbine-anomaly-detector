from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_model_errors,
    compute_rolling_error,
    detect_anomaly,
    load_champion_model,
    predict,
    save_predictions_to_db,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # nodes will be added here
            node(
                func=load_champion_model,
                inputs=["params:mlflow"],
                outputs="champion_model",
            ),
            node(
                func=predict,
                inputs=[
                    "features_data",
                    "champion_model",
                    "params:inference_pipeline.predictions_column_name",
                ],
                outputs="predictions",
            ),
            node(
                func=compute_model_errors,
                inputs=[
                    "predictions",
                    "target_data",
                    "params:inference_pipeline.anomaly_error_type",
                ],
                outputs="model_error",
            ),
            node(
                func=compute_rolling_error,
                inputs=["model_error", "params:inference_pipeline.rolling_window"],
                outputs="rolling_errors",
            ),
            node(
                func=detect_anomaly,
                inputs=[
                    "rolling_errors",
                    "params:inference_pipeline.anomaly_threshold",
                    "params:inference_pipeline.anomaly_error_type",
                ],
                outputs="anomalies",
            ),
            node(
                func=save_predictions_to_db,
                inputs=[
                    "predictions",
                    "params:inference_pipeline.predictions_column_name",
                    "params:data_manager.predictions_table_name",
                    "data_timestamps",
                    "params:data_manager",
                ],
                outputs=None,
            ),
            node(
                func=save_predictions_to_db,
                inputs=[
                    "rolling_errors",
                    "params:inference_pipeline.errors_column_names",
                    "params:data_manager.errors_table_name",
                    "data_timestamps",
                    "params:data_manager",
                ],
                outputs=None,
            ),
            node(
                func=save_predictions_to_db,
                inputs=[
                    "anomalies",
                    "params:inference_pipeline.anomalies_column_name",
                    "params:data_manager.anomalies_table_name",
                    "data_timestamps",
                    "params:data_manager",
                ],
                outputs=None,
            ),
        ]
    )
