from kedro.pipeline import Pipeline, node

from turbine_anomaly_detector.pipelines.feature_eng.nodes import (
    load_inference_data_from_db as load_inference_batch,
)
from turbine_anomaly_detector.pipelines.feature_eng.nodes import (
    load_training_data_from_db,
)

from .nodes import (
    get_retraining_trigger,
    get_wasserstein_distance_1d,
    save_retraining_trigger_to_db,
)


def create_monitoring_pipeline():
    return Pipeline(
        [
            node(
                func=load_training_data_from_db,
                inputs=[
                    "params:training_pipeline.start_timestamp",
                    "params:data_manager.raw_data_table_name",
                    "params:data_manager",
                ],
                outputs="reference_data",
            ),
            node(
                func=load_inference_batch,
                inputs=[
                    "params:inference_pipeline.batch_size",
                    "params:data_manager.raw_data_table_name",
                    "params:data_manager",
                ],
                outputs="monitored_data",
            ),
            node(
                func=get_wasserstein_distance_1d,
                inputs=[
                    "reference_data",
                    "monitored_data",
                    "params:monitoring_pipeline.monitored_feature",
                ],
                outputs="wasserstein_distance",
            ),
            node(
                func=get_retraining_trigger,
                inputs=[
                    "wasserstein_distance",
                    "params:monitoring_pipeline.wasserstein_threshold",
                    "monitored_data",
                ],
                outputs="retraining_trigger_df",
            ),
            node(
                func=save_retraining_trigger_to_db,
                inputs=[
                    "retraining_trigger_df",
                    "params:data_manager",
                    "params:data_manager.retraining_trigger_table_name",
                ],
                outputs=None,
            ),
        ]
    )
