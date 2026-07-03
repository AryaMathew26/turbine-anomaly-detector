from kedro.pipeline import Pipeline, node

from .nodes import (
    add_lag_features,
    add_rolling_features,
    drop_columns,
    get_data_timestamps,
    get_features_and_target,
    load_inference_data_from_db,
    load_training_data_from_db,
    remove_diff_outliers,
    rename_columns,
    smooth_signal,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # pipeline contains list of nodes
            node(
                func=rename_columns,
                inputs=[
                    "loaded_df",
                    "params:feature_eng_pipeline.rename_columns",
                ],  # df_train in conf/base/catalog.yml, params in conf/base/parameters.yml
                outputs="renamed_data",
            ),
            node(
                func=get_data_timestamps,
                inputs="renamed_data",
                outputs="data_timestamps",
            ),
            node(
                func=drop_columns,
                inputs=["renamed_data", "params:feature_eng_pipeline.drop_columns"],
                outputs="dropped_columns_data",
            ),
            node(
                func=remove_diff_outliers,
                inputs=[
                    "dropped_columns_data",
                    "params:feature_eng_pipeline.diff_outliers_thresholds",
                ],
                outputs="diff_outliers_removed_data",
            ),
            node(
                func=smooth_signal,
                inputs=[
                    "diff_outliers_removed_data",
                    "params:feature_eng_pipeline.smooth_signal.columns",
                    "params:feature_eng_pipeline.smooth_signal.window",
                    "params:feature_eng_pipeline.smooth_signal.method",
                ],
                outputs="smoothed_data",
            ),
            node(
                func=add_lag_features,
                inputs=["smoothed_data", "params:feature_eng_pipeline.lag_features"],
                outputs="lagged_data",
            ),
            node(
                func=add_rolling_features,
                inputs=["lagged_data", "params:feature_eng_pipeline.rolling_features"],
                outputs="rolled_stats_data",
            ),
            node(
                func=get_features_and_target,
                inputs=["rolled_stats_data", "params:feature_eng_pipeline.target"],
                outputs=["features_data", "target_data"],
            ),
        ]
    )


def load_training_data(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_training_data_from_db,
                inputs=[
                    "params:training_pipeline.start_timestamp",
                    "params:data_manager.raw_data_table_name",
                    "params:data_manager",
                ],
                outputs="loaded_df",
            ),
        ]
    )


def load_inference_data(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_inference_data_from_db,
                inputs=[
                    "params:inference_pipeline.batch_size",
                    "params:data_manager.raw_data_table_name",
                    "params:data_manager",
                ],
                outputs="loaded_df",
            ),
        ]
    )


def feat_eng_pipeline_training(**kwargs) -> Pipeline:
    return load_training_data() + create_pipeline()


def feat_eng_pipeline_inference(**kwargs) -> Pipeline:
    return load_inference_data() + create_pipeline()
