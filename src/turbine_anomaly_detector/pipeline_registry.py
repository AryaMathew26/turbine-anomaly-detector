from kedro.pipeline import Pipeline

from turbine_anomaly_detector.pipelines.feature_eng.pipeline import (
    feat_eng_pipeline_inference as create_feat_eng_pipeline_inference,
)
from turbine_anomaly_detector.pipelines.feature_eng.pipeline import (
    feat_eng_pipeline_training as create_feat_eng_pipeline_training,
)
from turbine_anomaly_detector.pipelines.inference.pipeline import (
    create_pipeline as create_inference_pipeline,
)
from turbine_anomaly_detector.pipelines.monitoring.pipeline import (
    create_monitoring_pipeline,
)

# from turbine_anomaly_detector.pipelines.feature_eng.pipeline import create_pipeline as create_feature_eng_pipeline
from turbine_anomaly_detector.pipelines.training.pipeline import (
    create_pipeline as create_training_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    # this must be called register_pipelines
    """
    Register the project's pipelines
    Returns:
        A mapping form pipeline names to 'Pipeline' objects
    """
    feat_eng_pipeline_training = create_feat_eng_pipeline_training()
    feat_eng_pipeline_inference = create_feat_eng_pipeline_inference()
    # feature_eng_pipeline = create_feature_eng_pipeline()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    monitoring_pipeline = create_monitoring_pipeline()
    # kedro run - to run the pipeline
    return {
        "__default__": feat_eng_pipeline_training + training_pipeline,
        "training": feat_eng_pipeline_training + training_pipeline,
        "inference": feat_eng_pipeline_inference + inference_pipeline,
        "monitoring": monitoring_pipeline,
    }
