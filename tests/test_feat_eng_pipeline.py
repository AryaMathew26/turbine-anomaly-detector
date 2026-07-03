"""Unit tests for feature_eng pipeline nodes."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from tests.conftest import OUTLIER_HIGH, OUTLIER_LOW  # noqa: E402
from turbine_anomaly_detector.pipelines.feature_eng.nodes import (  # noqa: E402
    add_lag_features,
    remove_diff_outliers,
)

# def test_add_lag_features_creates_expected_columns():
#     df = pd.DataFrame({"col1": [10, 20, 30, 40], "col2": [1, 2, 3, 4]})
#     result = add_lag_features(df, lags_dict={"col1": [1, 2], "col2": [1]})
#     assert "col1_lag1" in result.columns
#     assert "col1_lag2" in result.columns
#     assert "col2_lag1" in result.columns
#     assert result["col1_lag1"].iloc[1] == 10
#     assert result["col1_lag2"].iloc[2] == 10
#     assert result["col2_lag1"].iloc[1] == 1


def test_add_lag_features_creates_expected_columns(sample_df):
    result = add_lag_features(sample_df, lags_dict={"col1": [1, 2], "col2": [1]})
    assert "col1_lag1" in result.columns
    assert "col1_lag2" in result.columns
    assert "col2_lag1" in result.columns
    assert result["col1_lag1"].iloc[1] == 10  # noqa: PLR2004
    assert result["col1_lag2"].iloc[2] == 10  # noqa: PLR2004
    assert result["col2_lag1"].iloc[1] == 1  # noqa: PLR2004


def test_remove_diff_outliers_one_column(dataset_with_outliers):
    result = remove_diff_outliers(
        dataset_with_outliers,
        diff_threshold={"power": 30},
    )
    assert result.notna().all().all()  # make sure no NaN values are introduced
    assert result["power"].iloc[5] != OUTLIER_HIGH  # make sure the outlier is removed
    assert result["power"].iloc[10] != OUTLIER_LOW  # make sure the outlier is removed
