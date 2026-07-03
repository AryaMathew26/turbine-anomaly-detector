import numpy as np
import pandas as pd
import pytest

OUTLIER_HIGH = 200
OUTLIER_LOW = -10


@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [10, 20, 30, 40], "col2": [1, 2, 3, 4]})


@pytest.fixture
def dataset_with_outliers():
    """Small synthetic dataset with outliers in one column only."""
    n = 15
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    # Smooth series
    t = np.linspace(0, 2 * np.pi, n)
    power = 50 + 10 * np.sin(t)

    df = pd.DataFrame({"power": power, "Timestamps": timestamps})

    # Outliers in power only
    df.loc[5, "power"] = OUTLIER_HIGH  # spike
    df.loc[10, "power"] = OUTLIER_LOW  # impossible drop
    return df
