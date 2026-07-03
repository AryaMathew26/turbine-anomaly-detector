from typing import Any

import numpy as np
import pandas as pd

from app_data_manager.data_manager import DataManager


def get_wasserstein_distance_1d(
    reference_df: pd.DataFrame,
    monitored_df: pd.DataFrame,
    feature_col: str,
    bins: int | None = None,
) -> float:
    """
    Histogram-based approximation of 1D Wasserstein distance
    for a specific feature column.
    """

    reference = reference_df[feature_col].dropna().to_numpy()
    monitored = monitored_df[feature_col].dropna().to_numpy()

    full_dataset = np.concatenate((reference, monitored))

    if bins is None:
        _, bin_edges = np.histogram(full_dataset, bins="doane")
    else:
        bin_edges = np.linspace(
            min(reference.min(), monitored.min()),
            max(reference.max(), monitored.max()),
            bins + 1,
        )

    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    mon_hist, _ = np.histogram(monitored, bins=bin_edges)

    ref_p = ref_hist / ref_hist.sum()
    mon_p = mon_hist / mon_hist.sum()

    ref_cdf = np.cumsum(ref_p)
    mon_cdf = np.cumsum(mon_p)

    bin_widths = np.diff(bin_edges)

    return float(np.sum(np.abs(ref_cdf - mon_cdf) * bin_widths))


def get_retraining_trigger(
    wasserstein_distance: float,
    threshold: float,
    monitored_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine if retraining is needed based on Wasserstein distance.
    """
    last_timestamp = monitored_data["Timestamps"].iloc[-1]
    if wasserstein_distance > threshold:
        retraining_trigger = 1
    else:
        retraining_trigger = 0
    # Create a DataFrame with the retraining trigger information
    retraining_trigger_df = pd.DataFrame(
        {
            "Timestamps": [last_timestamp],
            "wasserstein_distance": [wasserstein_distance],
            "retraining_trigger": [retraining_trigger],
        }
    )
    return retraining_trigger_df


def save_retraining_trigger_to_db(
    retraining_trigger_df: pd.DataFrame,
    data_manager_config: dict[str, Any],
    db_table_name: str,
) -> None:
    """
    Save the retraining trigger information to the database.
    """
    # Initialize DataManager
    data_manager = DataManager(data_manager_config)
    # Save to retraining trigger table
    data_manager.insert_data_to_db(
        new_data=retraining_trigger_df, table_name=db_table_name
    )
