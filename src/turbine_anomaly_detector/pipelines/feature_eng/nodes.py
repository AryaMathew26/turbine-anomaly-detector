from typing import Any

import numpy as np
import pandas as pd

from app_data_manager.data_manager import DataManager


def rename_columns(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns in a dataframe
    """
    return df.rename(columns=columns)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns in a dataframe
    """
    return df.drop(columns=columns)


def remove_diff_outliers(
    df: pd.DataFrame, diff_threshold: dict[str, float]
) -> pd.DataFrame:
    """
    Remove outliers based on absolute first-order diff and forward-fill the gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to clean.
    diff_threshold : float
        Absolute diff threshold.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe with forward fill.
    outlier_idx : pd.Index
        Indices of removed outliers.
    """

    df_clean = df.copy()

    for col, thresh in diff_threshold.items():
        # 1. Compute absolute diff
        diff_vals = df_clean[col].diff(1).abs()

        # 2. Outlier mask
        outlier_mask = diff_vals > thresh
        outlier_idx = df_clean.index[outlier_mask]

        # 3. Remove outliers
        df_clean.loc[outlier_idx, col] = np.nan

        # 4. Forward fill (and backfill if needed)
        df_clean[col] = df_clean[col].ffill().bfill()

    return df_clean


def smooth_signal(
    df: pd.DataFrame, columns: list[str], window: int, method="mean"
) -> pd.DataFrame:
    """
    Smooth a time-series column using rolling mean or median.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to smooth.
    window : int
        Rolling window size.
    method : str
        "mean"  -> rolling mean filter
        "median" -> rolling median filter (robust smoothing)

    Returns
    -------
    df_smoothed : pd.DataFrame
        DataFrame with smoothed column.
    """

    df_smoothed = df.copy()
    for col in columns:
        if method == "mean":
            df_smoothed[col] = (
                df_smoothed[col]
                .rolling(window=window, min_periods=1, center=False)
                .mean()
            )

        elif method == "median":
            df_smoothed[col] = (
                df_smoothed[col]
                .rolling(window=window, min_periods=1, center=False)
                .median()
            )

        else:
            raise ValueError("method must be 'mean' or 'median'")

    return df_smoothed


def add_lag_features(
    df: pd.DataFrame, lags_dict: dict[str, list[int]], drop_na=False
) -> pd.DataFrame:
    """
    Add lag features to DataFrame.
    Parameters:
    - df: DataFrame
    - columns: list of column names (default: all numeric)
    - lags: int or list of lag periods (default: 1)
    - drop_na: bool, drop NaN rows (default: True)
    Returns: DataFrame with lag features
    """
    df_result = df.copy()

    # Create lag features
    for col, lags in lags_dict.items():
        for lag in lags:
            df_result[f"{col}_lag{lag}"] = df_result[col].shift(lag)

    if drop_na:
        return df_result.dropna()
    else:
        return df_result.bfill()  # Backward fill NaNs


def add_rolling_features(
    df: pd.DataFrame, stats_window_dict: dict[str, dict[str, list[int]]], drop_na=False
) -> pd.DataFrame:
    """
    Add rolling features to DataFrame.
    Parameters:
    - df: DataFrame
    - columns: list of column names
    - window_sizes: int or list of window sizes (default: 7)
    - stats: list of statistics ['mean', 'median', 'std', 'min', 'max', 'skew', 'kurt']
    - drop_na: bool, drop NaN rows (default: True)
    Returns: DataFrame with rolling features
    """
    df_result = df.copy()
    # Create rolling features
    for col in stats_window_dict.keys():
        for window in stats_window_dict[col]["windows"]:
            rolling = df_result[col].rolling(window)

            for stat in stats_window_dict[col]["stats"]:
                if stat == "mean":
                    df_result[f"{col}_roll{window}_mean"] = rolling.mean()
                elif stat == "median":
                    df_result[f"{col}_roll{window}_median"] = rolling.median()
                elif stat == "std":
                    df_result[f"{col}_roll{window}_std"] = rolling.std()
                elif stat == "min":
                    df_result[f"{col}_roll{window}_min"] = rolling.min()
                elif stat == "max":
                    df_result[f"{col}_roll{window}_max"] = rolling.max()
                elif stat == "skew":
                    df_result[f"{col}_roll{window}_skew"] = rolling.skew()
                elif stat == "kurt":
                    df_result[f"{col}_roll{window}_kurt"] = rolling.kurt()
    if drop_na:
        return df_result.dropna()
    else:
        return df_result.bfill()  # Backward fill NaNs


def get_features_and_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the features and target from the dataframe
    """
    x = df.drop(columns=target).copy()
    y = df[target].copy()
    return x, y


def load_training_data_from_db(
    start_timestamp: str,
    table_name: str,
    data_manager_config: dict[str, Any],
) -> pd.DataFrame:
    """"""
    data_manager = DataManager(data_manager_config)

    df = data_manager.get_data_since_timestamp(
        start_timestamp=start_timestamp, table_name=table_name
    )
    return df


def load_inference_data_from_db(
    batch_size: int,
    table_name: str,
    data_manager_config: dict[str, Any],
) -> pd.DataFrame:
    """"""
    data_manager = DataManager(data_manager_config)
    df = data_manager.get_last_n_points(n=batch_size, table_name=table_name)
    return df


def get_data_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the current timestamp from the dataframe
    """
    return df[["Timestamps"]].copy()
