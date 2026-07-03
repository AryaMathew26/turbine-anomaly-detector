import os
import time
from pathlib import Path

import pandas as pd

from app_data_manager.data_manager import DataManager
from app_data_manager.utils import read_config

project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)


def stream_data_to_db() -> None:
    """
    Stream data point-by-point to the database with a delay between each insertion.
    Uses streaming_frequency and raw_data_table_name from config (data_manager section).
    """
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    config_data_manager = config["data_manager"]
    data_manager = DataManager(config_data_manager)

    # Load inference data from config
    inference_data_folder = config_data_manager["inference_data_folder"]
    inference_data_filename = config_data_manager["inference_data_filename"]
    data_path = os.path.join(
        project_root, inference_data_folder, inference_data_filename
    )
    inference_data = pd.read_parquet(data_path)

    while True:
        # Clean database by reinitializing the raw data table
        data_manager.init_raw_db_table()
        # Initialize other tables
        data_manager.init_predictions_db_table()
        data_manager.init_errors_db_table()
        data_manager.init_anomalies_db_table()

        # Iterate over each row and insert one at a time
        for i, (idx, row) in enumerate(inference_data.iterrows()):
            # Convert single row to DataFrame for insertion
            row_df = row.to_frame().T

            # Insert single row
            data_manager.insert_data_to_db(
                row_df, table_name=config_data_manager["raw_data_table_name"]
            )

            # Sleep before next insertion
            if i < len(inference_data) - 1:  # Don't sleep after last row
                time.sleep(config_data_manager["streaming_frequency"])

            # print(idx)
            # print(data.iloc[-1])


if __name__ == "__main__":
    stream_data_to_db()
