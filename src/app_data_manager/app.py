import os
import sys
from pathlib import Path

from data_manager import DataManager  # type: ignore
from utils import read_config  # type: ignore

# Add project root and app_ui directory to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.chdir(project_root)


if __name__ == "__main__":
    # read the config file
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    # create an object of the data manager class
    data_manager = DataManager(config["data_manager"])
    # # intialise the raw db table
    # data_manager.init_raw_db_table()

    # # # get last 10 data points from the raw data table
    # last_n_points = data_manager.get_last_n_points(10, table_name=config["data_manager"]["raw_data_table_name"])
    # # print("Last 10 points of raw_data table: ", last_n_points)

    # # read the inference data
    # inference_data = pd.read_parquet(
    #     os.path.join(project_root, "data", "01_raw", "df_prod.parquet")
    # )
    # # insert data into db
    # data_manager.insert_data_to_db(inference_data, table_name="raw_data")
    # last_n_points = data_manager.get_last_n_points(10, table_name=config["data_manager"]["raw_data_table_name"])
    # # print("Last 10 points of raw_data table after insertion: ", last_n_points)

    # data_manager.init_predictions_db_table()
    # data_manager.init_errors_db_table()
    # data_manager.init_anomalies_db_table()

    # # retrieve data since a timestamp
    # data = data_manager.get_data_since_timestamp(
    #     start_timestamp="2010-01-10 00:00:00",
    #     table_name="raw_data"
    #     )
    # print(data)

    # # retrieve data by timestamp range
    # data = data_manager.get_data_by_timestamp_range(
    #     start_timestamp="2010-01-10 00:00:00",
    #     end_timestamp="2010-01-11 00:00:00",
    #     table_name="raw_data"
    # )
    # print(data)

    # predictions = data_manager.get_last_n_points(
    #     n=10,
    #     table_name="predictions"
    # )
    # print("Predictions", predictions)

    # errors = data_manager.get_last_n_points(
    #     n=10,
    #     table_name="errors"
    # )
    # print("Errors", errors)

    # anomalies = data_manager.get_last_n_points(
    #     n=10,
    #     table_name="anomalies"
    # )
    # print("Anomalies", anomalies)
    # data_manager.init_trigger_db_table()

    triggers = data_manager.get_last_n_points(n=10, table_name="retraining_trigger")
    # print("Triggers", triggers)
