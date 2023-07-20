import logging
import pandas as pd
from pathlib import Path
from .utils import load_data, save_data_pkl


def add_lag_feature_to_file(
    processed_data_path: Path,
    processed_data_file: str,
    final_data_path: Path,
    final_data_file: str,
) -> None:
    """Adds lag feature to the time series data file

    Args:
        processed_data_path (Path): processed data file path
        processed_data_file (str): processed data file name
        final_data_path (Path): data file path
        final_data_file (str): data file name
    """

    df = load_data(processed_data_path, processed_data_file)

    df = add_lag_feature_df(df)

    save_data_pkl(df, final_data_path, final_data_file)


def add_lag_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag feature to time series dataframe

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: dataframe with lag feature
    """
    logging.info("Performing feature engineering")

    # Set date column as index
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")

    # Reorder dataframe
    df.sort_index(ascending=True, inplace=True)

    # Add lag feature
    df["lag_1"] = df["positive"].shift(1)
    df.dropna(inplace=True)

    cols_for_training_model = ["lag_1", "positive"]
    logging.info("Columns for training model: %s ", cols_for_training_model)

    return df[cols_for_training_model]
