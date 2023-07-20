import logging
import pandas as pd
from pathlib import Path


def load_data(data_path: Path, data_file: str) -> pd.DataFrame:
    """Loads data

    Args:
        data_path (Path): data file path
        data_file (str): data file name

    Returns:
        df (pd.DataFrame): data loaded
    """

    data_file_name = Path().resolve().parent / data_path / data_file

    logging.info("Loading data pkl file: %s ", data_file_name)

    try:
        df = pd.read_pickle(data_file_name)
        logging.info("Utils load_data: Loaded data file %s", data_file_name)
    except FileNotFoundError as err:
        df = pd.DataFrame()
        logging.error("Error loading data file: %s", err)

    return df


def save_data_pkl(df: pd.DataFrame, data_path: Path, data_file: str) -> None:
    """Saves data as a .pkl file

    Args:
        df (pd.DataFrame): data
        data_path (Path): data file path
        data_file (str): data file name
    """

    data_file_name = Path().resolve().parent / data_path / data_file

    try:
        df.to_pickle(data_file_name)
        logging.info("Utils save_data_plkl saved data file %s", data_file_name)
    except Exception as err:
        logging.error("Error saving data file: %s", err)
