import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .utils import load_data, save_data_pkl

def split_train_test(final_data_path: Path,
                     final_data_file: str,
                     final_data_split_file_names: List[str],
                     features: List[str],
                     target: str,
                     train_size=.8) -> None:
    """Splits the data intro test and train and stores
       the resulting splits on the final directoy

    Args:
        final_data_path (Path): final data path
        final_data_file (str): final data file name
        final_data_split_file_names (List[str]): names for the 
                                                 split files
        features (List[str]): features for the model
        target (str): target column for the model
        train_size (float, optional): Train size. Defaults to .8.
    """
    logging.info("Spliting data into train and test")
    
    df = load_data(final_data_path, final_data_file)

    data_split = split_timeseries_train_test(df, 
                                             features,
                                             target,
                                             train_size=train_size)

    for data, data_file in zip(data_split, final_data_split_file_names):
        save_data_pkl(data, 
                    final_data_path, 
                    data_file
        )


def split_timeseries_train_test(df: pd.DataFrame,
                                features: List[str],
                                target: str,
                                train_size=0.8) -> List:
    """Splits time series data

    Args:
        df (pd.DataFrame): dataframe to split
        features (List[str]): features for the model
        target (str): target column for the model
        train_size (float, optional): size of the train data.
                                       Defaults to 0.8.

    Returns:
        List: list that contains file splits for training
    """
    try:
        train_set, test_set= np.split(df[features + [target]], 
                                    [int(train_size *len(df))])
        
        X_train = train_set[features]
        y_train = train_set[target]

        X_test = test_set[features]
        y_test = test_set[target]

        logging.info("Splitted data into train and test")
    except Exception as err:
        logging.error("Error splitting data file: %s", err)

    return [X_train, X_test, y_train, y_test]