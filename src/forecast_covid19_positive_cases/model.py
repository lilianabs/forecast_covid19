import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Tuple, List
from .utils import load_data


class TrainModel:
    """Contains functions for training, evaluating, predicting
    and saving model
    """

    def __init__(
        self,
        final_data_path: Path,
        split_data_names: List[str],
        models_path,
        model_file,
    ):
        """Initializes class attributes"""
        self.final_data_path = final_data_path
        self.split_data_names = split_data_names
        self.models_path = models_path
        self.model_file = model_file
        self.train_rmse = 0.0
        self.test_rmse = 0.0

    def train_model(self) -> None:
        """Runs the training pipeline:
        - Loads training data
        - Trains a sklearn model
        - Computes train and test RMSE
        - Saves model
        """
        X_train, X_test, y_train, y_test = self.__load_training_data()

        model = self.__train_model(X_train, y_train)

        self.__evaluate_model_rmse(model, X_train, X_test, y_train, y_test)

        self.__save_model(model, self.models_path, self.model_file)

    def __load_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads training data X_train, X_test, y_train, y_test

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: train
            and test split
        """
        training_files = []

        for data_file in self.split_data_names:
            train_file = load_data(self.final_data_path, data_file)

            training_files.append(train_file)

        return (
            training_files[0],
            training_files[1],
            training_files[2],
            training_files[3],
        )

    def __train_model(self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> BaseEstimator:
        """Trains a sklearn model

        Args:
            X_train (pd.DataFrame): features for training model
            y_train (pd.Series): target for training model

        Returns:
            model (BaseEstimator): trained model
        """
        logging.info("Training model")
        model = LinearRegression()

        model.fit(X_train, y_train)

        return model

    def __evaluate_model_rmse(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Evaluates a model

        Args:
            model (LogisticRegression): model
            X_train (pd.DataFrame): features for training model
            X_test (pd.DataFrame): features for testing model
            y_train (pd.Series): target for training model
            y_test (pd.Series): target for testing model

        Returns:
            train_rmse (float): train root mean squared error
            test_rmse (float): train root mean squared error
        """
        logging.info("Evaluating model")
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        self.test_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        self.train_rmse = round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4)

        logging.info("Train RMSE: %s", self.train_rmse)
        logging.info("Test RMSE: %s", self.test_rmse)

    def __forecast(self, model: BaseEstimator, X: pd.DataFrame) -> np.array:
        """Computes predictions

        Args:
            model (BaseEstimator): model
            X (pd.DataFrame): data for forecasting

        Returns:
            pd.Series: forecasting
        """
        logging.info("Computing predictions")
        return model.predict(X)

    def __save_model(
        self, model: BaseEstimator, models_path: Path, model_file: str
    ) -> None:
        """Saves a model for forecasting

        Args:
            model (BaseEstimator): trained model
            models_path (Path): path to save the model
            model_file (str): name for the model file
        """
        logging.info("Saving model as .pkl file")

        model_file_name = Path().resolve().parent / models_path / model_file

        try:
            pickle.dump(model, open(model_file_name, "wb"))
            logging.info("Saved model file %s", model_file_name)
        except Exception as err:
            logging.error("Error saving model file: %s", err)

    def get_train_test_error(self) -> None:
        """Returns train and test error

        Returns:
            float: train_rmse
            float: test_rmse
        """
        return self.train_rmse, self.test_rmse
