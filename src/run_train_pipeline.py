import logging

from config.config import (
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_FILE,
    FINAL_DATA_PATH,
    FINAL_DATA_FILE,
    FINAL_DATA_SPLIT_FILE_NAMES,
    MODELS_PATH,
    MODEL_FILE,
    FEATURES,
    TARGET,
)

from pathlib import Path
from forecast_covid19_positive_cases.feature_eng import add_lag_feature_to_file
from forecast_covid19_positive_cases.split_data import split_train_test
from forecast_covid19_positive_cases.model import TrainModel

logging_path = Path("logs/")
logging_file_name = "train_pipeline.log"

logging_file = logging_path / logging_file_name

logging.basicConfig(
    filename=logging_file,
    level=logging.INFO,
    filemode="w+",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def preprocess() -> None:
    """Preprocess the covid19 data for training the model"""
    add_lag_feature_to_file(
        PROCESSED_DATA_PATH,
        PROCESSED_DATA_FILE,
        FINAL_DATA_PATH,
        FINAL_DATA_FILE
    )


def split_data() -> None:
    """Splits data into train and test"""
    split_train_test(
        FINAL_DATA_PATH,
        FINAL_DATA_FILE,
        FINAL_DATA_SPLIT_FILE_NAMES,
        FEATURES,
        TARGET,
        train_size=0.8,
    )


def train_pipeline() -> None:
    """Runs the train pipeline"""
    lr_model = TrainModel(
        FINAL_DATA_PATH,
        FINAL_DATA_SPLIT_FILE_NAMES,
        MODELS_PATH,
        MODEL_FILE
    )
    lr_model.train_model()


def main():
    """Train model pipeline"""
    preprocess()
    split_data()
    train_pipeline()


if __name__ == "__main__":
    main()
