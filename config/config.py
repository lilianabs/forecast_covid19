from pathlib import Path

# Covid19 API URL
API_URL = "https://api.covidtracking.com/v1/us/daily.json"

# Directories 
RAW_DATA_PATH = Path("data/raw/")
PROCESSED_DATA_PATH = Path("data/processed/")
FINAL_DATA_PATH = Path("data/final/")
MODELS_PATH = Path("models/")

# File names 
RAW_DATA_JSON = "covid19_data.json"
PROCESSED_DATA_FILE = "covid19_data_processed.pkl"
FINAL_DATA_FILE = "covid19_training_data.pkl"
FINAL_X_TRAIN_FILE = "covid19_training_data_x_train.pkl"
FINAL_X_TEST_FILE = "covid19_training_data_x_test.pkl"
FINAL_Y_TRAIN_FILE = "covid19_training_data_y_train.pkl"
FINAL_Y_TEST_FILE = "covid19_training_data_y_test.pkl"
MODEL_FILE = "model.pkl"

FINAL_DATA_SPLIT_FILE_NAMES = [FINAL_X_TRAIN_FILE, 
                               FINAL_X_TEST_FILE,
                               FINAL_Y_TRAIN_FILE,
                               FINAL_Y_TEST_FILE
]

# Feature names
UNNECESARY_FEATURES_COVID_API_DATA = [
    'recovered', 'hash', 'lastModified', 'dateChecked'
]
FEATURES = ['lag_1']
TARGET = 'positive'