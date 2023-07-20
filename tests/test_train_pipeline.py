import json
import pandas as pd
import pytest
from pathlib import Path
from src.forecast_covid19_positive_cases.feature_eng import add_lag_feature_df
from src.forecast_covid19_positive_cases.split_data import split_timeseries_train_test
from src.forecast_covid19_positive_cases.model import TrainModel
from config.config import (
    FINAL_DATA_SPLIT_FILE_NAMES,
    MODELS_PATH,
    MODEL_FILE,
    FEATURES,
    TARGET,
)


@pytest.fixture
def raw_data_covid19():
    json_file = open("tests/raw_data_example.json")
    data_dict = json.load(json_file)
    data = pd.DataFrame(data_dict)
    return data


@pytest.fixture
def preprocessed_data_covid19(raw_data_covid19):
    return add_lag_feature_df(raw_data_covid19)


def test_add_lag_feature(raw_data_covid19):
    data_feat_eng = add_lag_feature_df(raw_data_covid19)
    feat_cols = ["lag_1", "positive"]
    assert list(data_feat_eng.columns) == feat_cols


def test_add_lag_feature_no_null_values(raw_data_covid19):
    data_feat_eng = add_lag_feature_df(raw_data_covid19)
    assert data_feat_eng.notnull().sum().sum() == 0


def test_time_series_split(preprocessed_data_covid19):
    split_data_lst = split_timeseries_train_test(
        preprocessed_data_covid19, FEATURES, TARGET
    )

    X_train = split_data_lst[0]
    X_test = split_data_lst[1]
    y_train = split_data_lst[2]
    y_test = split_data_lst[3]

    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES
    assert y_train.name == TARGET
    assert y_test.name == TARGET


def test_time_series_split_not_empty(preprocessed_data_covid19):
    split_data_lst = split_timeseries_train_test(
        preprocessed_data_covid19, FEATURES, TARGET
    )

    X_train = split_data_lst[0]
    X_test = split_data_lst[1]
    y_train = split_data_lst[2]
    y_test = split_data_lst[3]

    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES
    assert y_train.name == TARGET
    assert y_test.name == TARGET


def test_train_model(preprocessed_data_covid19):
    final_data_path = Path("forecast_covid19_positive_cases/data/final")
    lr_model = TrainModel(
        final_data_path, FINAL_DATA_SPLIT_FILE_NAMES, MODELS_PATH, MODEL_FILE
    )

    lr_model.train_model()

    train_rmse, test_rmse = lr_model.get_train_test_error()

    assert train_rmse < 26562
    assert test_rmse < 174968
