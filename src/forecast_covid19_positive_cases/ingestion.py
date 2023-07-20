import json
import logging
import requests
from pyspark.sql import SparkSession
from config.config import UNNECESARY_FEATURES_COVID_API_DATA
from pathlib import Path
from .utils import save_data_pkl


class Covid19Data:
    """Contains functions for extracting and processing Covid19 data

    Attributes:
              api_url (str): Covid19 API url
    """

    def __init__(self, api_url: str):
        """Initializes class attributes

        Args:
            api_url (str): Covid19 API url
        """
        self.api_url = api_url

        self.spark = SparkSession.builder.master("local[*]").appName("Test").getOrCreate()
        if self.spark is None:
            self.spark = globals().get("spark", None)

    def extract(self, raw_data_path: Path, raw_data_json: str) -> None:
        """Extracts data from covid19 API

        Args:
            raw_data_path (Path): raw data directory path
            raw_data_json (Path): json file name
        """
        logging.info("Making request to covid19 API")
        response = requests.get(self.api_url)

        if response.status_code == 200:
            data = response.json()

            json_file_path = (
                Path().resolve().parent / raw_data_path / raw_data_json
            )

            with open(json_file_path, "w") as f:
                json.dump(data, f)
                logging.info(
                    "Successfully extracted data from covid19 API"
                )
        else:
            logging.error("Response error from covid19 API: %s",
                          response.status_code)

    def transform(
        self,
        raw_data_path: Path,
        raw_data_json: str,
        processed_data_path: Path,
        processed_data_file: str,
    ) -> None:
        """Transforms data from covid19 API

        It removes unnecesary columns and selects rows that have data
        for all 56 US states and territories.

        Args:
            raw_data_path (Path): raw data directory path
            raw_data_json (str): json file name
            processed_data_path (Path): processed data directory path
            processed_data_file (str): pickle file name
        """

        logging.info("Transforming covid19 data")
        raw_data_file = Path().resolve().parent / raw_data_path / raw_data_json

        try:
            df = self.spark.read.json(str(raw_data_file))
            logging.info("Loaded raw data file %s", raw_data_file)
        except FileNotFoundError as err:
            df = self.spark.createDataFrame([])
            logging.error("Error loading raw data file: %s", err)

        for col in UNNECESARY_FEATURES_COVID_API_DATA:
            df = df.drop(col)

        df = df.filter(df.states == 56)

        logging.info("Storing transformed data")

        save_data_pkl(df.toPandas(), processed_data_path, processed_data_file)
