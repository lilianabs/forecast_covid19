import logging
from pathlib import Path
from forecast_covid19_positive_cases.ingestion import Covid19Data
from config.config import (
    API_URL,
    RAW_DATA_PATH,
    RAW_DATA_JSON,
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_FILE,
)

logging_path = Path("logs/")
logging_file_name = "etl_pipeline.log"

logging_file = logging_path / logging_file_name

logging.basicConfig(
    filename=logging_file,
    level=logging.INFO,
    filemode="w+",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def extract_data() -> None:
    """Runs the ETL job for extracting and processing covid19 data"""
    covid19data = Covid19Data(API_URL)
    covid19data.extract(RAW_DATA_PATH, RAW_DATA_JSON)
    covid19data.transform(
        RAW_DATA_PATH, RAW_DATA_JSON, PROCESSED_DATA_PATH, PROCESSED_DATA_FILE
    )


def main():
    """ETL pipeline"""
    extract_data()


if __name__ == "__main__":
    main()
