import logging
import uvicorn
import pickle
import pandas as pd

from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from config.config import MODELS_PATH, MODEL_FILE


logging_path = Path("logs/")
logging_file_name = "forecasting_covid19_api.log"
logging_file = logging_path / logging_file_name
logging.basicConfig(
    filename=logging_file,
    level=logging.INFO,
    filemode="w+",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class CovidData(BaseModel):
    date: str
    positive_prev_day: int


def load_model() -> BaseEstimator:
    model_file_path = MODELS_PATH / MODEL_FILE
    try:
        forecast_model = pickle.load(open(model_file_path, "rb"))
        logging.info("Loaded model pkl file: %s ", model_file_path)
    except FileNotFoundError as err:
        forecast_model = None
        logging.error("Error loading data file: %s", err)
    return forecast_model


app = FastAPI()
forecast_model = load_model()


@app.get("/")
def health_check():
    return {"status_code": 200, "health": "ok"}


@app.post("/predict")
def predict(covid_data: CovidData):
    input_data = pd.DataFrame(
        [[covid_data.date, covid_data.positive_prev_day]],
        columns=["date", "lag_1"]
    )

    input_data["date"] = pd.to_datetime(input_data["date"], format="%Y%m%d")
    input_data = input_data.set_index("date")

    logging.info("Input data after preprocessing: %s", input_data.head())

    prediction = int(forecast_model.predict(input_data)[0])

    logging.info("Prediction: %s ", prediction)

    if not prediction:
        raise HTTPException(status_code=400, detail="Invalid prediction")

    return {"Forecast number positive cases: ": prediction}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
