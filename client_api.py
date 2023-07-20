import requests

FORECAST_API_URL = "http://localhost:8000"
FORECAST_API_PREDICT_ENDPOINT = "/predict"


def health_check_api():
    # Health check
    response = requests.get(FORECAST_API_URL)
    print(f"API status: {response.status_code}")


def get_forecast():
    covid_data = {"date": "20210308", "positive_prev_day": 28789489}

    response = requests.post(
        FORECAST_API_URL + FORECAST_API_PREDICT_ENDPOINT, json=covid_data
    )
    prediction = response.json()
    print(f"Response status code: {response.status_code}")
    print(f"Forecast number of positve cases: {prediction}")


def main():
    health_check_api()
    get_forecast()


if __name__ == "__main__":
    main()
