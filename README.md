# Forecast covid 19 positive cases

This repository contains a forecasting model that predicts the number of positive covid19 cases. It makes predictions via request to the forecasting API.

The training pipeline consists of the following steps:

1. Get data from [The COVID Tracking Project API](https://covidtracking.com/data/api).
2. Preprocess the data:
   1. Remove unnecesary columns. 
   2. Filter out rows that don't contain data for all of the 56 US states and territories.
3. Perform feature engineering:
   1. Reorder data in ascending order.
   2. Creates a lag feature for one time step.
4. Splits data intro train and test (uses a custom function to split the data since is indexed by date).
5. Trains a Linear regression model using the lag feature.
6. Saves the Linear refression model as a pkl.
7. Deploys the model via an API.


>**Note:** For more information on the lastest version of the model, see the [model card](model_card.md).


## Run the project
Before running the API locally, make sure you have Python 3.9+ installed.

To run the API, follow theses steps:

1. Create a new Python environment:

   ```
   python3 -m venv venv
   ```

1. Activate the Python environment:

   ```
   source venv/bin/activate
   ```

1. Update pip:

   ```
   python3 -m pip install --upgrade pip setuptools wheel
   ```

1. Install the project:

   ```
   python3 -m pip install -e
   ```

1. Install the project requirements:

   ```
   python3 -m pip install -r requirements.txt
   ```

1. Extract the covid19 from the API:

   ```
   python3 src/run_etl.py
   ```

1. Train the forecasting model:

   ```
   python3 src/run_train_pipeline.py
   ```

2. Run the forecasting API: 

   ```
   python3 main.py
   ```


You can run a sample request by running the script `client_api.py` or using the following command:
   
   ``` 
   curl -i -X POST -H "Content-Type:application/json" -d "{ \"date\": \"20210308\", \"positive_prev_day\": 28789489 }" http://localhost:8000/predict
   ```

>**Note:** The command above is an example and might need a different formating depending on the comand line you're using.

Also, note that you can check the executions logs in the `logs/forecasting_covid19.log` file.