# Model Card

Last updated: May 2023

## Model Details

This is a simple Linear Regression model.

## Intended Use

This model is used to forecast the number of covid19 positve cases.

## Training Data

The data used for training comes from [The COVID Tracking Project API](https://covidtracking.com/data/api). It has the following features:

* lag_1: Lag feature with a one time stepl.

## Evaluation Data

The covid19 data was split into training and test with train_size = 80%

## Metrics
The model achived the following metrics:

Train RMSE: 26561.7899
Test  RMSE: 174967.3088

## Ethical Considerations

The forecasting predictions are only informative and should not be considered as 100% accurate.

## Caveats and Recommendations

The can model can be improved by considering more lag features as well as using more advance training algorithms and neural network achitectures.