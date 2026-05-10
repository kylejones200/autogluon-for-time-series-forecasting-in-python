# AutoGluon for Time Series Forecasting in Python

AutoGluon is auto ML Python library that streamlines feature engineering, model selection, hyperparameter tuning, and evaluation. AutoGluon...

### AutoGluon for Time Series Forecasting in Python
**AutoGluon** is auto ML Python library that streamlines feature engineering, model selection, hyperparameter tuning, and evaluation. AutoGluon has a `TimeSeriesPredictor` specifically made for time series forecasting.

AutoGluon's time series module automates a lot of things --- including model selection, tuning, and feature engineering. It can create ensembles by combining multiple models. The API is easy to use (especially if you have built models with AWS SageMaker before).

### AutoGluon Workflow
The workflow with AutoGluon for time series involves preparing the dataset, initializing the `TimeSeriesPredictor` , training models automatically, and generating forecasts.

Here is an example with synthetic data. We start by declaring:

- **time_column**: The timestamp for each observation.
- **target_column**: The value to be predicted.
- **item_id_column (optional)**: Identifies different time series in a dataset (for multivariate forecasting).


Output:


#### Initializing and Training the Predictor
To initialize the TimeSeriesPredictor we need to specify the target, time, and (optionally) the item ID columns.

We don't have to do anything for the Predictor --- it will automatically select a model for us.


The predictor will automatically do feature engineering (e.g., lagged values, seasonality features) and train multiple models, including statistical models (ARIMA), machine learning models (LightGBM, CatBoost), and deep learning models (N-BEATS).

Generate forecasts for the next `prediction_length` time steps.


Output:


AutoGluon includes built-in visualization tools.


There are three different products in our our dataset. So we get a prediction for each one.


AutoGluon provides metrics like RMSE, MAPE, and MAE to evaluate model performance.


Validation score from lowest (best) to highest (worst):


It takes a long time to run Autogluon (this run took 24 mins). You can see the models that take a long time in the chart above --- chronos (the LLM model) was the biggest culprit requiring 16 and a half minutes on its own.

### Deployment and Saving Models
I don't talk about model management in most of my articles but it is important for deploying models and using them in the real world. autogluon lets us save the trained model for reuse. we can use this later for testing or we can put this into a contianer and use it for inference as new data comes in.


### So what?
AutoGluon is a nice library. I like the automation but sometimes it feels like it picks really random models that I wouldn't have picked (maybe that is a good thing). I don't love how the ensemble works and I would prefer to use other libraries for ensembles because it just take so long to run. If I could only use one time series library, I would not pick AutoGluon.
