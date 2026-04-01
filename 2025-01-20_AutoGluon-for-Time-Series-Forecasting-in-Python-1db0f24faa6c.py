# Description: Short example for AutoGluon for Time Series Forecasting in Python.



from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries import TimeSeriesPredictor
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Create more complex sample data with seasonal patterns
np.random.seed(42)  # for reproducibility

# Generate timestamps for 3 years of monthly data
timestamps = pd.date_range("2020-01-01", periods=36, freq="ME")

# Function to create seasonal pattern
def create_seasonal_data(base_value, trend, seasonal_amplitude, noise_level):
    time = np.arange(len(timestamps))
    # Trend component
    trend_component = base_value + trend * time
    # Seasonal component (yearly seasonality)
    seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * time / 12)
    # Random noise
    noise = np.random.normal(0, noise_level, len(time))
    return trend_component + seasonal_component + noise

# Create different patterns for different items
data = {
    "item_id": ["A"] * 36 + ["B"] * 36 + ["C"] * 36,
    "timestamp": timestamps.tolist() * 3,
    "sales": np.concatenate([
        # Item A: Strong seasonality, moderate trend, low noise
        create_seasonal_data(base_value=1000, trend=15, seasonal_amplitude=200, noise_level=30),
        # Item B: Moderate seasonality, high trend, moderate noise
        create_seasonal_data(base_value=500, trend=25, seasonal_amplitude=100, noise_level=50),
        # Item C: Weak seasonality, negative trend, high noise
        create_seasonal_data(base_value=1500, trend=-10, seasonal_amplitude=50, noise_level=100)
    ])
}

# Create DataFrame and set multi-index
df = pd.DataFrame(data)
df = df.set_index(['item_id', 'timestamp'])

# Convert to TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column='item_id',
    timestamp_column='timestamp'
)

item_id   timestamp  sales

# Initialize predictor
predictor = TimeSeriesPredictor(
    prediction_length=6,      # Forecast horizon
    eval_metric='MASE',      # Evaluation metric
    target='sales',          # Target variable
)

# Train the predictor
predictor.fit(train_data=train_data)

# Generate forecasts
forecasts = predictor.predict(train_data)
logger.info(forecasts.head())

# Plot forecasted vs actual values
predictor.plot(train_data)

# Evaluate model performance
performance = predictor.evaluate(train_data)
logger.info(performance)

item_id   timestamp      sales
0       A 2022-01-31  350.1234
1       A 2022-02-28  360.5678
2       A 2022-03-31  370.9876
3       B 2022-01-31  310.4321
4       B 2022-02-28  320.8765

# Plot forecasted vs actual values
predictor.plot(train_data)

# Evaluate model performance
performance = predictor.evaluate(df)
logger.info(performance)

predictor.save("timeseries_model")

"""
Load the Model
Load the saved model to make predictions on new data.
"""

predictor = TimeSeriesPredictor.load("timeseries_model")
new_forecasts = predictor.predict(new_data)
