"""
Multivariate Time Series Forecasting with Historical Backtesting
Using ETTh1 Dataset (Electricity Transformer Temperature)
Model: Linear Regression
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from darts.datasets import ETTh1Dataset
from darts.models import LinearRegressionModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries

np.random.seed(42)


print("Loading ETTh1 Dataset...")

dataset = ETTh1Dataset()
series = dataset.load()

print(f"Dataset shape: {series.shape}")
print(f"Time range: {series.start_time()} to {series.end_time()}")
print(f"Frequency: {series.freq}")
print(f"\nComponents (Variables):")
for i, comp in enumerate(series.components, 1):
    print(f"  {i}. {comp}")

# Create temporal features as covariates
hour = datetime_attribute_timeseries(series, attribute='hour')
dayofweek = datetime_attribute_timeseries(series, attribute='dayofweek')
month = datetime_attribute_timeseries(series, attribute='month')

covariates = hour.stack(dayofweek).stack(month)

# Scale covariates
scaler_cov = Scaler()
covariates_scaled = scaler_cov.fit_transform(covariates)

# Split: 70% train, 15% validation, 15% test
train_size = int(len(series) * 0.70)
val_size = int(len(series) * 0.15)

train = series[:train_size]
val = series[train_size:train_size + val_size]
test = series[train_size + val_size:]

# Scale the data
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)
series_scaled = scaler.transform(series)


model = LinearRegressionModel(
    lags=24,  # Use past 24 hours (1 day)
    lags_past_covariates=24,  # Use past 24 hours of covariates
    output_chunk_length=12  # Predict 12 hours ahead
)

model.fit(
    series=train_scaled,
    past_covariates=covariates_scaled
)


val_forecast = model.predict(
    n=len(val),
    series=train_scaled,
    past_covariates=covariates_scaled
)

val_forecast_original = scaler.inverse_transform(val_forecast)

print("\nValidation Metrics (by component):")
print("-" * 60)
for component in series.components:
    mae_score = mae(val[component], val_forecast_original[component])
    rmse_score = rmse(val[component], val_forecast_original[component])
    print(f"{component:s}: MAE={mae_score:7.3f} : RMSE={rmse_score:7.3f}")

train_val_scaled = concatenate([train_scaled, val_scaled])

test_forecast = model.predict(
    n=len(test),
    series=train_val_scaled,
    past_covariates=covariates_scaled
)

# Inverse transform to original scale
test_forecast_original = scaler.inverse_transform(test_forecast)

# Calculate test metrics
print("\nTest Metrics (by component):")
print("-" * 60)
for component in series.components:
    mae_score = mae(test[component], test_forecast_original[component])
    rmse_score = rmse(test[component], test_forecast_original[component])
    print(f"{component:6s}: MAE={mae_score:7.3f} | RMSE={rmse_score:7.3f}")

# Historical Backtesting

backtest_start = train_size  

print(f"\nBacktest configuration:")
print(f"  Start point: {series.time_index[backtest_start]}")
print(f"  Forecast horizon: {model.output_chunk_length} hours")
print(f"  Stride: {model.output_chunk_length} hours (continuous coverage)")


backtest_forecast = model.historical_forecasts(
    series=series_scaled,
    past_covariates=covariates_scaled,
    start=backtest_start,
    forecast_horizon=model.output_chunk_length,
    stride=12,  # Move 12 hours forward (same as forecast horizon for continuous coverage)
    retrain=False,
    last_points_only=True,  
    verbose=True
)

# Concatenate all backtests
backtest_concat = concatenate(backtest_forecast, axis=0)
backtest_original = scaler.inverse_transform(backtest_concat)

# Calculate backtest metrics on the overlapping period
print("\nBacktest Metrics (by component):")
actual_subset = series[backtest_original.start_time():backtest_original.end_time()]
for component in series.components:
    mae_score = mae(actual_subset[component], backtest_original[component])
    rmse_score = rmse(actual_subset[component], backtest_original[component])
    print(f"{component:6s}: MAE={mae_score:7.3f} | RMSE={rmse_score:7.3f}")

