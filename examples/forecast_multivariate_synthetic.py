"""
Multivariate Time Series Forecasting with Historical Backtesting
Using Darts library and Linear Regression model
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.models import LinearRegressionModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae
from darts.utils.timeseries_generation import datetime_attribute_timeseries

np.random.seed(42)

# Generate Synthetic Multivariate Time Series
print("Generating synthetic multivariate time series...")

dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n = len(dates)
t = np.arange(n)

# Variable 1: Sales with weekly seasonality and trend
sales = 100 + 0.05 * t + 20 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 5, n)

# Variable 2: Marketing spend (influences sales)
marketing = 50 + 0.03 * t + 10 * np.sin(2 * np.pi * t / 7 + 1) + np.random.normal(0, 3, n)

# Variable 3: Website visits
visits = 500 + 0.2 * t + 100 * np.sin(2 * np.pi * t / 7 + 0.5) + np.random.normal(0, 20, n)

# Create multivariate TimeSeries
df = pd.DataFrame({
    'sales': sales,
    'marketing': marketing,
    'visits': visits
}, index=dates)

series = TimeSeries.from_dataframe(df)
print(f"Series shape: {series.shape}")
print(f"Components: {series.components}")

# Day of week and month as covariates
dow = datetime_attribute_timeseries(series, attribute='dayofweek')
month = datetime_attribute_timeseries(series, attribute='month')

covariates = dow.stack(month)

# Scale covariates
scaler_cov = Scaler()
covariates_scaled = scaler_cov.fit_transform(covariates)

# Train/Test Split
print("\nSplitting data...")

# Use last 90 days for testing
train, test = series[:-90], series[-90:]
print(f"Train length: {len(train)}, Test length: {len(test)}")

# Scale the data
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)
series_scaled = scaler.transform(series)

print("\nTraining Linear Regression model...")

model = LinearRegressionModel(
    lags=14,  # Use past 14 days
    lags_past_covariates=14,  # Use past 14 days of covariates
    output_chunk_length=7  # Predict 7 days ahead
)

# Train the model
model.fit(
    series=train_scaled,
    past_covariates=covariates_scaled
)

print("Model training complete!")

# Make Predictions
forecast = model.predict(
    n=len(test),
    series=train_scaled,
    past_covariates=covariates_scaled
)

# Inverse transform to original scale
forecast_original = scaler.inverse_transform(forecast)

# Calculate metrics for each component
print("\nForecast Metrics:")
for component in series.components:
    mape_score = mape(test[component], forecast_original[component])
    mae_score = mae(test[component], forecast_original[component])
    print(f"  {component}: MAPE = {mape_score:.2f}%, MAE = {mae_score:.2f}")

# Historical Backtesting
print("\nHistorical backtesting...")

# Backtest parameters
backtest_forecast = model.historical_forecasts(
    series=series_scaled,
    past_covariates=covariates_scaled,
    start=0.7,  # Start at 70% of the data
    forecast_horizon=7,  # 7-day ahead forecasts
    stride=7,  # Move 7 days forward each time
    retrain=False,  # Don't retrain (use same model)
    last_points_only=False,  # Keep all forecast points
    verbose=True
)

# Concatenate all backtests
backtest_concat = concatenate(backtest_forecast)
backtest_original = scaler.inverse_transform(backtest_concat)

# Calculate backtest metrics
print("\nBacktest Metrics:")
for component in series.components:
    mape_score = mape(series[component], backtest_original[component])
    mae_score = mae(series[component], backtest_original[component])
    print(f"  {component}: MAPE = {mape_score:.2f}%, MAE = {mae_score:.2f}")


fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Multivariate Time Series Forecasting with Linear Regression', 
             fontsize=16, fontweight='bold')

# Plot each component
for idx, component in enumerate(series.components):
    # Left column: Forecast vs Actual
    ax1 = axes[idx, 0]
    train[component].plot(ax=ax1, label='Train', linewidth=1.5)
    test[component].plot(ax=ax1, label='Test (Actual)', linewidth=2, color='blue')
    forecast_original[component].plot(ax=ax1, label='Forecast', linewidth=2, 
                                      color='red', linestyle='--')
    ax1.set_title(f'{component.capitalize()} - Forecast vs Actual')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right column: Historical Backtest
    ax2 = axes[idx, 1]
    series[component].plot(ax=ax2, label='Actual', linewidth=1.5, alpha=0.7)
    backtest_original[component].plot(ax=ax2, label='Backtest', linewidth=2, 
                                      color='green', linestyle='--')
    ax2.set_title(f'{component.capitalize()} - Historical Backtest')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=train.end_time(), color='gray', linestyle=':', 
                linewidth=2, label='Train/Test Split')

plt.tight_layout()
plt.show()