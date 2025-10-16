# src/forecasting/eda_sarima_forecast.py
# Author: Finn Porter
# Purpose: Exploratory SARIMA forecasting on 5G network data (EDA stage)
# Note: This script explores temporal trends and seasonality in average latency using SARIMA.

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_data_with_imputation.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare time-series data

# Convert UNIX timestamp to datetime
df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# Compute average latency from svr1–svr4
df["avg_latency"] = df[["svr1", "svr2", "svr3", "svr4"]].mean(axis=1)

# Resample hourly to get mean latency per hour
df_hourly = df.resample("H", on="timestamp").mean(numeric_only=True)
df_hourly = df_hourly.dropna(subset=["avg_latency"])
print(f"Resampled hourly data shape: {df_hourly.shape}")

# Fit SARIMA model
print("\nFitting SARIMA model (order=(1,1,1), seasonal_order=(1,1,1,24)) ...")
model = SARIMAX(df_hourly["avg_latency"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
fit = model.fit(disp=False)

# Forecast next 24 hours
forecast_steps = 24
forecast = fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(
    start=df_hourly.index[-1] + pd.Timedelta(hours=1),
    periods=forecast_steps,
    freq="H"
)
forecast = pd.Series(forecast.values, index=forecast_index)
print(f"Forecast generated for next {forecast_steps} hours")

# Plot actual vs forecasted values
plt.figure(figsize=(10, 5))
plt.plot(df_hourly.index, df_hourly["avg_latency"], label="Observed", linewidth=2)
plt.plot(forecast.index, forecast, label="Forecast (Next 24h)", color="red", linewidth=2)
plt.title("EDA SARIMA Forecast: Average Latency (Next 24 Hours)")
plt.xlabel("Time")
plt.ylabel("Average Latency (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "eda_sarima_latency_forecast.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Forecast plot saved to: {plot_path}")

# Save forecast values to CSV
forecast_df = pd.DataFrame({
    "timestamp": forecast.index,
    "predicted_avg_latency": forecast.values
})
forecast_csv_path = os.path.join(RESULTS_DIR, "eda_sarima_forecast_values.csv")
forecast_df.to_csv(forecast_csv_path, index=False)
print(f"Forecast values saved to: {forecast_csv_path}")

# Model summary
print("\nSARIMA Model Summary:")
print(fit.summary())

print("\nOutputs saved in 'results/'.")