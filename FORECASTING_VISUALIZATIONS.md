# Forecasting Model Visualizations

## Overview
All 5 forecasting models now generate PNG visualizations showing predicted vs actual values for the test set.

## Implementation Details

### Visualization Helper Function
**Location**: `ui/backend/main.py` (lines ~215-245)

```python
def save_forecasting_plot(y_true, y_pred, model_name, target_metric, max_points=500)
```

**Features**:
- Uses matplotlib Agg backend (non-interactive, server-safe)
- Creates `PROJECT_ROOT/results/` directory if needed
- Plots first 500 points for readability
- Saves as `{model_name}_{target_metric}.png` (150 DPI)
- Returns filename for UI to display
- Error handling with try/except

**Visualization Style**:
- Blue solid line: Actual values (alpha=0.7)
- Red dashed line: Predicted values
- Grid enabled (alpha=0.3)
- Title: "{MODEL} Forecast - {metric}"
- Axes labeled with time steps and metric name

## Models Updated

### 1. XGBoost (`run_xgboost`)
- **Lines**: ~948-1030
- **Visualization**: Direct predictions vs actual on test set
- **File**: `xgboost_{target_metric}.png`

### 2. ARIMA (`run_arima`)
- **Lines**: ~1085-1180
- **Visualization**: Forecasted steps vs actual time series
- **File**: `arima_{target_metric}.png`
- **Note**: Plots only `forecast_steps` points (default 100)

### 3. SARIMA (`run_sarima`)
- **Lines**: ~1180-1270
- **Visualization**: Seasonal forecast vs actual
- **File**: `sarima_{target_metric}.png`
- **Note**: Plots only `forecast_steps` points (default 100)

### 4. LSTM (`run_lstm`)
- **Lines**: ~1270-1410
- **Visualization**: Sequence predictions vs actual
- **File**: `lstm_{target_metric}.png`
- **Note**: Uses flattened arrays from inverse_transform

### 5. GRU (`run_gru`)
- **Lines**: ~1410-1640
- **Visualization**: Sequence predictions vs actual
- **File**: `gru_{target_metric}.png`
- **Note**: Uses numpy arrays from PyTorch predictions

## API Response Format

All forecasting models now return:

```json
{
  "model": "model_name",
  "target_metric": "avg_latency",
  "train_metrics": {
    "mae": float,
    "rmse": float,
    "r2": float,
    "n_samples": int
  },
  "test_metrics": {
    "mae": float,
    "rmse": float,
    "r2": float,
    "n_samples": int
  },
  "output_files": ["model_name_avg_latency.png"]
}
```

## Frontend Display

**Location**: `ui/frontend/src/components/ResultsDisplay.js`

The UI will display:
1. Train/test metrics comparison
2. Overfitting alert if train R² >> test R²
3. PNG visualization loaded from `/api/results/{filename}`

## File Locations

**Saved PNGs**: `c:\Users\13min\Final-Group-ML-Project-Theme-5\results\`

Example files:
- `xgboost_avg_latency.png`
- `arima_avg_latency.png`
- `sarima_avg_latency.png`
- `lstm_avg_latency.png`
- `gru_avg_latency.png`

## Usage

Run any forecasting model from the UI:

1. Select model (XGBoost, ARIMA, SARIMA, LSTM, or GRU)
2. Configure hyperparameters
3. Click "Run Model"
4. View metrics and visualization in results panel

The visualization will automatically show:
- **Blue line**: What actually happened in the test set
- **Red line**: What the model predicted
- **Alignment**: How closely predictions match reality

## Benefits

1. **Visual validation**: Quickly see if predictions follow actual trends
2. **Pattern detection**: Identify systematic over/under-prediction
3. **Temporal analysis**: See if model captures time-based patterns
4. **Model comparison**: Compare prediction quality across different models
5. **Presentation-ready**: High-quality PNGs for reports/presentations

## Technical Notes

- **Performance**: Only first 500 points plotted to keep files small
- **Memory**: Matplotlib uses Agg backend (no GUI overhead)
- **Safety**: All file operations wrapped in try/except
- **Concurrency**: Each model run creates unique filename with metric name
- **Scalability**: Works with any target metric (avg_latency, avg_throughput, etc.)
