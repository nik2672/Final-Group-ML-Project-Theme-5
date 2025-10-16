# Machine Learning Models for 5G Network Performance Analysis

This directory contains implementations of clustering and forecasting models for analyzing 5G network performance based on GPS-tagged latency and throughput measurements.

## Models Implemented

### Clustering Models (`clustering/main.py`)

**1. K-Means Clustering**
- Uses the elbow method to automatically determine optimal number of clusters
- Evaluates clusters using:
  - Silhouette Score (higher is better)
  - Davies-Bouldin Index (lower is better)
  - Inertia (within-cluster sum of squares)
- Identifies geographic and performance-based network zones

**2. DBSCAN Clustering**
- Density-based clustering for finding odd-shaped clusters
- Automatically detects outlier zones
- No need to specify number of clusters beforehand
- Good for identifying anomalous network behavior

### Forecasting Models (`forecasting/main.py`)

**1. XGBoost Regression**
- Gradient boosting model for performance forecasting
- Uses engineered features:
  - Temporal features (hour, day, peak hours)
  - Lag values (previous hour metrics)
  - Rolling averages
  - Spatial features (zone aggregates)
- Provides feature importance analysis

**2. ARIMA (AutoRegressive Integrated Moving Average)**
- Statistical baseline for time series forecasting
- Good for capturing temporal patterns in single metrics
- Uses order (p=2, d=1, q=2) by default

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Data Requirements

Before running the models, ensure you have completed the data pipeline:

```bash
# Step 1: Concatenate raw data
python src/preprocessing/concatenate.py

# Step 2: Clean data
python src/preprocessing/cleaned_dataset_no_imputation.py

# Step 3: Impute missing values
python src/preprocessing/cleaned_dataset_with_imputation.py

# Step 4: Feature engineering (REQUIRED for models)
python src/features/feature_engineering.py
```

This will generate:
- `data/features_for_clustering.csv` - Optimized for clustering
- `data/features_for_forecasting.csv` - Optimized for forecasting

## Usage

### Run Clustering Models

```bash
python src/models/clustering/main.py
```

**Output:**
- `results/clustering/elbow_analysis.png` - Elbow method plots
- `results/clustering/kmeans_clusters.png` - K-Means cluster visualization
- `results/clustering/dbscan_clusters.png` - DBSCAN cluster visualization
- `results/clustering/kmeans_clusters.csv` - Cluster assignments (K-Means)
- `results/clustering/dbscan_clusters.csv` - Cluster assignments (DBSCAN)

### Run Forecasting Models

```bash
python src/models/forecasting/main.py
```

**Output:**
- `results/forecasting/xgboost_*.png` - XGBoost predictions and residuals
- `results/forecasting/feature_importance_*.png` - Feature importance plots
- `results/forecasting/arima_*.png` - ARIMA forecasts
- `results/forecasting/model_comparison.csv` - Performance metrics comparison

## Model Details

### Clustering Features Used

- **Spatial:** latitude, longitude, location_hash
- **Performance:** avg_latency, std_latency, total_throughput
- **Zone Aggregates:** zone_avg_latency, zone_avg_upload, zone_avg_download

### Forecasting Targets

The models forecast three key performance metrics:
1. **avg_latency** - Average network latency (ms)
2. **upload_bitrate** - Upload throughput (Mbps)
3. **download_bitrate** - Download throughput (Mbps)

### Evaluation Metrics

**Clustering:**
- Silhouette Score: Measures cluster cohesion and separation
- Davies-Bouldin Index: Ratio of within-cluster to between-cluster distances
- Cluster size distribution

**Forecasting:**
- MAE (Mean Absolute Error): Average prediction error
- RMSE (Root Mean Squared Error): Penalizes larger errors more
- R² Score: Proportion of variance explained (XGBoost only)

## Customization

### Adjusting Clustering Parameters

Edit `src/models/clustering/main.py`:

```python
# K-Means: Change max clusters to test
optimal_k = elbow_method(X_scaled, max_k=15)  # Default: 10

# DBSCAN: Tune epsilon and minimum samples
dbscan_model, labels = run_dbscan(X_scaled, eps=0.3, min_samples=15)
```

### Adjusting Forecasting Parameters

Edit `src/models/forecasting/main.py`:

```python
# XGBoost: Modify hyperparameters
model = XGBRegressor(
    n_estimators=200,      # Default: 100
    learning_rate=0.05,    # Default: 0.1
    max_depth=8,           # Default: 6
    random_state=42
)

# ARIMA: Change order (p, d, q)
arima_model, forecast = train_arima(ts, order=(3, 1, 3))  # Default: (2,1,2)
```

## Interpreting Results

### K-Means Clusters
- Lower cluster IDs typically represent better performance zones
- Review cluster characteristics in console output
- Use cluster assignments for network planning and optimization

### DBSCAN Outliers
- Points labeled as -1 are outliers (anomalous zones)
- High outlier percentage may indicate noisy data or need for parameter tuning

### XGBoost Feature Importance
- Shows which features most impact predictions
- Lag features typically have high importance for time series
- Helps identify key drivers of network performance

### ARIMA Forecasts
- Best for short-term predictions
- May not capture complex feature interactions like XGBoost
- Good statistical baseline for comparison

## Troubleshooting

**Issue:** `FileNotFoundError: features_for_*.csv not found`
- **Solution:** Run feature engineering first: `python src/features/feature_engineering.py`

**Issue:** DBSCAN finds only 1 cluster or all noise
- **Solution:** Adjust `eps` parameter (try values between 0.1 and 1.0)

**Issue:** XGBoost poor performance (low R²)
- **Solution:** Check for sufficient training data and feature quality

**Issue:** ARIMA convergence warnings
- **Solution:** Try different ARIMA orders or check for stationarity

## Next Steps

1. **Hyperparameter Tuning:** Use GridSearchCV for XGBoost optimization
2. **Advanced Models:** Try LSTM for time series or Hierarchical clustering
3. **Ensemble Methods:** Combine multiple forecasts for better accuracy
4. **Real-time Prediction:** Deploy models for live network monitoring
