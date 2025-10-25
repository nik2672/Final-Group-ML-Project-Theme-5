# readme.md

## Project Overview

ML project for 5G network performance analysis using GPS-tagged latency and throughput measurements. The codebase implements a data processing pipeline that transforms raw CSV files into ML-ready datasets through concatenation, cleaning, imputation, and feature engineering.

## Project Structure

```
├── src/
│   ├── preprocessing/     # Data preprocessing pipeline
│   ├── features/         # Feature engineering
│   └── models/           # ML models (clustering & forecasting)
│       ├── clustering/   # K-Means, DBSCAN, HDBSCAN, BIRCH, OPTICS
│       └── forecasting/  # XGBoost, ARIMA, SRNN, GRU, LSTM
├── notebooks/            # Jupyter notebooks for analysis/EDA
├── results/             # Model outputs and visualizations
│   ├── clustering/      # Cluster assignments and plots
│   └── forecasting/     # Predictions and metrics
├── docs/                # Project documentation
├── ui/                  # Web interface for model training
└── data/               # Data files (gitignored)
```

## Data Processing Pipeline

The pipeline must be executed in this specific order:

1. **Raw data concatenation** (`src/preprocessing/concatenate.py`)
   - Combines multiple CSV files from `data/` directory
   - Creates `day_id` from Year-Month-Date columns
   - Processes data by day (as required by assignment)
   - Outputs: `data/combined_raw.csv`

2. **Data cleaning** (`src/preprocessing/cleaned_dataset_no_imputation.py`)
   - Standardizes column schema to match target format
   - Converts units (sizes to MB, bitrates to Mbps)
   - Filters invalid GPS coordinates (removes values like 0.0, 99.999, 999.0)
   - Processes by day using `day_id`
   - Input: `data/combined_raw.csv`
   - Output: `data/processed_data_no_imputation.csv`

3. **Imputation** (`src/preprocessing/cleaned_dataset_with_imputation.py`)
   - Drops completely empty columns
   - Fills numeric columns with median
   - Fills categorical columns with "Unknown"
   - Removes constant columns
   - Processes by day using `day_id`
   - Input: `data/processed_data_no_imputation.csv`
   - Output: `data/processed_data_with_imputation.csv`

4. **Feature Engineering** (`src/features/leakage_sage_feature_engineering.py`)
   - Extracts temporal, spatial, and network performance features
   - Creates zone-level and day-level aggregations
   - Outputs 3 files optimized for different ML tasks
   - Input: `data/processed_data_with_imputation.csv`
   - Outputs:
     - `data/features_engineered_{train/test}_improved.csv` - Full feature set
     - `data/features_for_clustering_{train/test} improved.csv` - Clustering-optimized features
     - `data/features_for_forecasting_{train/test}_improved.csv` - Forecasting-optimized features

## Machine Learning Models

The project implements clustering and forecasting models for network performance analysis.

### Clustering Models

The project implements and compares 5 clustering algorithms:

**K-Means** (`kmeans_analysis.py`, `kmeans_final.py`):
- Partitioning method that groups data into k clusters
- Uses elbow method to determine optimal cluster count
- Identifies geographic and performance-based network zones
- Scripts: `kmeans_analysis.py` (exploration), `kmeans_final.py` (final model)

**DBSCAN** (`comparison.py`):
- Density-based clustering for arbitrary-shaped clusters
- Automatically detects outlier zones (anomalous network behavior)
- No need to pre-specify cluster count
- Useful for identifying problematic network areas

**HDBSCAN** (`hdbscan_main.py`):
- Hierarchical density-based clustering
- Better noise handling than DBSCAN
- Automatically determines optimal density thresholds
- Requires: `conda install -c conda-forge hdbscan`

**BIRCH** (`comparison.py`):
- Balanced Iterative Reducing and Clustering using Hierarchies
- Memory-efficient for large datasets
- Incrementally builds a cluster feature tree

**OPTICS** (`comparison.py`):
- Ordering Points To Identify Clustering Structure
- Extension of DBSCAN for varying density clusters
- Produces a reachability plot for cluster analysis

**Model Comparison** (`comparison.py`, `tuning.py`):
- `comparison.py`: Compares all 5 algorithms with standardized metrics
- `tuning.py`: Hyperparameter optimization for clustering models
- Evaluates using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score

### Forecasting Models

The project implements and compares multiple forecasting approaches:

**XGBoost** (`xgb_arima.py`):
- Gradient boosting regression for performance forecasting
- Uses engineered features: lag values, rolling averages, temporal features
- Provides feature importance analysis for interpretability
- Forecasts: `avg_latency`, `upload_bitrate`, `download_bitrate`
- Primary choice for multivariate forecasting

**ARIMA/SARIMA** (`xgb_arima.py`, `sarima_forecasting.py`):
- Statistical time series forecasting
- ARIMA(2,1,2) model for trend and autocorrelation
- SARIMA extends ARIMA with seasonal components
- Good baseline for univariate forecasting
- Statistical comparison benchmark

**Simple RNN (SRNN)** (`srnn_simple.py`):
- Recurrent Neural Network with basic architecture
- Captures temporal dependencies in time series
- Multiple training runs with different random seeds
- Outputs training history, predictions, and evaluation metrics

**GRU** (`gru_LEAN.py`):
- Gated Recurrent Unit neural network
- More efficient than LSTM with similar performance
- Better at capturing long-term dependencies than simple RNN
- LEAN architecture optimized for time series

**LSTM** (`lstm_model.py`):
- Long Short-Term Memory neural network
- Handles long-term dependencies and vanishing gradients
- Standard deep learning approach for sequence prediction
- Memory cells preserve information over time

**Model Comparison** (`comparison_forcasting.py`):
- Unified comparison framework for all forecasting models
- Standardizes metrics across different model outputs (CSV files)
- Generates per-target performance visualizations
- Identifies best model for each metric (MAE, RMSE, R²)

## Running the Pipeline

### Prerequisites: Data Preprocessing & Feature Engineering

Before running any ML models, you **must** complete the data pipeline:

```bash
# Install dependencies
pip install -r requirements.txt

# Step 0: Exploratory data analysis (optional)
jupyter notebook notebooks/partial_eda.ipynb

# Step 1: Concatenate raw data files
python src/preprocessing/concatenate.py

# Step 2: Clean and standardize data
python src/preprocessing/cleaned_dataset_no_imputation.py

# Step 3: Impute missing values
python src/preprocessing/cleaned_dataset_with_imputation.py

# Step 4: Feature engineering (REQUIRED before ML models)
python src/features/leakage_safe_feature_engineering.py
```

### Running ML Models

You have **two options** for running the ML models:

#### Option 1: Command Line (Backend Scripts)

Run models directly from the command line using Python scripts:

```bash
# Run all ML models individually (clustering + forecasting)
python src/models/clustering/*.py
python src/models/forecasting/*.py

# Or run specific models:
python src/models/clustering/kmeans_final.py
python src/models/forecasting/xgb_arima.py
```

#### Option 2: Web UI 

The UI provides an interactive interface to run models with custom hyperparameters, view results, and compare runs.

```bash
# Terminal 1 - Start backend server
cd ui/backend
python main.py

# Terminal 2 - Start frontend (in a new terminal)
cd ui/frontend
npm install  # First time only
npm start
```

**Access the UI:** Open browser at `http://localhost:3000`

**Features:**
- Select any of the 10 models (K-Means, DBSCAN, BIRCH, OPTICS, HDBSCAN, XGBoost, ARIMA, SARIMA, LSTM, GRU)
- Configure hyperparameters with sliders and inputs
- Choose target metrics for forecasting (avg_latency, avg_throughput, etc.)
- View results, plots, and metrics in real-time
- Track run history and compare experiments
- Data status indicator shows feature file availability

**See `ui/README.md` for detailed UI documentation and troubleshooting.**

### Optional: Quality Assurance

```bash
# Step 6: Quality assurance (optional)
jupyter notebook notebooks/data_assurance.ipynb
```

### ML Model Outputs

**Clustering results** (`results/clustering/`):
- `kmeans_clusters.png` - K-Means cluster visualization
- `dbscan_clusters.png` - DBSCAN cluster visualization
- `hdbscan_clusters.png` - HDBSCAN cluster visualization
- `birch_clusters.png` - BIRCH cluster visualization
- `optics_clusters.png` - OPTICS cluster visualization
- `elbow_analysis.png` - Optimal k selection for K-Means
- `all_algorithms_comparison.png` - Side-by-side algorithm comparison
- `algorithm_similarity_matrix.png` - Cluster assignment similarity heatmap
- `performance_summary.png` - Metrics comparison across all algorithms
- `*.csv` files - Cluster assignments and evaluation metrics

**Forecasting results** (`results/forecasting/`):

*XGBoost & ARIMA outputs:*
- `xgboost_avg_latency.png` - XGBoost predictions for latency
- `xgboost_upload_bitrate.png` - XGBoost predictions for upload
- `xgboost_download_bitrate.png` - XGBoost predictions for download
- `arima_avg_latency.png` - ARIMA predictions for latency
- `arima_upload_bitrate.png` - ARIMA predictions for upload
- `arima_download_bitrate.png` - ARIMA predictions for download
- `feature_importance_*.png` - XGBoost feature importance (one per target metric)

*Neural Network (SRNN) outputs:*
- `srnn_engineered_predictions.png` (and variations 1, 2) - SRNN prediction plots
- `srnn_engineered_training_history.png` (and variations 1, 2) - Loss curves over epochs
- `srnn_engineered_evaluation_metrics.csv` (and variations 1, 2) - Performance metrics per target

*Comparison outputs:*
- Generated by `comparison_forcasting.py` - Unified metrics comparing all models

## Dataset Schema

Final cleaned dataset contains these columns (in order):
- Time fields: `time`, `Convert_time`, `DATES`, `TIME`, `DAY`, `YEAR`, `MONTH`, `DATE`, `HOUR`, `MIN`, `SEC`
- GPS: `latitude`, `longitude`
- Server latency: `svr1`, `svr2`, `svr3`, `svr4`
- Upload metrics: `upload_transfer_size_mbytes`, `upload_bitrate_mbits/sec`
- Download metrics: `download_transfer_size_rx_mbytes`, `download_bitrate_rx_mbits/sec`
- Other: `application_data`, `square_id`

All numeric columns use consistent units (MB for sizes, Mbps for bitrates, milliseconds for latency).

## Key Data Quality Rules

- **Invalid GPS values**: `{0.0, 99.999, 99.9999, 999.0, 999.999}` are placeholder errors and must be filtered
- **GPS range validation**: Latitude must be in [-90, 90], longitude in [-180, 180]
- **No negative values** allowed in latency or throughput metrics
- **Duplicate handling**: Exact duplicates are data collection errors (found 109,379 in dataset) and should be removed
- **Outliers**: Rows with z-score > 3 in latency/throughput columns are flagged but retained

## Path Configuration

All scripts use absolute paths derived from `os.path.dirname(os.path.abspath(__file__))` to ensure portability. The `data/` folder is always resolved relative to the script location.

## Dependencies

Core libraries (install via `pip install -r requirements.txt`):
- **pandas** (≥1.5.0) - Data manipulation and CSV processing
- **numpy** (≥1.23.0) - Numerical computations
- **scikit-learn** (≥1.2.0) - Preprocessing, clustering (K-Means, DBSCAN, BIRCH, OPTICS), evaluation metrics
- **xgboost** (≥1.7.0) - Gradient boosting for forecasting
- **statsmodels** (≥0.14.0) - ARIMA/SARIMA time series forecasting
- **tensorflow** (≥2.10.0) - Deep learning models (LSTM, GRU, Simple RNN)
- **matplotlib** (≥3.6.0) - Visualization and plotting
- **scipy** (≥1.10.0) - Statistical analysis (for data_assurance.ipynb)
- **jupyter** (≥1.0.0) - Interactive notebooks for EDA
- **tqdm** (≥4.65.0) - Progress bars for long-running operations

**Optional (requires conda):**
- **hdbscan** - Hierarchical DBSCAN clustering
  ```bash
  conda install -c conda-forge hdbscan
  ```

## Model Evaluation Metrics

**Clustering:**
- **Silhouette Score** (0-1, higher is better): Measures cluster cohesion and separation
- **Davies-Bouldin Index** (≥0, lower is better): Within-cluster to between-cluster distance ratio
- **Calinski-Harabasz Score** (≥0, higher is better): Ratio of between-cluster to within-cluster dispersion
- **Inertia** (≥0, lower is better): Within-cluster sum of squared distances (K-Means only)

**Forecasting:**
- **MAE (Mean Absolute Error)** (≥0, lower is better): Average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error)** (≥0, lower is better): Square root of average squared differences, penalizes large errors more heavily
- **R² (R-squared)** (-∞ to 1, higher is better): Proportion of variance explained by the model (1.0 = perfect fit, 0.0 = baseline)
- **MAPE (Mean Absolute Percentage Error)** (≥0%, lower is better): Average percentage deviation from actual values