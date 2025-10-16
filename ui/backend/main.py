"""
FastAPI backend for running ML models via the UI.
Executes clustering and forecasting models with user-defined hyperparameters.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="5G ML Model Runner API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelRequest(BaseModel):
    model: str
    hyperparameters: Dict[str, Any]


@app.get("/")
def read_root():
    return {"message": "5G ML Model Runner API", "status": "running"}


@app.post("/api/run-model")
async def run_model(request: ModelRequest):
    """
    Execute ML model with specified hyperparameters.
    """
    model = request.model
    params = request.hyperparameters

    start_time = time.time()

    try:
        if model == "kmeans":
            result = run_kmeans(params)
        elif model == "dbscan":
            result = run_dbscan(params)
        elif model == "xgboost":
            result = run_xgboost(params)
        elif model == "arima":
            result = run_arima(params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        result["status"] = "success"

        return result

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "status": "error",
            "message": str(e),
            "execution_time": execution_time
        }


def run_kmeans(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run K-Means clustering model."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    # Load data
    data_path = PROJECT_ROOT / 'data' / 'features_for_clustering.csv'
    if not data_path.exists():
        raise FileNotFoundError("Clustering features not found. Run feature engineering first.")

    df = pd.read_csv(data_path, low_memory=False)

    # Aggregate by zone
    if 'square_id' in df.columns:
        zone_agg = df.groupby('square_id').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'avg_latency': 'mean',
            'std_latency': 'mean',
            'total_throughput': 'mean',
            'zone_avg_latency': 'first',
            'zone_avg_upload': 'first',
            'zone_avg_download': 'first'
        }).reset_index()
        df = zone_agg

    features_to_use = [
        'latitude', 'longitude', 'avg_latency', 'std_latency',
        'total_throughput', 'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    features_to_use = [f for f in features_to_use if f in df.columns]

    X = df[features_to_use].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run K-Means
    n_clusters = int(params.get('n_clusters', 5))
    max_iter = int(params.get('max_iter', 300))
    random_state = int(params.get('random_state', 42))

    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)

    # Calculate metrics
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    return {
        "model": "kmeans",
        "metrics": {
            "silhouette_score": float(sil_score),
            "davies_bouldin_score": float(db_score),
            "n_clusters": int(n_clusters),
            "inertia": float(kmeans.inertia_)
        },
        "output_files": ["kmeans_clusters.csv", "kmeans_clusters.png"]
    }


def run_dbscan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run DBSCAN clustering model."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    # Load data
    data_path = PROJECT_ROOT / 'data' / 'features_for_clustering.csv'
    if not data_path.exists():
        raise FileNotFoundError("Clustering features not found. Run feature engineering first.")

    df = pd.read_csv(data_path, low_memory=False)

    # Aggregate by zone
    if 'square_id' in df.columns:
        zone_agg = df.groupby('square_id').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'avg_latency': 'mean',
            'std_latency': 'mean',
            'total_throughput': 'mean',
            'zone_avg_latency': 'first',
            'zone_avg_upload': 'first',
            'zone_avg_download': 'first'
        }).reset_index()
        df = zone_agg

    features_to_use = [
        'latitude', 'longitude', 'avg_latency', 'std_latency',
        'total_throughput', 'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    features_to_use = [f for f in features_to_use if f in df.columns]

    X = df[features_to_use].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run DBSCAN
    eps = float(params.get('eps', 1.5))
    min_samples = int(params.get('min_samples', 5))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)

    # Calculate silhouette score (excluding outliers)
    mask = labels != -1
    sil_score = None
    if mask.sum() > 0 and n_clusters > 1:
        sil_score = float(silhouette_score(X_scaled[mask], labels[mask]))

    return {
        "model": "dbscan",
        "metrics": {
            "n_clusters": int(n_clusters),
            "n_outliers": int(n_outliers),
            "silhouette_score": sil_score if sil_score else 0.0
        },
        "output_files": ["dbscan_clusters.csv", "dbscan_clusters.png"]
    }


def run_xgboost(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run XGBoost forecasting model."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor

    # Load data
    data_path = PROJECT_ROOT / 'data' / 'features_for_forecasting.csv'
    if not data_path.exists():
        raise FileNotFoundError("Forecasting features not found. Run feature engineering first.")

    df = pd.read_csv(data_path, low_memory=False)

    target_col = 'avg_latency'
    df_clean = df.dropna(subset=[target_col])

    # Prepare features
    exclude_cols = [target_col, 'square_id', 'day', 'day_id']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    X = df_clean[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(0)
    y = df_clean[target_col]

    # Train-test split
    test_size = float(params.get('test_size', 0.2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train XGBoost
    n_estimators = int(params.get('n_estimators', 100))
    learning_rate = float(params.get('learning_rate', 0.1))
    max_depth = int(params.get('max_depth', 6))

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model": "xgboost",
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "output_files": ["xgboost_avg_latency.png", "feature_importance_avg_latency.png"]
    }


def run_arima(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run ARIMA forecasting model."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')

    # Load data
    data_path = PROJECT_ROOT / 'data' / 'features_for_forecasting.csv'
    if not data_path.exists():
        raise FileNotFoundError("Forecasting features not found. Run feature engineering first.")

    df = pd.read_csv(data_path, low_memory=False)

    target_col = 'avg_latency'
    df_clean = df.dropna(subset=[target_col])

    # Sample data
    sample_size = int(params.get('sample_size', 50000))
    if len(df_clean) > sample_size:
        df_clean = df_clean.sample(n=sample_size, random_state=42)

    if 'hour' in df_clean.columns:
        df_clean = df_clean.sort_values('hour')

    ts = df_clean[target_col].values

    # Train ARIMA
    p = int(params.get('p', 2))
    d = int(params.get('d', 1))
    q = int(params.get('q', 2))
    forecast_steps = int(params.get('forecast_steps', 50))

    train_size = int(len(ts) * 0.8)
    train_data = ts[:train_size]
    test_data = ts[train_size:]

    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()

    forecast_steps = min(forecast_steps, len(test_data))
    forecast = model_fit.forecast(steps=forecast_steps)

    # Metrics
    test_subset = test_data[:forecast_steps]
    mae = mean_absolute_error(test_subset, forecast)
    rmse = np.sqrt(mean_squared_error(test_subset, forecast))

    return {
        "model": "arima",
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "aic": float(model_fit.aic),
            "bic": float(model_fit.bic)
        },
        "output_files": ["arima_avg_latency.png"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
