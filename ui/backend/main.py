"""
FastAPI backend for running ML models via the UI.
Executes clustering and forecasting models with user-defined hyperparameters.
"""

import os
import sys
import time
import traceback
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess

# Set matplotlib backend to non-interactive before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

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
    target_metric: Optional[str] = "avg_latency"  # For forecasting models


@app.get("/")
def read_root():
    return {"message": "5G ML Model Runner API", "status": "running"}


@app.get("/api/data-status")
def check_data_status():
    """Check if required data files exist."""
    clustering_train_path = PROJECT_ROOT / 'data' / 'features_for_clustering_train_improved.csv'
    clustering_test_path = PROJECT_ROOT / 'data' / 'features_for_clustering_test_improved.csv'
    forecasting_path = PROJECT_ROOT / 'data' / 'features_for_forecasting.csv'

    return {
        "clustering_train_data": clustering_train_path.exists(),
        "clustering_test_data": clustering_test_path.exists(),
        "forecasting_data": forecasting_path.exists(),
        "clustering_train_path": str(clustering_train_path),
        "clustering_test_path": str(clustering_test_path),
        "forecasting_path": str(forecasting_path)
    }


@app.get("/api/results/{filename}")
def get_result_file(filename: str):
    """Serve result files (images, CSV)."""
    results_dir = PROJECT_ROOT / 'results'

    # Search in clustering and forecasting subdirectories
    for subdir in ['clustering', 'forecasting']:
        file_path = results_dir / subdir / filename
        if file_path.exists():
            return FileResponse(file_path)

    raise HTTPException(status_code=404, detail="File not found")


@app.post("/api/run-model")
async def run_model(request: ModelRequest):
    """
    Execute ML model with specified hyperparameters.
    """
    model = request.model
    params = request.hyperparameters
    target_metric = request.target_metric

    start_time = time.time()

    try:
        if model == "kmeans":
            result = run_kmeans(params)
        elif model == "dbscan":
            result = run_dbscan(params)
        elif model == "birch":
            result = run_birch(params)
        elif model == "optics":
            result = run_optics(params)
        elif model == "hdbscan":
            result = run_hdbscan(params)
        elif model == "xgboost":
            result = run_xgboost(params, target_metric)
        elif model == "arima":
            result = run_arima(params, target_metric)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        
        # Only set status to success if not already set (e.g., HDBSCAN error)
        if "status" not in result:
            result["status"] = "success"
            
        result["hyperparameters"] = params
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        return result

    except Exception as e:
        execution_time = time.time() - start_time
        error_trace = traceback.format_exc()

        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "stack_trace": error_trace,
            "execution_time": execution_time
        }


def load_and_prepare_clustering_data():
    """Helper function to load and prepare both train and test clustering data."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    train_path = PROJECT_ROOT / 'data' / 'features_for_clustering_train_improved.csv'
    test_path = PROJECT_ROOT / 'data' / 'features_for_clustering_test_improved.csv'
    
    if not train_path.exists():
        raise FileNotFoundError("Training clustering features not found. Run feature engineering first.")
    
    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False) if test_path.exists() else None
    
    # Aggregate train data by zone
    if 'square_id' in df_train.columns:
        zone_agg = df_train.groupby('square_id').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'avg_latency': 'mean',
            'std_latency': 'mean',
            'total_throughput': 'mean',
            'zone_avg_latency': 'first',
            'zone_avg_upload': 'first',
            'zone_avg_download': 'first'
        }).reset_index()
        df_train = zone_agg
    
    # Aggregate test data by zone
    if df_test is not None and 'square_id' in df_test.columns:
        zone_agg_test = df_test.groupby('square_id').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'avg_latency': 'mean',
            'std_latency': 'mean',
            'total_throughput': 'mean',
            'zone_avg_latency': 'first',
            'zone_avg_upload': 'first',
            'zone_avg_download': 'first'
        }).reset_index()
        df_test = zone_agg_test
    
    features_to_use = [
        'latitude', 'longitude', 'avg_latency', 'std_latency',
        'total_throughput', 'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    features_to_use = [f for f in features_to_use if f in df_train.columns]
    
    X_train = df_train[features_to_use].dropna()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test = None
    X_test_scaled = None
    if df_test is not None:
        X_test = df_test[features_to_use].dropna()
        X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_train_scaled, X_test, X_test_scaled, scaler


def run_kmeans(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run K-Means clustering model on both train and test data."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    # Load and prepare data
    X_train, X_train_scaled, X_test, X_test_scaled, scaler = load_and_prepare_clustering_data()

    # Run K-Means with Elbow Method (automatic k selection)
    max_k = int(params.get('max_k', 8))
    max_iter = int(params.get('max_iter', 300))
    random_state = int(params.get('random_state', 42))

    # Elbow method to find optimal k
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, max_iter=max_iter, random_state=random_state, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_train_scaled)
        sil_score_temp = silhouette_score(X_train_scaled, labels_temp)
        silhouette_scores.append(sil_score_temp)
    
    # Find optimal k
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    # Run final K-Means with optimal k
    kmeans = KMeans(
        n_clusters=optimal_k,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10
    )
    labels_train = kmeans.fit_predict(X_train_scaled)

    # Calculate train metrics
    sil_score_train = silhouette_score(X_train_scaled, labels_train)
    db_score_train = davies_bouldin_score(X_train_scaled, labels_train)
    ch_score_train = calinski_harabasz_score(X_train_scaled, labels_train)
    
    # Evaluate on test data if available
    test_metrics = None
    labels_test = None
    if X_test is not None:
        labels_test = kmeans.predict(X_test_scaled)
        
        # Only calculate metrics if we have enough clusters
        unique_labels_test = len(np.unique(labels_test))
        if unique_labels_test > 1:
            sil_score_test = silhouette_score(X_test_scaled, labels_test)
            db_score_test = davies_bouldin_score(X_test_scaled, labels_test)
            ch_score_test = calinski_harabasz_score(X_test_scaled, labels_test)
            
            test_metrics = {
                "silhouette_score": float(sil_score_test),
                "davies_bouldin_score": float(db_score_test),
                "calinski_harabasz_score": float(ch_score_test)
            }

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with cluster assignments (train)
    result_df_train = X_train.copy()
    result_df_train['cluster'] = labels_train
    csv_path_train = results_dir / 'kmeans_clusters_train.csv'
    result_df_train.to_csv(csv_path_train, index=False)
    
    # Save test CSV if available
    if labels_test is not None:
        result_df_test = X_test.copy()
        result_df_test['cluster'] = labels_test
        csv_path_test = results_dir / 'kmeans_clusters_test.csv'
        result_df_test.to_csv(csv_path_test, index=False)
    
    # Generate and save visualization (train)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=labels_train, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'K-Means Clustering - Train (n_clusters={optimal_k})')
    plt.xlabel(X_train.columns[0])
    plt.ylabel(X_train.columns[1])
    png_path_train = results_dir / 'kmeans_clusters_train.png'
    plt.savefig(png_path_train, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate test visualization if available
    if labels_test is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=labels_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'K-Means Clustering - Test (n_clusters={optimal_k})')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        png_path_test = results_dir / 'kmeans_clusters_test.png'
        plt.savefig(png_path_test, dpi=300, bbox_inches='tight')
        plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["kmeans_clusters_train.csv", "kmeans_clusters_train.png", 
                     "kmeans_clusters_test.csv", "kmeans_clusters_test.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    result = {
        "model": "kmeans",
        "train_metrics": {
            "silhouette_score": float(sil_score_train),
            "davies_bouldin_score": float(db_score_train),
            "calinski_harabasz_score": float(ch_score_train),
            "n_clusters": int(optimal_k),
            "inertia": float(kmeans.inertia_)
        },
        "output_files": output_files
    }
    
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    
    return result


def run_dbscan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run DBSCAN clustering model on both train and test data."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import matplotlib.pyplot as plt

    # Load and prepare data
    X_train, X_train_scaled, X_test, X_test_scaled, scaler = load_and_prepare_clustering_data()

    # Run DBSCAN
    eps = float(params.get('eps', 1.5))
    min_samples = int(params.get('min_samples', 5))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels_train = dbscan.fit_predict(X_train_scaled)

    n_clusters_train = len(set(labels_train)) - (1 if -1 in labels_train else 0)
    n_outliers_train = list(labels_train).count(-1)

    # Calculate train metrics (excluding outliers)
    mask_train = labels_train != -1
    sil_score_train = None
    db_score_train = None
    ch_score_train = None
    if mask_train.sum() > 0 and n_clusters_train > 1:
        sil_score_train = float(silhouette_score(X_train_scaled[mask_train], labels_train[mask_train]))
        db_score_train = float(davies_bouldin_score(X_train_scaled[mask_train], labels_train[mask_train]))
        ch_score_train = float(calinski_harabasz_score(X_train_scaled[mask_train], labels_train[mask_train]))

    # Evaluate on test data if available
    test_metrics = None
    labels_test = None
    if X_test is not None:
        labels_test = dbscan.fit_predict(X_test_scaled)
        n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        n_outliers_test = list(labels_test).count(-1)
        
        mask_test = labels_test != -1
        if mask_test.sum() > 0 and n_clusters_test > 1:
            sil_score_test = float(silhouette_score(X_test_scaled[mask_test], labels_test[mask_test]))
            db_score_test = float(davies_bouldin_score(X_test_scaled[mask_test], labels_test[mask_test]))
            ch_score_test = float(calinski_harabasz_score(X_test_scaled[mask_test], labels_test[mask_test]))
            
            test_metrics = {
                "silhouette_score": sil_score_test,
                "davies_bouldin_score": db_score_test,
                "calinski_harabasz_score": ch_score_test,
                "n_clusters": int(n_clusters_test),
                "n_outliers": int(n_outliers_test)
            }

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train CSV with cluster assignments
    result_df_train = X_train.copy()
    result_df_train['cluster'] = labels_train
    csv_path_train = results_dir / 'dbscan_clusters_train.csv'
    result_df_train.to_csv(csv_path_train, index=False)
    
    # Save test CSV if available
    if labels_test is not None:
        result_df_test = X_test.copy()
        result_df_test['cluster'] = labels_test
        csv_path_test = results_dir / 'dbscan_clusters_test.csv'
        result_df_test.to_csv(csv_path_test, index=False)
    
    # Generate train visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=labels_train, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'DBSCAN Clustering - Train (eps={eps}, min_samples={min_samples})')
    plt.xlabel(X_train.columns[0])
    plt.ylabel(X_train.columns[1])
    png_path_train = results_dir / 'dbscan_clusters_train.png'
    plt.savefig(png_path_train, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate test visualization if available
    if labels_test is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=labels_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'DBSCAN Clustering - Test (eps={eps}, min_samples={min_samples})')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        png_path_test = results_dir / 'dbscan_clusters_test.png'
        plt.savefig(png_path_test, dpi=300, bbox_inches='tight')
        plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["dbscan_clusters_train.csv", "dbscan_clusters_train.png",
                     "dbscan_clusters_test.csv", "dbscan_clusters_test.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    result = {
        "model": "dbscan",
        "train_metrics": {
            "silhouette_score": sil_score_train if sil_score_train else 0.0,
            "davies_bouldin_score": db_score_train if db_score_train else 0.0,
            "calinski_harabasz_score": ch_score_train if ch_score_train else 0.0,
            "n_clusters": int(n_clusters_train),
            "n_outliers": int(n_outliers_train)
        },
        "output_files": output_files
    }
    
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    
    return result


def run_birch(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Birch clustering model on both train and test data."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import Birch
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
    import matplotlib.pyplot as plt

    # Load and prepare data
    X_train, X_train_scaled, X_test, X_test_scaled, scaler = load_and_prepare_clustering_data()

    # Run Birch with automatic cluster detection
    max_clusters = int(params.get('max_clusters', 7))
    threshold = float(params.get('threshold', 0.3))
    branching_factor = int(params.get('branching_factor', 50))

    # Find optimal number of clusters by testing different values
    best_score = -1
    best_clusters = 2
    best_birch = None
    best_labels = None
    
    for n_clusters in range(2, max_clusters + 1):
        birch = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor
        )
        labels_temp = birch.fit_predict(X_train_scaled)
        
        # Only calculate silhouette if we have multiple clusters
        if len(np.unique(labels_temp)) > 1:
            score = silhouette_score(X_train_scaled, labels_temp)
            if score > best_score:
                best_score = score
                best_clusters = n_clusters
                best_birch = birch
                best_labels = labels_temp
    
    # Use the best configuration
    if best_birch is None:
        # Fallback to default if optimization fails
        best_birch = Birch(
            n_clusters=2,
            threshold=threshold,
            branching_factor=branching_factor
        )
        labels_train = best_birch.fit_predict(X_train_scaled)
        optimal_clusters = 2
    else:
        labels_train = best_labels
        optimal_clusters = best_clusters

    # Calculate train metrics
    sil_score_train = silhouette_score(X_train_scaled, labels_train)
    db_score_train = davies_bouldin_score(X_train_scaled, labels_train)
    ch_score_train = calinski_harabasz_score(X_train_scaled, labels_train)
    
    # Detect outliers using silhouette-based method
    sample_silhouette_values_train = silhouette_samples(X_train_scaled, labels_train)
    outlier_threshold = -0.1
    n_outliers_train = (sample_silhouette_values_train < outlier_threshold).sum()

    # Evaluate on test data if available
    test_metrics = None
    labels_test = None
    if X_test is not None:
        labels_test = best_birch.predict(X_test_scaled)
        
        if len(np.unique(labels_test)) > 1:
            sil_score_test = silhouette_score(X_test_scaled, labels_test)
            db_score_test = davies_bouldin_score(X_test_scaled, labels_test)
            ch_score_test = calinski_harabasz_score(X_test_scaled, labels_test)
            
            sample_silhouette_values_test = silhouette_samples(X_test_scaled, labels_test)
            n_outliers_test = (sample_silhouette_values_test < outlier_threshold).sum()
            
            test_metrics = {
                "silhouette_score": float(sil_score_test),
                "davies_bouldin_score": float(db_score_test),
                "calinski_harabasz_score": float(ch_score_test),
                "n_clusters": int(len(np.unique(labels_test))),
                "n_outliers": int(n_outliers_test)
            }

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train CSV with cluster assignments
    result_df_train = X_train.copy()
    result_df_train['cluster'] = labels_train
    result_df_train['silhouette_score'] = sample_silhouette_values_train
    result_df_train['is_outlier'] = sample_silhouette_values_train < outlier_threshold
    csv_path_train = results_dir / 'birch_clusters_train.csv'
    result_df_train.to_csv(csv_path_train, index=False)
    
    # Save test CSV if available
    if labels_test is not None:
        result_df_test = X_test.copy()
        result_df_test['cluster'] = labels_test
        sample_silhouette_values_test = silhouette_samples(X_test_scaled, labels_test)
        result_df_test['silhouette_score'] = sample_silhouette_values_test
        result_df_test['is_outlier'] = sample_silhouette_values_test < outlier_threshold
        csv_path_test = results_dir / 'birch_clusters_test.csv'
        result_df_test.to_csv(csv_path_test, index=False)
    
    # Generate train visualization
    plt.figure(figsize=(10, 8))
    colors_train = labels_train.copy().astype(float)
    colors_train[sample_silhouette_values_train < outlier_threshold] = -1
    scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=colors_train, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'Birch Clustering - Train (n_clusters={optimal_clusters}, threshold={threshold})')
    plt.xlabel(X_train.columns[0])
    plt.ylabel(X_train.columns[1])
    png_path_train = results_dir / 'birch_clusters_train.png'
    plt.savefig(png_path_train, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate test visualization if available
    if labels_test is not None:
        plt.figure(figsize=(10, 8))
        colors_test = labels_test.copy().astype(float)
        sample_silhouette_values_test = silhouette_samples(X_test_scaled, labels_test)
        colors_test[sample_silhouette_values_test < outlier_threshold] = -1
        scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=colors_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Birch Clustering - Test (n_clusters={optimal_clusters}, threshold={threshold})')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        png_path_test = results_dir / 'birch_clusters_test.png'
        plt.savefig(png_path_test, dpi=300, bbox_inches='tight')
        plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["birch_clusters_train.csv", "birch_clusters_train.png",
                     "birch_clusters_test.csv", "birch_clusters_test.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    result = {
        "model": "birch",
        "train_metrics": {
            "silhouette_score": float(sil_score_train),
            "davies_bouldin_score": float(db_score_train),
            "calinski_harabasz_score": float(ch_score_train),
            "n_clusters": int(optimal_clusters),
            "n_outliers": int(n_outliers_train)
        },
        "output_files": output_files
    }
    
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    
    return result


def run_optics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run OPTICS clustering model on both train and test data."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import OPTICS
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import matplotlib.pyplot as plt

    # Load and prepare data
    X_train, X_train_scaled, X_test, X_test_scaled, scaler = load_and_prepare_clustering_data()

    # Run OPTICS
    min_samples = int(params.get('min_samples', 5))
    max_eps = float(params.get('max_eps', 2.0))
    xi = float(params.get('xi', 0.1))
    cluster_method = params.get('cluster_method', 'xi')

    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        cluster_method=cluster_method,
        xi=xi if cluster_method == 'xi' else 0.05,
        n_jobs=-1
    )
    labels_train = optics.fit_predict(X_train_scaled)

    n_clusters_train = len(set(labels_train)) - (1 if -1 in labels_train else 0)
    n_outliers_train = list(labels_train).count(-1)

    # Calculate train metrics (excluding outliers)
    mask_train = labels_train != -1
    sil_score_train = None
    db_score_train = None
    ch_score_train = None
    if mask_train.sum() > 0 and n_clusters_train > 1:
        sil_score_train = float(silhouette_score(X_train_scaled[mask_train], labels_train[mask_train]))
        db_score_train = float(davies_bouldin_score(X_train_scaled[mask_train], labels_train[mask_train]))
        ch_score_train = float(calinski_harabasz_score(X_train_scaled[mask_train], labels_train[mask_train]))

    # Evaluate on test data if available
    test_metrics = None
    labels_test = None
    if X_test is not None:
        # For OPTICS, we need to refit on test data since it doesn't have a predict method
        optics_test = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            cluster_method=cluster_method,
            xi=xi if cluster_method == 'xi' else 0.05,
            n_jobs=-1
        )
        labels_test = optics_test.fit_predict(X_test_scaled)
        n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        n_outliers_test = list(labels_test).count(-1)
        
        mask_test = labels_test != -1
        if mask_test.sum() > 0 and n_clusters_test > 1:
            sil_score_test = float(silhouette_score(X_test_scaled[mask_test], labels_test[mask_test]))
            db_score_test = float(davies_bouldin_score(X_test_scaled[mask_test], labels_test[mask_test]))
            ch_score_test = float(calinski_harabasz_score(X_test_scaled[mask_test], labels_test[mask_test]))
            
            test_metrics = {
                "silhouette_score": sil_score_test,
                "davies_bouldin_score": db_score_test,
                "calinski_harabasz_score": ch_score_test,
                "n_clusters": int(n_clusters_test),
                "n_outliers": int(n_outliers_test)
            }

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train CSV
    result_df_train = X_train.copy()
    result_df_train['cluster'] = labels_train
    csv_path_train = results_dir / 'optics_clusters_train.csv'
    result_df_train.to_csv(csv_path_train, index=False)
    
    # Save test CSV if available
    if labels_test is not None:
        result_df_test = X_test.copy()
        result_df_test['cluster'] = labels_test
        csv_path_test = results_dir / 'optics_clusters_test.csv'
        result_df_test.to_csv(csv_path_test, index=False)
    
    # Generate train visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=labels_train, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'OPTICS Clustering - Train (min_samples={min_samples}, max_eps={max_eps})')
    plt.xlabel(X_train.columns[0])
    plt.ylabel(X_train.columns[1])
    png_path_train = results_dir / 'optics_clusters_train.png'
    plt.savefig(png_path_train, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate test visualization if available
    if labels_test is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=labels_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'OPTICS Clustering - Test (min_samples={min_samples}, max_eps={max_eps})')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        png_path_test = results_dir / 'optics_clusters_test.png'
        plt.savefig(png_path_test, dpi=300, bbox_inches='tight')
        plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["optics_clusters_train.csv", "optics_clusters_train.png",
                     "optics_clusters_test.csv", "optics_clusters_test.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    result = {
        "model": "optics",
        "train_metrics": {
            "silhouette_score": sil_score_train if sil_score_train else 0.0,
            "davies_bouldin_score": db_score_train if db_score_train else 0.0,
            "calinski_harabasz_score": ch_score_train if ch_score_train else 0.0,
            "n_clusters": int(n_clusters_train),
            "n_outliers": int(n_outliers_train)
        },
        "output_files": output_files
    }
    
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    
    return result


def run_hdbscan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run HDBSCAN clustering model on both train and test data."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import matplotlib.pyplot as plt

    try:
        import hdbscan
        use_real_hdbscan = True
    except ImportError:
        # Use DBSCAN as a fallback with hierarchical-like behavior
        from sklearn.cluster import DBSCAN
        use_real_hdbscan = False

    # Load and prepare data
    X_train, X_train_scaled, X_test, X_test_scaled, scaler = load_and_prepare_clustering_data()

    # Run HDBSCAN or fallback
    min_cluster_size = int(params.get('min_cluster_size', 8))
    min_samples = params.get('min_samples', 5)
    if min_samples is not None:
        min_samples = int(min_samples)

    if use_real_hdbscan:
        # Use real HDBSCAN if available
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
            core_dist_n_jobs=-1
        )
        labels_train = hdb.fit_predict(X_train_scaled)
        model_note = "HDBSCAN"
    else:
        # Fallback: Use DBSCAN with parameters derived from HDBSCAN params
        eps = max(0.3, min_cluster_size * 0.1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples or 5, n_jobs=-1)
        labels_train = dbscan.fit_predict(X_train_scaled)
        model_note = "HDBSCAN (DBSCAN Fallback)"

    n_clusters_train = len(set(labels_train)) - (1 if -1 in labels_train else 0)
    n_outliers_train = list(labels_train).count(-1)

    # Calculate train metrics (excluding outliers)
    mask_train = labels_train != -1
    sil_score_train = None
    db_score_train = None
    ch_score_train = None
    if mask_train.sum() > 0 and n_clusters_train > 1:
        sil_score_train = float(silhouette_score(X_train_scaled[mask_train], labels_train[mask_train]))
        db_score_train = float(davies_bouldin_score(X_train_scaled[mask_train], labels_train[mask_train]))
        ch_score_train = float(calinski_harabasz_score(X_train_scaled[mask_train], labels_train[mask_train]))

    # Evaluate on test data if available
    test_metrics = None
    labels_test = None
    if X_test is not None:
        if use_real_hdbscan:
            # HDBSCAN needs to refit on test data
            hdb_test = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0,
                cluster_selection_method="eom",
                core_dist_n_jobs=-1
            )
            labels_test = hdb_test.fit_predict(X_test_scaled)
        else:
            eps = max(0.3, min_cluster_size * 0.1)
            dbscan_test = DBSCAN(eps=eps, min_samples=min_samples or 5, n_jobs=-1)
            labels_test = dbscan_test.fit_predict(X_test_scaled)
        
        n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        n_outliers_test = list(labels_test).count(-1)
        
        mask_test = labels_test != -1
        if mask_test.sum() > 0 and n_clusters_test > 1:
            sil_score_test = float(silhouette_score(X_test_scaled[mask_test], labels_test[mask_test]))
            db_score_test = float(davies_bouldin_score(X_test_scaled[mask_test], labels_test[mask_test]))
            ch_score_test = float(calinski_harabasz_score(X_test_scaled[mask_test], labels_test[mask_test]))
            
            test_metrics = {
                "silhouette_score": sil_score_test,
                "davies_bouldin_score": db_score_test,
                "calinski_harabasz_score": ch_score_test,
                "n_clusters": int(n_clusters_test),
                "n_outliers": int(n_outliers_test)
            }

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train CSV
    result_df_train = X_train.copy()
    result_df_train['cluster'] = labels_train
    csv_path_train = results_dir / 'hdbscan_clusters_train.csv'
    result_df_train.to_csv(csv_path_train, index=False)
    
    # Save test CSV if available
    if labels_test is not None:
        result_df_test = X_test.copy()
        result_df_test['cluster'] = labels_test
        csv_path_test = results_dir / 'hdbscan_clusters_test.csv'
        result_df_test.to_csv(csv_path_test, index=False)
    
    # Generate train visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=labels_train, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{model_note} - Train (min_cluster_size={min_cluster_size})')
    plt.xlabel(X_train.columns[0])
    plt.ylabel(X_train.columns[1])
    png_path_train = results_dir / 'hdbscan_clusters_train.png'
    plt.savefig(png_path_train, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate test visualization if available
    if labels_test is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=labels_test, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'{model_note} - Test (min_cluster_size={min_cluster_size})')
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        png_path_test = results_dir / 'hdbscan_clusters_test.png'
        plt.savefig(png_path_test, dpi=300, bbox_inches='tight')
        plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["hdbscan_clusters_train.csv", "hdbscan_clusters_train.png",
                     "hdbscan_clusters_test.csv", "hdbscan_clusters_test.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    result = {
        "model": "hdbscan",
        "model_note": model_note,
        "train_metrics": {
            "silhouette_score": sil_score_train if sil_score_train else 0.0,
            "davies_bouldin_score": db_score_train if db_score_train else 0.0,
            "calinski_harabasz_score": ch_score_train if ch_score_train else 0.0,
            "n_clusters": int(n_clusters_train),
            "n_outliers": int(n_outliers_train)
        },
        "output_files": output_files
    }
    
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    
    return result


def run_xgboost(params: Dict[str, Any], target_metric: str = "avg_latency") -> Dict[str, Any]:
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

    target_col = target_metric
    if target_col not in df.columns:
        raise ValueError(f"Target metric '{target_col}' not found in data. Available: {list(df.columns)}")

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
        "target_metric": target_col,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "output_files": [f"xgboost_{target_col}.png", f"feature_importance_{target_col}.png"]
    }


def run_arima(params: Dict[str, Any], target_metric: str = "avg_latency") -> Dict[str, Any]:
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

    target_col = target_metric
    if target_col not in df.columns:
        raise ValueError(f"Target metric '{target_col}' not found in data. Available: {list(df.columns)}")

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
        "target_metric": target_col,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "aic": float(model_fit.aic),
            "bic": float(model_fit.bic)
        },
        "output_files": [f"arima_{target_col}.png"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
