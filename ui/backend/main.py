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
    clustering_path = PROJECT_ROOT / 'data' / 'features_for_clustering.csv'
    forecasting_path = PROJECT_ROOT / 'data' / 'features_for_forecasting.csv'

    return {
        "clustering_data": clustering_path.exists(),
        "forecasting_data": forecasting_path.exists(),
        "clustering_path": str(clustering_path),
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

    # Run K-Means with Elbow Method (automatic k selection)
    max_k = int(params.get('max_k', 8))
    max_iter = int(params.get('max_iter', 300))
    random_state = int(params.get('random_state', 42))

    # Elbow method to find optimal k
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, max_iter=max_iter, random_state=random_state, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        sil_score_temp = silhouette_score(X_scaled, labels_temp)
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
    labels = kmeans.fit_predict(X_scaled)

    # Calculate metrics
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    
    # Add Calinski-Harabasz score for comprehensive evaluation
    from sklearn.metrics import calinski_harabasz_score
    ch_score = calinski_harabasz_score(X_scaled, labels)

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with cluster assignments
    result_df = X.copy()
    result_df['cluster'] = labels
    csv_path = results_dir / 'kmeans_clusters.csv'
    result_df.to_csv(csv_path, index=False)
    
    # Generate and save visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'K-Means Clustering (n_clusters={optimal_k})')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    png_path = results_dir / 'kmeans_clusters.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["kmeans_clusters.csv", "kmeans_clusters.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    return {
        "model": "kmeans",
        "metrics": {
            "silhouette_score": float(sil_score),
            "davies_bouldin_score": float(db_score),
            "calinski_harabasz_score": float(ch_score),
            "n_clusters": int(optimal_k),
            "inertia": float(kmeans.inertia_)
        },
        "output_files": output_files
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

    # Calculate silhouette score and Davies-Bouldin score (excluding outliers)
    mask = labels != -1
    sil_score = None
    db_score = None
    if mask.sum() > 0 and n_clusters > 1:
        sil_score = float(silhouette_score(X_scaled[mask], labels[mask]))
        # Import davies_bouldin_score
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        db_score = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
        ch_score = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))
    else:
        ch_score = 0.0

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with cluster assignments
    result_df = X.copy()
    result_df['cluster'] = labels
    csv_path = results_dir / 'dbscan_clusters.csv'
    result_df.to_csv(csv_path, index=False)
    
    # Generate and save visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    png_path = results_dir / 'dbscan_clusters.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["dbscan_clusters.csv", "dbscan_clusters.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    return {
        "model": "dbscan",
        "metrics": {
            "silhouette_score": sil_score if sil_score else 0.0,
            "davies_bouldin_score": db_score if db_score else 0.0,
            "calinski_harabasz_score": ch_score if ch_score else 0.0,
            "n_clusters": int(n_clusters),
            "n_outliers": int(n_outliers)
        },
        "output_files": output_files
    }


def run_birch(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Birch clustering model."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import Birch
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
        labels_temp = birch.fit_predict(X_scaled)
        
        # Only calculate silhouette if we have multiple clusters
        if len(np.unique(labels_temp)) > 1:
            score = silhouette_score(X_scaled, labels_temp)
            if score > best_score:
                best_score = score
                best_clusters = n_clusters
                best_birch = birch
                best_labels = labels_temp
    
    # Use the best configuration
    if best_birch is None:
        # Fallback to default if optimization fails
        birch = Birch(
            n_clusters=2,
            threshold=threshold,
            branching_factor=branching_factor
        )
        labels = birch.fit_predict(X_scaled)
        optimal_clusters = 2
    else:
        labels = best_labels
        optimal_clusters = best_clusters

    # Calculate metrics
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    # Add Calinski-Harabasz score for comprehensive evaluation
    from sklearn.metrics import calinski_harabasz_score
    ch_score = calinski_harabasz_score(X_scaled, labels)
    
    # Detect outliers using silhouette-based method
    import numpy as np
    from sklearn.metrics import silhouette_samples
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    
    # Consider points with very low silhouette scores as outliers
    outlier_threshold = -0.1  # Points with silhouette score < -0.1 are potential outliers
    n_outliers = (sample_silhouette_values < outlier_threshold).sum()

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with cluster assignments and silhouette scores
    result_df = X.copy()
    result_df['cluster'] = labels
    result_df['silhouette_score'] = sample_silhouette_values
    result_df['is_outlier'] = sample_silhouette_values < outlier_threshold
    csv_path = results_dir / 'birch_clusters.csv'
    result_df.to_csv(csv_path, index=False)
    
    # Generate and save visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    
    # Color by cluster, but highlight outliers
    colors = labels.copy().astype(float)
    colors[sample_silhouette_values < outlier_threshold] = -1  # Mark outliers
    
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'Birch Clustering (n_clusters={optimal_clusters}, threshold={threshold})\nOutliers in dark')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    png_path = results_dir / 'birch_clusters.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["birch_clusters.csv", "birch_clusters.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    return {
        "model": "birch",
        "metrics": {
            "silhouette_score": float(sil_score),
            "davies_bouldin_score": float(db_score),
            "calinski_harabasz_score": float(ch_score),
            "n_clusters": int(optimal_clusters),
            "n_outliers": int(n_outliers)
        },
        "output_files": output_files
    }


def run_optics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run OPTICS clustering model."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import OPTICS
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

    # Run OPTICS
    min_samples = int(params.get('min_samples', 5))
    max_eps = float(params.get('max_eps', 2.0))
    xi = float(params.get('xi', 0.1))

    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        cluster_method='xi',
        xi=xi,
        n_jobs=-1
    )
    labels = optics.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)

    # Calculate silhouette score and Davies-Bouldin score (excluding outliers)
    mask = labels != -1
    sil_score = None
    db_score = None
    if mask.sum() > 0 and n_clusters > 1:
        sil_score = float(silhouette_score(X_scaled[mask], labels[mask]))
        # Import davies_bouldin_score
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        db_score = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
        ch_score = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))
    else:
        ch_score = 0.0

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with cluster assignments
    result_df = X.copy()
    result_df['cluster'] = labels
    csv_path = results_dir / 'optics_clusters.csv'
    result_df.to_csv(csv_path, index=False)
    
    # Generate and save visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'OPTICS Clustering (min_samples={min_samples}, max_eps={max_eps})')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    png_path = results_dir / 'optics_clusters.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["optics_clusters.csv", "optics_clusters.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    return {
        "model": "optics",
        "metrics": {
            "silhouette_score": sil_score if sil_score else 0.0,
            "davies_bouldin_score": db_score if db_score else 0.0,
            "calinski_harabasz_score": ch_score if ch_score else 0.0,
            "n_clusters": int(n_clusters),
            "n_outliers": int(n_outliers)
        },
        "output_files": output_files
    }


def run_hdbscan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run HDBSCAN clustering model."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    try:
        import hdbscan
        use_real_hdbscan = True
    except ImportError:
        # Use DBSCAN as a fallback with hierarchical-like behavior
        from sklearn.cluster import DBSCAN
        use_real_hdbscan = False

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
        labels = hdb.fit_predict(X_scaled)
        model_note = "HDBSCAN"
    else:
        # Fallback: Use DBSCAN with parameters derived from HDBSCAN params
        # Convert min_cluster_size to eps approximation and use min_samples
        eps = max(0.3, min_cluster_size * 0.1)  # Rough conversion
        dbscan = DBSCAN(eps=eps, min_samples=min_samples or 5, n_jobs=-1)
        labels = dbscan.fit_predict(X_scaled)
        model_note = "HDBSCAN (DBSCAN Fallback)"

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)

    # Calculate silhouette score and Davies-Bouldin score (excluding outliers)
    mask = labels != -1
    sil_score = None
    db_score = None
    ch_score = None
    if mask.sum() > 0 and n_clusters > 1:
        sil_score = float(silhouette_score(X_scaled[mask], labels[mask]))
        # Import davies_bouldin_score
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        db_score = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
        ch_score = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))
    else:
        ch_score = 0.0

    # Save results to files
    results_dir = PROJECT_ROOT / 'results' / 'clustering'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with cluster assignments
    result_df = X.copy()
    result_df['cluster'] = labels
    csv_path = results_dir / 'hdbscan_clusters.csv'
    result_df.to_csv(csv_path, index=False)
    
    # Generate and save visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{model_note} (min_cluster_size={min_cluster_size})')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    png_path = results_dir / 'hdbscan_clusters.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Check which output files actually exist
    output_files = []
    for filename in ["hdbscan_clusters.csv", "hdbscan_clusters.png"]:
        if (results_dir / filename).exists():
            output_files.append(filename)

    return {
        "model": "hdbscan",
        "model_implementation": model_note,
        "metrics": {
            "silhouette_score": sil_score if sil_score else 0.0,
            "davies_bouldin_score": db_score if db_score else 0.0,
            "calinski_harabasz_score": ch_score if ch_score else 0.0,
            "n_clusters": int(n_clusters),
            "n_outliers": int(n_outliers)
        },
        "output_files": output_files
    }


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
