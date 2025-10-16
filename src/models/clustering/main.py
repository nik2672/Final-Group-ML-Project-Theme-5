"""
Unified Clustering for 5G network performance analysis.
Implements K-Means (with elbow method) and DBSCAN for zone segmentation.
Auto-detects M-series chips for optimization.
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
from tqdm import tqdm
import multiprocessing

# Config paths
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'clustering')

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Detect M-series chip
N_CORES = multiprocessing.cpu_count()
IS_M_SERIES = platform.system() == 'Darwin' and platform.machine() == 'arm64'

if IS_M_SERIES:
    print(f"✓ M-Series chip detected ({N_CORES} cores) - Using optimized algorithms")
else:
    print(f"Running on standard architecture ({N_CORES} cores)")


def load_clustering_data():
    """Load clustering features with progress bar."""
    print("\nLoading clustering features...")
    input_path = os.path.join(DATA_PATH, 'features_for_clustering.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Clustering features not found at {input_path}. "
            "Please run feature engineering first: python src/features/feature_engineering.py"
        )

    # Optimized loading for M-series
    if IS_M_SERIES:
        file_size = os.path.getsize(input_path)
        chunk_size = 100000
        print(f"Reading data ({file_size / (1024**2):.1f} MB)...")

        chunks = []
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading CSV") as pbar:
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                pbar.update(chunk_size * 100)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded {len(df):,} samples with {len(df.columns)} features")
    return df


def prepare_features(df):
    """
    Prepare and scale features for clustering.
    Theme 5: Aggregate by zones for meaningful clustering.
    """
    print("\nAggregating data by zone (square_id)...")

    # Zone-level aggregation (Theme 5 requirement: "grouping zones")
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

        print(f"Aggregated to {len(zone_agg)} zones (from {len(df):,} measurements)")
        df = zone_agg

    # Feature selection
    features_to_use = [
        'latitude', 'longitude',
        'avg_latency', 'std_latency', 'total_throughput',
        'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    features_to_use = [f for f in features_to_use if f in df.columns]

    print(f"Using features: {features_to_use}")

    # Extract and clean
    X = df[features_to_use].copy()
    X_clean = X.dropna()

    print(f"Features shape: {X_clean.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    return X_scaled, features_to_use, scaler, X_clean.index, df


def elbow_method(X, max_k=10):
    """Elbow method to find optimal k."""
    print(f"\nElbow method analysis (k=2 to k={max_k})...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    # M-series optimization: lloyd algorithm performs better
    algorithm = 'lloyd' if IS_M_SERIES else 'elkan'

    for k in tqdm(K_range, desc="Elbow analysis"):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            algorithm=algorithm,
            max_iter=300
        )
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        labels = kmeans.labels_
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        tqdm.write(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil_score:.3f}")

    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n✓ Optimal k (by silhouette score): {optimal_k}")

    return inertias, silhouette_scores, optimal_k


def plot_elbow_analysis(inertias, silhouette_scores, K_range, optimal_k):
    """Plot elbow method results."""
    print("Generating elbow analysis plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method - Inertia', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Elbow Method - Silhouette Score', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_PATH, 'elbow_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def run_kmeans(X, n_clusters, feature_names):
    """Run K-Means clustering."""
    print(f"\nRunning K-Means (k={n_clusters})...")

    algorithm = 'lloyd' if IS_M_SERIES else 'elkan'

    with tqdm(total=1, desc="K-Means clustering") as pbar:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            algorithm=algorithm,
            max_iter=300
        )
        labels = kmeans.fit_predict(X)
        pbar.update(1)

    # Metrics
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    print(f"\nK-Means Results:")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette Score: {sil_score:.3f}")
    print(f"  Davies-Bouldin Index: {db_score:.3f} (lower is better)")
    print(f"  Cluster distribution: {np.bincount(labels)}")

    return kmeans, labels


def run_dbscan(X, eps=0.5, min_samples=5):
    """Run DBSCAN clustering."""
    print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")

    # M-series can use parallel jobs
    n_jobs = N_CORES if IS_M_SERIES else 1

    with tqdm(total=1, desc="DBSCAN clustering") as pbar:
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=n_jobs,
            algorithm='auto'
        )
        labels = dbscan.fit_predict(X)
        pbar.update(1)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nDBSCAN Results:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Outliers: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            sil_score = silhouette_score(X[mask], labels[mask])
            print(f"  Silhouette Score (excl. outliers): {sil_score:.3f}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster distribution: {dict(zip(unique_labels, counts))}")

    return dbscan, labels


def plot_clusters_2d(X, labels, title, feature_names, output_name):
    """Plot clusters using first 2 features (lat/lon)."""
    print(f"Generating plot: {output_name}")
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')

    plt.xlabel(feature_names[0] if len(feature_names) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(feature_names[1] if len(feature_names) > 1 else 'Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(OUTPUT_PATH, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def analyze_clusters(df, labels, cluster_type='kmeans'):
    """Analyze cluster characteristics."""
    df_analysis = df.copy()
    df_analysis['cluster'] = labels

    print(f"\n{cluster_type.upper()} Cluster Analysis:")
    print("=" * 60)

    for cluster_id in sorted(df_analysis['cluster'].unique()):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} (n={len(cluster_data):,}):")

        if 'avg_latency' in df_analysis.columns:
            print(f"  Avg Latency: {cluster_data['avg_latency'].mean():.2f} ms")
        if 'total_throughput' in df_analysis.columns:
            print(f"  Avg Throughput: {cluster_data['total_throughput'].mean():.2f} Mbps")
        if 'latitude' in df_analysis.columns and 'longitude' in df_analysis.columns:
            print(f"  GPS Center: ({cluster_data['latitude'].mean():.4f}, {cluster_data['longitude'].mean():.4f})")

    # Save results
    output_file = os.path.join(OUTPUT_PATH, f'{cluster_type}_clusters.csv')
    df_analysis.to_csv(output_file, index=False)
    print(f"\n✓ Saved cluster assignments: {output_file}")


def main():
    """Main execution flow."""
    print("=" * 70)
    print("5G Network Performance Clustering Analysis")
    if IS_M_SERIES:
        print(f"M-Series Optimized Mode ({N_CORES} cores)")
    print("=" * 70)

    # Load data
    df = load_clustering_data()

    # Prepare features (zone-level aggregation)
    X_scaled, feature_names, scaler, valid_indices, df_zones = prepare_features(df)
    df_valid = df_zones.loc[valid_indices].copy()

    # 1. Elbow Method
    inertias, silhouette_scores, optimal_k = elbow_method(X_scaled, max_k=10)
    plot_elbow_analysis(inertias, silhouette_scores, range(2, 11), optimal_k)

    # 2. K-Means
    kmeans_model, kmeans_labels = run_kmeans(X_scaled, n_clusters=optimal_k, feature_names=feature_names)
    plot_clusters_2d(X_scaled, kmeans_labels, f'K-Means Clustering (k={optimal_k})',
                     feature_names, 'kmeans_clusters.png')
    analyze_clusters(df_valid, kmeans_labels, cluster_type='kmeans')

    # 3. DBSCAN (optimized parameters for zone-level data)
    dbscan_model, dbscan_labels = run_dbscan(X_scaled, eps=1.5, min_samples=5)
    plot_clusters_2d(X_scaled, dbscan_labels, 'DBSCAN Clustering',
                     feature_names, 'dbscan_clusters.png')
    analyze_clusters(df_valid, dbscan_labels, cluster_type='dbscan')

    print("\n" + "=" * 70)
    print("✓ Clustering Analysis Complete!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
