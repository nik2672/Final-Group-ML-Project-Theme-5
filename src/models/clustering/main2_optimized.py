"""
OPTIMIZED 5G Network Performance Zone Analysis with Parameter Grid Search

This enhanced version implements comprehensive parameter optimization for both:
1. Birch: Memory-efficient hierarchical clustering 
2. OPTICS: Density-based clustering with automatic cluster discovery

Key improvements over main2.py:
- Systematic parameter grid search (like HDBSCAN)
- Multiple evaluation metrics (silhouette, davies-bouldin)
- Saves all trial results for analysis
- PCA-based visualization for better high-dimensional representation
- Multi-criteria model selection
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import Birch, OPTICS
from sklearn.neighbors import NearestNeighbors
import multiprocessing
from tqdm import tqdm

# Setup paths
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'clustering_optimized')
os.makedirs(OUTPUT_PATH, exist_ok=True)

N_CORES = multiprocessing.cpu_count()
IS_M_SERIES = platform.system() == 'Darwin' and platform.machine() == 'arm64'

print(f"{'M-Series' if IS_M_SERIES else 'Standard'} architecture detected ({N_CORES} cores)")


def load_clustering_data():
    """Load train/test clustering data."""
    print("\n=== Loading 5G network data ===")
    
    train_path = os.path.join(DATA_PATH, 'features_for_clustering_train_improved.csv')
    test_path = os.path.join(DATA_PATH, 'features_for_clustering_test_improved.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Data files not found. Run: python src/features/leakage_safe_feature_engineering.py"
        )

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    
    print(f"Train: {len(train_df):,} records | Test: {len(test_df):,} records")
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Prepare features with zone aggregation and scaling."""
    print("\n=== Preparing features (leakage-safe) ===")
    
    def aggregate_zones(df, label):
        if 'square_id' not in df.columns:
            return df.copy()
        
        agg_dict = {
            'latitude': 'mean',
            'longitude': 'mean', 
            'avg_latency': 'mean',
            'std_latency': 'mean',
            'total_throughput': 'mean',
            'zone_avg_latency': 'first',
            'zone_avg_upload': 'first',
            'zone_avg_download': 'first'
        }
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        zone_df = df.groupby('square_id').agg(agg_dict).reset_index()
        print(f"{label}: {len(df):,} → {len(zone_df):,} zones")
        return zone_df
    
    train_processed = aggregate_zones(train_df, "Train")
    test_processed = aggregate_zones(test_df, "Test")
    
    # Select features
    feature_cols = [
        'latitude', 'longitude', 'avg_latency', 'std_latency', 'total_throughput',
        'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    feature_cols = [col for col in feature_cols if col in train_processed.columns]
    
    if len(feature_cols) < 3:
        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != 'square_id'][:8]
    
    print(f"Features: {feature_cols}")
    
    X_train = train_processed[feature_cols].fillna(0).values
    X_test = test_processed[feature_cols].fillna(0).values
    
    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaled train: {X_train_scaled.shape} | Scaled test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, feature_cols, scaler, train_processed, test_processed


def safe_silhouette(X, labels):
    """Compute silhouette score excluding outliers."""
    mask = labels != -1
    if mask.sum() <= 1 or len(np.unique(labels[mask])) < 2:
        return None
    try:
        return silhouette_score(X[mask], labels[mask])
    except:
        return None


def count_clusters(labels):
    """Count clusters and outliers."""
    n_outliers = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, n_outliers


def pca_2d(X):
    """Simple PCA for 2D visualization."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ vt[:2].T


def run_birch_optimized(X):
    """
    Run Birch clustering with comprehensive parameter grid search.
    
    Tests combinations of:
    - n_clusters: Final number of clusters
    - threshold: Radius of subcluster
    - branching_factor: Maximum CF subclusters in each node
    """
    print("\n=== BIRCH CLUSTERING WITH PARAMETER OPTIMIZATION ===")
    
    param_grid = {
        'n_clusters': [2, 3, 4, 5, 6, 8],
        'threshold': [0.2, 0.3, 0.5, 0.7],
        'branching_factor': [30, 50, 70]
    }
    
    trials = []
    best = {
        "score": -1e9,
        "labels": None,
        "model": None,
        "params": None,
        "n_clusters": 0
    }
    
    total_trials = len(param_grid['n_clusters']) * len(param_grid['threshold']) * len(param_grid['branching_factor'])
    print(f"Testing {total_trials} parameter combinations...")
    
    with tqdm(total=total_trials, desc="Birch parameter sweep") as pbar:
        for n_clust in param_grid['n_clusters']:
            for thresh in param_grid['threshold']:
                for branch in param_grid['branching_factor']:
                    try:
                        model = Birch(
                            n_clusters=n_clust,
                            threshold=thresh,
                            branching_factor=branch
                        )
                        labels = model.fit_predict(X)
                        
                        # Evaluate with multiple metrics
                        sil = silhouette_score(X, labels)
                        db = davies_bouldin_score(X, labels)
                        ch = calinski_harabasz_score(X, labels)
                        
                        # Combined score (higher is better)
                        # Normalize CH to 0-1 range approximately
                        ch_normalized = min(ch / 1000.0, 1.0)
                        combined_score = sil + ch_normalized - (db / 10.0)
                        
                        trials.append({
                            'n_clusters': n_clust,
                            'threshold': thresh,
                            'branching_factor': branch,
                            'silhouette': sil,
                            'davies_bouldin': db,
                            'calinski_harabasz': ch,
                            'combined_score': combined_score
                        })
                        
                        # Update best model
                        if combined_score > best["score"]:
                            best.update({
                                "score": combined_score,
                                "labels": labels,
                                "model": model,
                                "params": {
                                    'n_clusters': n_clust,
                                    'threshold': thresh,
                                    'branching_factor': branch
                                },
                                "n_clusters": n_clust
                            })
                    
                    except Exception as e:
                        trials.append({
                            'n_clusters': n_clust,
                            'threshold': thresh,
                            'branching_factor': branch,
                            'silhouette': np.nan,
                            'davies_bouldin': np.nan,
                            'calinski_harabasz': np.nan,
                            'combined_score': -1e9
                        })
                    
                    pbar.update(1)
    
    # Save trials
    trials_df = pd.DataFrame(trials).sort_values('combined_score', ascending=False)
    trials_path = os.path.join(OUTPUT_PATH, 'birch_trials.csv')
    trials_df.to_csv(trials_path, index=False)
    print(f"\nSaved trials to: {trials_path}")
    
    print(f"\nBest Birch Configuration:")
    print(f"  Parameters: {best['params']}")
    print(f"  Clusters: {best['n_clusters']}")
    print(f"  Silhouette: {trials_df.iloc[0]['silhouette']:.3f}")
    print(f"  Davies-Bouldin: {trials_df.iloc[0]['davies_bouldin']:.3f}")
    print(f"  Calinski-Harabasz: {trials_df.iloc[0]['calinski_harabasz']:.1f}")
    
    return best["labels"], best["model"], best["n_clusters"], trials_df


def run_optics_optimized(X):
    """
    Run OPTICS clustering with comprehensive parameter grid search.
    
    Tests combinations of:
    - min_samples: Minimum neighbors for core point
    - max_eps: Maximum neighborhood distance
    - xi: Steepness threshold for cluster detection
    - cluster_method: Algorithm for extracting clusters
    """
    print("\n=== OPTICS CLUSTERING WITH PARAMETER OPTIMIZATION ===")
    
    param_grid = {
        'min_samples': [3, 5, 8, 10, 15],
        'max_eps': [1.0, 1.5, 2.0, 3.0, np.inf],
        'xi': [0.05, 0.1, 0.15, 0.2],
        'cluster_method': ['xi', 'dbscan']
    }
    
    trials = []
    best = {
        "score": -1e9,
        "labels": None,
        "model": None,
        "params": None,
        "n_clusters": 0,
        "n_outliers": 0
    }
    
    total_trials = (len(param_grid['min_samples']) * len(param_grid['max_eps']) * 
                    len(param_grid['xi']) * len(param_grid['cluster_method']))
    print(f"Testing {total_trials} parameter combinations...")
    
    with tqdm(total=total_trials, desc="OPTICS parameter sweep") as pbar:
        for min_s in param_grid['min_samples']:
            for max_e in param_grid['max_eps']:
                for xi_val in param_grid['xi']:
                    for method in param_grid['cluster_method']:
                        try:
                            model = OPTICS(
                                min_samples=min_s,
                                max_eps=max_e,
                                cluster_method=method,
                                xi=xi_val if method == 'xi' else 0.05,
                                n_jobs=-1
                            )
                            labels = model.fit_predict(X)
                            
                            n_clusters, n_outliers = count_clusters(labels)
                            sil = safe_silhouette(X, labels)
                            
                            # Calinski-Harabasz (only for non-outlier points)
                            ch = None
                            if n_clusters > 1 and -1 in labels:
                                mask = labels != -1
                                if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
                                    try:
                                        ch = calinski_harabasz_score(X[mask], labels[mask])
                                    except:
                                        ch = None
                            elif n_clusters > 1:
                                try:
                                    ch = calinski_harabasz_score(X, labels)
                                except:
                                    ch = None
                            
                            # Multi-criteria score
                            if sil is not None and n_clusters > 0:
                                # Prefer: high silhouette, fewer outliers, reasonable cluster count
                                outlier_penalty = n_outliers / len(X)
                                ch_bonus = (ch / 1000.0) if ch else 0
                                combined_score = sil + ch_bonus - (outlier_penalty * 0.5)
                            else:
                                combined_score = -1.0
                            
                            trials.append({
                                'min_samples': min_s,
                                'max_eps': max_e if max_e != np.inf else 'inf',
                                'xi': xi_val,
                                'cluster_method': method,
                                'n_clusters': n_clusters,
                                'n_outliers': n_outliers,
                                'outlier_pct': f"{outlier_penalty*100:.1f}%",
                                'silhouette': sil if sil is not None else np.nan,
                                'calinski_harabasz': ch if ch is not None else np.nan,
                                'combined_score': combined_score
                            })
                            
                            # Update best (prefer high silhouette, then low outliers, then more clusters)
                            is_better = (combined_score > best["score"] or
                                       (np.isclose(combined_score, best["score"]) and 
                                        (n_outliers < best["n_outliers"] or
                                         (n_outliers == best["n_outliers"] and n_clusters > best["n_clusters"]))))
                            
                            if is_better and n_clusters > 0:
                                best.update({
                                    "score": combined_score,
                                    "labels": labels,
                                    "model": model,
                                    "params": {
                                        'min_samples': min_s,
                                        'max_eps': max_e,
                                        'xi': xi_val,
                                        'cluster_method': method
                                    },
                                    "n_clusters": n_clusters,
                                    "n_outliers": n_outliers
                                })
                        
                        except Exception as e:
                            trials.append({
                                'min_samples': min_s,
                                'max_eps': max_e if max_e != np.inf else 'inf',
                                'xi': xi_val,
                                'cluster_method': method,
                                'n_clusters': 0,
                                'n_outliers': 0,
                                'outlier_pct': 'N/A',
                                'silhouette': np.nan,
                                'calinski_harabasz': np.nan,
                                'combined_score': -1e9
                            })
                        
                        pbar.update(1)
    
    # Save trials
    trials_df = pd.DataFrame(trials).sort_values(
        ['combined_score', 'n_outliers', 'n_clusters'],
        ascending=[False, True, False]
    )
    trials_path = os.path.join(OUTPUT_PATH, 'optics_trials.csv')
    trials_df.to_csv(trials_path, index=False)
    print(f"\nSaved trials to: {trials_path}")
    
    print(f"\nBest OPTICS Configuration:")
    print(f"  Parameters: {best['params']}")
    print(f"  Clusters: {best['n_clusters']}")
    print(f"  Outliers: {best['n_outliers']} ({best['n_outliers']/len(X)*100:.1f}%)")
    if trials_df.iloc[0]['silhouette'] == trials_df.iloc[0]['silhouette']:  # Check not NaN
        print(f"  Silhouette (excl. noise): {trials_df.iloc[0]['silhouette']:.3f}")
    if trials_df.iloc[0]['calinski_harabasz'] == trials_df.iloc[0]['calinski_harabasz']:  # Check not NaN
        print(f"  Calinski-Harabasz (excl. noise): {trials_df.iloc[0]['calinski_harabasz']:.1f}")
    
    return best["labels"], best["model"], best["n_clusters"], trials_df


def plot_results_pca(X, labels, filename, title, method_name):
    """Create PCA-based 2D visualization."""
    print(f"Creating {method_name} visualization with PCA...")
    
    X_2d = pca_2d(X)
    
    plt.figure(figsize=(10, 8))
    
    # Handle outliers
    noise_mask = labels == -1
    cluster_mask = ~noise_mask
    
    if noise_mask.any():
        plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                   c='lightgray', s=40, alpha=0.6, label='Outliers')
    
    if cluster_mask.any():
        scatter = plt.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
                            c=labels[cluster_mask], cmap='Set3', s=50, alpha=0.8)
        plt.colorbar(scatter, label='Cluster ID')
    
    n_clusters, n_outliers = count_clusters(labels)
    plt.title(f"{title}\nClusters: {n_clusters} | Outliers: {n_outliers}", 
             fontsize=14, fontweight='bold')
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.grid(alpha=0.3)
    
    if noise_mask.any():
        plt.legend()
    
    output_path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def assign_test_clusters_knn(X_train, train_labels, X_test, n_neighbors=5):
    """Assign test data using KNN (leakage-safe)."""
    print(f"\nAssigning test data using {n_neighbors}-NN...")
    
    # Handle outliers for OPTICS
    if -1 in train_labels:
        mask = train_labels != -1
        if mask.sum() == 0:
            return np.full(len(X_test), -1, dtype=int)
        
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, mask.sum())).fit(X_train[mask])
        indices = nn.kneighbors(X_test, return_distance=False)
        
        test_labels = []
        train_labels_clean = train_labels[mask]
        for idx in indices:
            neighbor_labels = train_labels_clean[idx]
            valid = neighbor_labels[neighbor_labels != -1]
            test_labels.append(np.bincount(valid).argmax() if len(valid) > 0 else -1)
        
        return np.array(test_labels, dtype=int)
    
    else:
        # Standard case (Birch)
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
        indices = nn.kneighbors(X_test, return_distance=False)
        
        test_labels = []
        for idx in indices:
            test_labels.append(np.bincount(train_labels[idx]).argmax())
        
        return np.array(test_labels, dtype=int)


def save_results(df, labels, method_name):
    """Save clustering results to CSV."""
    results_df = df.copy()
    results_df[f'{method_name}_cluster'] = labels
    
    output_file = os.path.join(OUTPUT_PATH, f'{method_name}_clusters.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Saved {method_name} results: {output_file}")
    
    # Summary
    print(f"\n{method_name.upper()} Cluster Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        if cluster_id == -1:
            print(f"  Outliers: {count:,}")
        else:
            print(f"  Cluster {cluster_id}: {count:,}")


def main():
    """Execute optimized clustering pipeline."""
    print("=" * 70)
    print("OPTIMIZED 5G NETWORK CLUSTERING WITH PARAMETER GRID SEARCH")
    print("=" * 70)
    
    try:
        # Load data
        train_df, test_df = load_clustering_data()
        
        # Prepare features
        X_train, X_test, features, scaler, train_proc, test_proc = prepare_features(train_df, test_df)
        
        # Run Birch with optimization
        birch_labels, birch_model, n_birch, birch_trials = run_birch_optimized(X_train)
        birch_test = assign_test_clusters_knn(X_train, birch_labels, X_test)
        
        # Run OPTICS with optimization
        optics_labels, optics_model, n_optics, optics_trials = run_optics_optimized(X_train)
        optics_test = assign_test_clusters_knn(X_train, optics_labels, X_test)
        
        # Visualizations with PCA (TRAIN)
        print("\n=== Creating TRAIN visualizations ===")
        plot_results_pca(X_train, birch_labels, 'birch_clusters_train_pca.png',
                        'Birch Clustering - TRAIN (PCA)', 'Birch Train')
        plot_results_pca(X_train, optics_labels, 'optics_clusters_train_pca.png',
                        'OPTICS Clustering - TRAIN (PCA)', 'OPTICS Train')
        
        # Visualizations with PCA (TEST)
        print("\n=== Creating TEST visualizations ===")
        plot_results_pca(X_test, birch_test, 'birch_clusters_test_pca.png',
                        'Birch Clustering - TEST (PCA)', 'Birch Test')
        plot_results_pca(X_test, optics_test, 'optics_clusters_test_pca.png',
                        'OPTICS Clustering - TEST (PCA)', 'OPTICS Test')
        
        # Save results
        print("\n=== Saving results ===")
        save_results(train_proc, birch_labels, 'birch_train')
        save_results(test_proc, birch_test, 'birch_test')
        save_results(train_proc, optics_labels, 'optics_train')
        save_results(test_proc, optics_test, 'optics_test')
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print(f"Birch: {n_birch} clusters")
        print(f"OPTICS: {n_optics} clusters")
        print(f"\nResults saved to: {OUTPUT_PATH}")
        print("=" * 70)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
