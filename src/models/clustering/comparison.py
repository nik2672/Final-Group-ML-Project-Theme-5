"""
Standalone Model Comparison - Train & Evaluate 5 Clustering Models

This script independently trains all 5 clustering models with predefined optimal parameters
and evaluates them on both train and test data WITHOUT requiring tuning.py results.

Models:
1. KMeans - Partition-based clustering
2. DBSCAN - Density-based spatial clustering
3. Birch - Hierarchical clustering with CF tree
4. OPTICS - Density-based with reachability ordering
5. HDBSCAN - Hierarchical density-based clustering

Steps:
1. Load and preprocess train/test data
2. Train each model with best-known parameters
3. Evaluate using 3 metrics (Silhouette, DBI, CH)
4. Calculate combined scores and rankings
5. Generate visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("WARNING: HDBSCAN not available. Install with: conda install -c conda-forge hdbscan")

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "clustering", "standalone_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STANDALONE MODEL COMPARISON - 5 CLUSTERING ALGORITHMS")
print("="*80)


def load_data():
    """Load and preprocess train/test data."""
    print("\n=== Loading Data ===")
    train_path = os.path.join(DATA_PATH, "features_for_clustering_train_improved.csv")
    test_path = os.path.join(DATA_PATH, "features_for_clustering_test_improved.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found in {DATA_PATH}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded TRAIN: {train_df.shape}, TEST: {test_df.shape}")
    
    # Aggregate by square_id for zone-level analysis
    if "square_id" in train_df.columns:
        print("Aggregating by square_id for zone-level features...")
        train_df = train_df.groupby("square_id").agg({
            "latitude": "mean",
            "longitude": "mean",
            "avg_latency": "mean",
            "std_latency": "mean",
            "total_throughput": "mean",
            "zone_avg_latency": "first",
            "zone_avg_upload": "first",
            "zone_avg_download": "first"
        }).reset_index()
        
        test_df = test_df.groupby("square_id").agg({
            "latitude": "mean",
            "longitude": "mean",
            "avg_latency": "mean",
            "std_latency": "mean",
            "total_throughput": "mean",
            "zone_avg_latency": "first",
            "zone_avg_upload": "first",
            "zone_avg_download": "first"
        }).reset_index()
        print(f"After aggregation - TRAIN: {train_df.shape}, TEST: {test_df.shape}")
    
    # Select feature columns
    feature_cols = ["latitude", "longitude", "avg_latency", "std_latency", 
                   "total_throughput", "zone_avg_latency", "zone_avg_upload", "zone_avg_download"]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Scale features
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaled - TRAIN: {X_train_scaled.shape}, TEST: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, train_df, test_df, feature_cols


def assign_test_to_clusters(X_train, labels_train, X_test):
    """Assign test data to clusters using KNN (for density-based models)."""
    mask = labels_train != -1
    X_train_valid = X_train[mask]
    labels_train_valid = labels_train[mask]
    
    if len(X_train_valid) == 0:
        return np.full(len(X_test), -1)
    
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X_train_valid)
    distances, indices = knn.kneighbors(X_test)
    labels_test = labels_train_valid[indices.flatten()]
    
    return labels_test


def safe_metrics(X, labels):
    """Calculate metrics, handling outliers."""
    mask = labels != -1
    n_valid = mask.sum()
    n_outliers = (~mask).sum()
    n_clusters = len(set(labels[mask])) if n_valid > 0 else 0
    
    metrics = {
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_pct': (n_outliers / len(labels)) * 100,
        'silhouette': np.nan,
        'davies_bouldin': np.nan,
        'calinski_harabasz': np.nan
    }
    
    if n_valid > 1 and n_clusters > 1:
        try:
            metrics['silhouette'] = silhouette_score(X[mask], labels[mask])
        except:
            pass
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X[mask], labels[mask])
        except:
            pass
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X[mask], labels[mask])
        except:
            pass
    
    return metrics


def train_all_models(X_train, X_test):
    """Train all 5 models with optimal parameters."""
    print("\n=== Training All Models ===")
    
    results = []
    models_data = {}
    
    # 1. KMeans
    print("\n[1/5] Training KMeans...")
    params_kmeans = {'n_clusters': 2}
    model = KMeans(n_clusters=params_kmeans['n_clusters'], random_state=42)
    labels_train = model.fit_predict(X_train)
    labels_test = model.predict(X_test)
    
    metrics_train = safe_metrics(X_train, labels_train)
    metrics_test = safe_metrics(X_test, labels_test)
    
    results.append({
        'Model': 'KMeans',
        'Params': f"k={params_kmeans['n_clusters']}",
        'Train_Silhouette': metrics_train['silhouette'],
        'Test_Silhouette': metrics_test['silhouette'],
        'Train_DBI': metrics_train['davies_bouldin'],
        'Test_DBI': metrics_test['davies_bouldin'],
        'Train_CH': metrics_train['calinski_harabasz'],
        'Test_CH': metrics_test['calinski_harabasz'],
        'Train_Clusters': metrics_train['n_clusters'],
        'Test_Clusters': metrics_test['n_clusters'],
        'Train_Outliers': metrics_train['n_outliers'],
        'Test_Outliers': metrics_test['n_outliers']
    })
    
    models_data['KMeans'] = {'train_labels': labels_train, 'test_labels': labels_test}
    print(f"  Train: Sil={metrics_train['silhouette']:.3f}, DBI={metrics_train['davies_bouldin']:.3f}, CH={metrics_train['calinski_harabasz']:.1f}")
    print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, DBI={metrics_test['davies_bouldin']:.3f}, CH={metrics_test['calinski_harabasz']:.1f}")
    
    # 2. DBSCAN
    print("\n[2/5] Training DBSCAN...")
    params_dbscan = {'eps': 0.5, 'min_samples': 3}
    model = DBSCAN(eps=params_dbscan['eps'], min_samples=params_dbscan['min_samples'])
    labels_train = model.fit_predict(X_train)
    labels_test = assign_test_to_clusters(X_train, labels_train, X_test)
    
    metrics_train = safe_metrics(X_train, labels_train)
    metrics_test = safe_metrics(X_test, labels_test)
    
    results.append({
        'Model': 'DBSCAN',
        'Params': f"eps={params_dbscan['eps']},min={params_dbscan['min_samples']}",
        'Train_Silhouette': metrics_train['silhouette'],
        'Test_Silhouette': metrics_test['silhouette'],
        'Train_DBI': metrics_train['davies_bouldin'],
        'Test_DBI': metrics_test['davies_bouldin'],
        'Train_CH': metrics_train['calinski_harabasz'],
        'Test_CH': metrics_test['calinski_harabasz'],
        'Train_Clusters': metrics_train['n_clusters'],
        'Test_Clusters': metrics_test['n_clusters'],
        'Train_Outliers': metrics_train['n_outliers'],
        'Test_Outliers': metrics_test['n_outliers']
    })
    
    models_data['DBSCAN'] = {'train_labels': labels_train, 'test_labels': labels_test}
    print(f"  Train: Sil={metrics_train['silhouette']:.3f}, Clusters={metrics_train['n_clusters']}, Outliers={metrics_train['n_outliers']}")
    print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, Clusters={metrics_test['n_clusters']}, Outliers={metrics_test['n_outliers']}")
    
    # 3. Birch
    print("\n[3/5] Training Birch...")
    params_birch = {'n_clusters': 2, 'threshold': 0.2, 'branching_factor': 70}
    model = Birch(n_clusters=params_birch['n_clusters'], 
                 threshold=params_birch['threshold'],
                 branching_factor=params_birch['branching_factor'])
    labels_train = model.fit_predict(X_train)
    labels_test = model.predict(X_test)
    
    metrics_train = safe_metrics(X_train, labels_train)
    metrics_test = safe_metrics(X_test, labels_test)
    
    results.append({
        'Model': 'Birch',
        'Params': f"k={params_birch['n_clusters']},thr={params_birch['threshold']},bf={params_birch['branching_factor']}",
        'Train_Silhouette': metrics_train['silhouette'],
        'Test_Silhouette': metrics_test['silhouette'],
        'Train_DBI': metrics_train['davies_bouldin'],
        'Test_DBI': metrics_test['davies_bouldin'],
        'Train_CH': metrics_train['calinski_harabasz'],
        'Test_CH': metrics_test['calinski_harabasz'],
        'Train_Clusters': metrics_train['n_clusters'],
        'Test_Clusters': metrics_test['n_clusters'],
        'Train_Outliers': metrics_train['n_outliers'],
        'Test_Outliers': metrics_test['n_outliers']
    })
    
    models_data['Birch'] = {'train_labels': labels_train, 'test_labels': labels_test}
    print(f"  Train: Sil={metrics_train['silhouette']:.3f}, DBI={metrics_train['davies_bouldin']:.3f}, CH={metrics_train['calinski_harabasz']:.1f}")
    print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, DBI={metrics_test['davies_bouldin']:.3f}, CH={metrics_test['calinski_harabasz']:.1f}")
    
    # 4. OPTICS
    print("\n[4/5] Training OPTICS...")
    params_optics = {'min_samples': 5, 'max_eps': np.inf, 'xi': 0.2, 'cluster_method': 'xi'}
    model = OPTICS(min_samples=params_optics['min_samples'],
                  max_eps=params_optics['max_eps'],
                  xi=params_optics['xi'],
                  cluster_method=params_optics['cluster_method'])
    labels_train = model.fit_predict(X_train)
    labels_test = assign_test_to_clusters(X_train, labels_train, X_test)
    
    metrics_train = safe_metrics(X_train, labels_train)
    metrics_test = safe_metrics(X_test, labels_test)
    
    results.append({
        'Model': 'OPTICS',
        'Params': f"min={params_optics['min_samples']},eps=inf,xi={params_optics['xi']},method={params_optics['cluster_method']}",
        'Train_Silhouette': metrics_train['silhouette'],
        'Test_Silhouette': metrics_test['silhouette'],
        'Train_DBI': metrics_train['davies_bouldin'],
        'Test_DBI': metrics_test['davies_bouldin'],
        'Train_CH': metrics_train['calinski_harabasz'],
        'Test_CH': metrics_test['calinski_harabasz'],
        'Train_Clusters': metrics_train['n_clusters'],
        'Test_Clusters': metrics_test['n_clusters'],
        'Train_Outliers': metrics_train['n_outliers'],
        'Test_Outliers': metrics_test['n_outliers']
    })
    
    models_data['OPTICS'] = {'train_labels': labels_train, 'test_labels': labels_test}
    print(f"  Train: Sil={metrics_train['silhouette']:.3f}, Clusters={metrics_train['n_clusters']}, Outliers={metrics_train['n_outliers']}")
    print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, Clusters={metrics_test['n_clusters']}, Outliers={metrics_test['n_outliers']}")
    
    # 5. HDBSCAN
    if HDBSCAN_AVAILABLE:
        print("\n[5/5] Training HDBSCAN...")
        params_hdbscan = {'min_cluster_size': 5, 'min_samples': 3}
        model = hdbscan.HDBSCAN(min_cluster_size=params_hdbscan['min_cluster_size'],
                               min_samples=params_hdbscan['min_samples'],
                               cluster_selection_method='eom')
        labels_train = model.fit_predict(X_train)
        labels_test = assign_test_to_clusters(X_train, labels_train, X_test)
        
        metrics_train = safe_metrics(X_train, labels_train)
        metrics_test = safe_metrics(X_test, labels_test)
        
        results.append({
            'Model': 'HDBSCAN',
            'Params': f"mcs={params_hdbscan['min_cluster_size']},ms={params_hdbscan['min_samples']}",
            'Train_Silhouette': metrics_train['silhouette'],
            'Test_Silhouette': metrics_test['silhouette'],
            'Train_DBI': metrics_train['davies_bouldin'],
            'Test_DBI': metrics_test['davies_bouldin'],
            'Train_CH': metrics_train['calinski_harabasz'],
            'Test_CH': metrics_test['calinski_harabasz'],
            'Train_Clusters': metrics_train['n_clusters'],
            'Test_Clusters': metrics_test['n_clusters'],
            'Train_Outliers': metrics_train['n_outliers'],
            'Test_Outliers': metrics_test['n_outliers']
        })
        
        models_data['HDBSCAN'] = {'train_labels': labels_train, 'test_labels': labels_test}
        print(f"  Train: Sil={metrics_train['silhouette']:.3f}, Clusters={metrics_train['n_clusters']}, Outliers={metrics_train['n_outliers']}")
        print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, Clusters={metrics_test['n_clusters']}, Outliers={metrics_test['n_outliers']}")
    else:
        print("\n[5/5] HDBSCAN skipped - not installed")
    
    return pd.DataFrame(results), models_data


def calculate_rankings(results_df):
    """Calculate normalized combined scores and rankings."""
    print("\n=== Calculating Rankings ===")
    
    # Normalize metrics to 0-1 scale
    results_ranked = results_df.copy()
    
    # Silhouette: higher is better
    sil_min = results_df['Test_Silhouette'].min()
    sil_max = results_df['Test_Silhouette'].max()
    results_ranked['Sil_Normalized'] = (results_df['Test_Silhouette'] - sil_min) / (sil_max - sil_min) if sil_max > sil_min else 0
    
    # DBI: lower is better, so invert
    dbi_min = results_df['Test_DBI'].min()
    dbi_max = results_df['Test_DBI'].max()
    results_ranked['DBI_Normalized'] = 1 - ((results_df['Test_DBI'] - dbi_min) / (dbi_max - dbi_min)) if dbi_max > dbi_min else 0
    
    # CH: higher is better
    ch_min = results_df['Test_CH'].min()
    ch_max = results_df['Test_CH'].max()
    results_ranked['CH_Normalized'] = (results_df['Test_CH'] - ch_min) / (ch_max - ch_min) if ch_max > ch_min else 0
    
    # Combined score: weighted average
    results_ranked['Combined_Score'] = (
        results_ranked['Sil_Normalized'] * 0.40 +
        results_ranked['DBI_Normalized'] * 0.30 +
        results_ranked['CH_Normalized'] * 0.30
    )
    
    # Sort by combined score
    results_ranked = results_ranked.sort_values('Combined_Score', ascending=False)
    
    print("\nRanking Methodology:")
    print("  - Silhouette Score: 40% weight (higher better)")
    print("  - Davies-Bouldin Index: 30% weight (lower better, inverted)")
    print("  - Calinski-Harabasz: 30% weight (higher better)")
    print("  - All metrics normalized to 0-1 scale")
    
    return results_ranked


def visualize_comparison(X_train, X_test, models_data, results_df):
    """Create comparison visualizations."""
    print("\n=== Generating Visualizations ===")
    
    # PCA for 2D projection
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    
    n_models = len(models_data)
    
    # Cluster visualizations
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        # Train
        ax_train = axes[0, idx]
        labels_train = data['train_labels']
        scatter = ax_train.scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                                  c=labels_train, cmap='tab10', alpha=0.6, s=30)
        ax_train.set_title(f"{model_name} - TRAIN\n{len(set(labels_train))-1 if -1 in labels_train else len(set(labels_train))} clusters", 
                          fontsize=12)
        ax_train.set_xlabel("PC1")
        ax_train.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax_train)
        
        # Test
        ax_test = axes[1, idx]
        labels_test = data['test_labels']
        scatter = ax_test.scatter(X_test_2d[:, 0], X_test_2d[:, 1], 
                                c=labels_test, cmap='tab10', alpha=0.6, s=30)
        ax_test.set_title(f"{model_name} - TEST\n{len(set(labels_test))-1 if -1 in labels_test else len(set(labels_test))} clusters", 
                         fontsize=12)
        ax_test.set_xlabel("PC1")
        ax_test.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax_test)
    
    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, "all_models_comparison.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {vis_path}")
    plt.close()
    
    # Metrics bar charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    # Silhouette
    ax = axes[0, 0]
    ax.bar(x - width/2, results_df['Train_Silhouette'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_Silhouette'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # DBI
    ax = axes[0, 1]
    ax.bar(x - width/2, results_df['Train_DBI'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_DBI'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # CH
    ax = axes[1, 0]
    ax.bar(x - width/2, results_df['Train_CH'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_CH'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Score (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Cluster count
    ax = axes[1, 1]
    ax.bar(x - width/2, results_df['Train_Clusters'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_Clusters'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Number of Clusters Discovered')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"Saved metrics comparison: {metrics_path}")
    plt.close()


def main():
    """Main execution."""
    # Load data
    X_train, X_test, train_df, test_df, feature_cols = load_data()
    
    # Train all models
    results_df, models_data = train_all_models(X_train, X_test)
    
    # Calculate rankings
    results_ranked = calculate_rankings(results_df)
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    ranked_path = os.path.join(OUTPUT_DIR, "ranked_models.csv")
    results_ranked[['Model', 'Params', 'Test_Silhouette', 'Test_DBI', 'Test_CH',
                    'Sil_Normalized', 'DBI_Normalized', 'CH_Normalized', 'Combined_Score']].to_csv(ranked_path, index=False)
    print(f"Ranked results saved to: {ranked_path}")
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Generalization analysis
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS")
    print("="*80)
    for _, row in results_df.iterrows():
        if not np.isnan(row['Train_Silhouette']) and not np.isnan(row['Test_Silhouette']):
            ratio = (row['Test_Silhouette'] / row['Train_Silhouette']) * 100
            print(f"{row['Model']:10s}: {ratio:.1f}% retention (Train={row['Train_Silhouette']:.3f}, Test={row['Test_Silhouette']:.3f})")
    
    # Rankings
    print("\n" + "="*80)
    print("FINAL RANKING (by Combined Score)")
    print("="*80)
    for rank, (idx, row) in enumerate(results_ranked.iterrows(), 1):
        print(f"{rank}. {row['Model']:10s} - Score: {row['Combined_Score']:.4f} | "
              f"Sil: {row['Sil_Normalized']:.3f} | DBI: {row['DBI_Normalized']:.3f} | CH: {row['CH_Normalized']:.3f}")
    
    # Winner
    winner = results_ranked.iloc[0]
    print("\n" + "=" * 40)
    print(f"WINNER: {winner['Model']}")
    print(f"Parameters: {winner['Params']}")
    print(f"Combined Score: {winner['Combined_Score']:.4f}")
    print(f"  • Silhouette: {winner['Test_Silhouette']:.4f}")
    print(f"  • DBI: {winner['Test_DBI']:.4f}")
    print(f"  • CH: {winner['Test_CH']:.4f}")
    print("=" * 40)
    
    # Visualizations
    visualize_comparison(X_train, X_test, models_data, results_df)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
