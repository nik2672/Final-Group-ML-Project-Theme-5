"""
Final Model Comparison with Best Tuned Parameters

This script trains all 5 clustering models using their optimal hyperparameters
(found by tuning.py) and evaluates them on BOTH train and test data.

Purpose: Final evaluation to compare model performance and generalization

Steps:
1. Load best parameters from tuning results
2. Train each model on TRAIN data with optimal params
3. Assign TEST data to clusters using KNN
4. Calculate metrics on both train and test
5. Generate visualizations and comparison tables
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
TUNING_RESULTS = os.path.join(PROJECT_ROOT, "results", "clustering", "tuning")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "clustering", "final_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("FINAL MODEL COMPARISON - TRAIN & TEST EVALUATION")
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
    
    # Select feature columns (exclude identifiers and target)
    feature_cols = ["latitude", "longitude", "avg_latency", "std_latency", 
                   "total_throughput", "zone_avg_latency", "zone_avg_upload", "zone_avg_download"]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Fill missing values and scale
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaled - TRAIN: {X_train_scaled.shape}, TEST: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, train_df, test_df, feature_cols


def load_best_params():
    """Load optimal parameters from tuning results."""
    print("\n=== Loading Best Parameters from Tuning ===")
    best_params_path = os.path.join(TUNING_RESULTS, "best_per_model_combined.csv")
    
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Tuning results not found. Run tuning.py first.")
    
    df = pd.read_csv(best_params_path)
    print(f"Loaded best parameters for {len(df)} models")
    
    # Parse parameters into dictionary
    params_dict = {}
    for _, row in df.iterrows():
        model_name = row['Model']
        param_str = row['Param']
        
        # Parse parameter string
        if model_name == "KMeans":
            k = int(param_str.split('=')[1])
            params_dict[model_name] = {'n_clusters': k}
            
        elif model_name == "DBSCAN":
            parts = param_str.split(',')
            eps = float(parts[0].split('=')[1])
            min_samples = int(parts[1].split('=')[1])
            params_dict[model_name] = {'eps': eps, 'min_samples': min_samples}
            
        elif model_name == "Birch":
            parts = param_str.split(',')
            k = int(parts[0].split('=')[1])
            thr = float(parts[1].split('=')[1])
            bf = int(parts[2].split('=')[1])
            params_dict[model_name] = {'n_clusters': k, 'threshold': thr, 'branching_factor': bf}
            
        elif model_name == "OPTICS":
            parts = param_str.split(',')
            min_samples = int(parts[0].split('=')[1])
            max_eps_str = parts[1].split('=')[1]
            max_eps = np.inf if max_eps_str == 'inf' else float(max_eps_str)
            xi = float(parts[2].split('=')[1])
            method = parts[3].split('=')[1]
            params_dict[model_name] = {'min_samples': min_samples, 'max_eps': max_eps, 
                                      'xi': xi, 'cluster_method': method}
            
        elif model_name == "HDBSCAN":
            parts = param_str.split(',')
            mcs = int(parts[0].split('=')[1])
            ms_str = parts[1].split('=')[1]
            ms = None if ms_str == 'None' else int(ms_str)
            params_dict[model_name] = {'min_cluster_size': mcs, 'min_samples': ms}
    
    print("\nParsed parameters:")
    for model, params in params_dict.items():
        print(f"  {model}: {params}")
    
    return params_dict


def assign_test_to_clusters(X_train, labels_train, X_test):
    """Assign test data to clusters using KNN."""
    # Remove outliers from training for KNN
    mask = labels_train != -1
    X_train_valid = X_train[mask]
    labels_train_valid = labels_train[mask]
    
    if len(X_train_valid) == 0:
        # All outliers, assign all test as outliers
        return np.full(len(X_test), -1)
    
    # Use KNN to find nearest training samples
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X_train_valid)
    distances, indices = knn.kneighbors(X_test)
    
    # Assign test points to cluster of nearest training point
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


def train_and_evaluate_all(X_train, X_test, params_dict):
    """Train all models and evaluate on train/test."""
    print("\n=== Training and Evaluating All Models ===")
    
    results = []
    models_data = {}
    
    # 1. KMeans
    if "KMeans" in params_dict:
        print("\n[1/5] KMeans...")
        params = params_dict["KMeans"]
        model = KMeans(n_clusters=params['n_clusters'], random_state=42)
        labels_train = model.fit_predict(X_train)
        labels_test = model.predict(X_test)
        
        metrics_train = safe_metrics(X_train, labels_train)
        metrics_test = safe_metrics(X_test, labels_test)
        
        results.append({
            'Model': 'KMeans',
            'Params': f"k={params['n_clusters']}",
            'Train_Silhouette': metrics_train['silhouette'],
            'Test_Silhouette': metrics_test['silhouette'],
            'Train_DBI': metrics_train['davies_bouldin'],
            'Test_DBI': metrics_test['davies_bouldin'],
            'Train_CH': metrics_train['calinski_harabasz'],
            'Test_CH': metrics_test['calinski_harabasz'],
            'Train_Clusters': metrics_train['n_clusters'],
            'Test_Clusters': metrics_test['n_clusters']
        })
        
        models_data['KMeans'] = {
            'train_labels': labels_train,
            'test_labels': labels_test,
            'model': model
        }
        print(f"  Train: Sil={metrics_train['silhouette']:.3f}, DBI={metrics_train['davies_bouldin']:.3f}")
        print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, DBI={metrics_test['davies_bouldin']:.3f}")
    
    # 2. DBSCAN
    if "DBSCAN" in params_dict:
        print("\n[2/5] DBSCAN...")
        params = params_dict["DBSCAN"]
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels_train = model.fit_predict(X_train)
        labels_test = assign_test_to_clusters(X_train, labels_train, X_test)
        
        metrics_train = safe_metrics(X_train, labels_train)
        metrics_test = safe_metrics(X_test, labels_test)
        
        results.append({
            'Model': 'DBSCAN',
            'Params': f"eps={params['eps']},min={params['min_samples']}",
            'Train_Silhouette': metrics_train['silhouette'],
            'Test_Silhouette': metrics_test['silhouette'],
            'Train_DBI': metrics_train['davies_bouldin'],
            'Test_DBI': metrics_test['davies_bouldin'],
            'Train_CH': metrics_train['calinski_harabasz'],
            'Test_CH': metrics_test['calinski_harabasz'],
            'Train_Clusters': metrics_train['n_clusters'],
            'Test_Clusters': metrics_test['n_clusters']
        })
        
        models_data['DBSCAN'] = {
            'train_labels': labels_train,
            'test_labels': labels_test,
            'model': model
        }
        print(f"  Train: Sil={metrics_train['silhouette']:.3f}, Clusters={metrics_train['n_clusters']}, Outliers={metrics_train['n_outliers']}")
        print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, Clusters={metrics_test['n_clusters']}, Outliers={metrics_test['n_outliers']}")
    
    # 3. Birch
    if "Birch" in params_dict:
        print("\n[3/5] Birch...")
        params = params_dict["Birch"]
        model = Birch(n_clusters=params['n_clusters'], threshold=params['threshold'], 
                     branching_factor=params['branching_factor'])
        labels_train = model.fit_predict(X_train)
        labels_test = model.predict(X_test)
        
        metrics_train = safe_metrics(X_train, labels_train)
        metrics_test = safe_metrics(X_test, labels_test)
        
        results.append({
            'Model': 'Birch',
            'Params': f"k={params['n_clusters']},thr={params['threshold']},bf={params['branching_factor']}",
            'Train_Silhouette': metrics_train['silhouette'],
            'Test_Silhouette': metrics_test['silhouette'],
            'Train_DBI': metrics_train['davies_bouldin'],
            'Test_DBI': metrics_test['davies_bouldin'],
            'Train_CH': metrics_train['calinski_harabasz'],
            'Test_CH': metrics_test['calinski_harabasz'],
            'Train_Clusters': metrics_train['n_clusters'],
            'Test_Clusters': metrics_test['n_clusters']
        })
        
        models_data['Birch'] = {
            'train_labels': labels_train,
            'test_labels': labels_test,
            'model': model
        }
        print(f"  Train: Sil={metrics_train['silhouette']:.3f}, DBI={metrics_train['davies_bouldin']:.3f}")
        print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, DBI={metrics_test['davies_bouldin']:.3f}")
    
    # 4. OPTICS
    if "OPTICS" in params_dict:
        print("\n[4/5] OPTICS...")
        params = params_dict["OPTICS"]
        model = OPTICS(min_samples=params['min_samples'], max_eps=params['max_eps'],
                      xi=params['xi'], cluster_method=params['cluster_method'])
        labels_train = model.fit_predict(X_train)
        labels_test = assign_test_to_clusters(X_train, labels_train, X_test)
        
        metrics_train = safe_metrics(X_train, labels_train)
        metrics_test = safe_metrics(X_test, labels_test)
        
        max_eps_str = 'inf' if np.isinf(params['max_eps']) else params['max_eps']
        results.append({
            'Model': 'OPTICS',
            'Params': f"min={params['min_samples']},eps={max_eps_str},xi={params['xi']},method={params['cluster_method']}",
            'Train_Silhouette': metrics_train['silhouette'],
            'Test_Silhouette': metrics_test['silhouette'],
            'Train_DBI': metrics_train['davies_bouldin'],
            'Test_DBI': metrics_test['davies_bouldin'],
            'Train_CH': metrics_train['calinski_harabasz'],
            'Test_CH': metrics_test['calinski_harabasz'],
            'Train_Clusters': metrics_train['n_clusters'],
            'Test_Clusters': metrics_test['n_clusters']
        })
        
        models_data['OPTICS'] = {
            'train_labels': labels_train,
            'test_labels': labels_test,
            'model': model
        }
        print(f"  Train: Sil={metrics_train['silhouette']:.3f}, Clusters={metrics_train['n_clusters']}, Outliers={metrics_train['n_outliers']}")
        print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, Clusters={metrics_test['n_clusters']}, Outliers={metrics_test['n_outliers']}")
    
    # 5. HDBSCAN
    if "HDBSCAN" in params_dict and HDBSCAN_AVAILABLE:
        print("\n[5/5] HDBSCAN...")
        params = params_dict["HDBSCAN"]
        model = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], 
                               min_samples=params['min_samples'],
                               cluster_selection_method='eom')
        labels_train = model.fit_predict(X_train)
        labels_test = assign_test_to_clusters(X_train, labels_train, X_test)
        
        metrics_train = safe_metrics(X_train, labels_train)
        metrics_test = safe_metrics(X_test, labels_test)
        
        ms_str = 'None' if params['min_samples'] is None else params['min_samples']
        results.append({
            'Model': 'HDBSCAN',
            'Params': f"mcs={params['min_cluster_size']},ms={ms_str}",
            'Train_Silhouette': metrics_train['silhouette'],
            'Test_Silhouette': metrics_test['silhouette'],
            'Train_DBI': metrics_train['davies_bouldin'],
            'Test_DBI': metrics_test['davies_bouldin'],
            'Train_CH': metrics_train['calinski_harabasz'],
            'Test_CH': metrics_test['calinski_harabasz'],
            'Train_Clusters': metrics_train['n_clusters'],
            'Test_Clusters': metrics_test['n_clusters']
        })
        
        models_data['HDBSCAN'] = {
            'train_labels': labels_train,
            'test_labels': labels_test,
            'model': model
        }
        print(f"  Train: Sil={metrics_train['silhouette']:.3f}, Clusters={metrics_train['n_clusters']}, Outliers={metrics_train['n_outliers']}")
        print(f"  Test:  Sil={metrics_test['silhouette']:.3f}, Clusters={metrics_test['n_clusters']}, Outliers={metrics_test['n_outliers']}")
    
    return pd.DataFrame(results), models_data


def visualize_comparison(X_train, X_test, models_data, results_df):
    """Create comparison visualizations."""
    print("\n=== Generating Visualizations ===")
    
    # Use PCA for 2D projection
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    
    n_models = len(models_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        # Train plot
        ax_train = axes[0, idx]
        labels_train = data['train_labels']
        scatter_train = ax_train.scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                                        c=labels_train, cmap='tab10', alpha=0.6, s=30)
        ax_train.set_title(f"{model_name} - TRAIN\n{len(set(labels_train))-1} clusters", fontsize=12)
        ax_train.set_xlabel("PC1")
        ax_train.set_ylabel("PC2")
        plt.colorbar(scatter_train, ax=ax_train)
        
        # Test plot
        ax_test = axes[1, idx]
        labels_test = data['test_labels']
        scatter_test = ax_test.scatter(X_test_2d[:, 0], X_test_2d[:, 1], 
                                      c=labels_test, cmap='tab10', alpha=0.6, s=30)
        ax_test.set_title(f"{model_name} - TEST\n{len(set(labels_test))-1} clusters", fontsize=12)
        ax_test.set_xlabel("PC1")
        ax_test.set_ylabel("PC2")
        plt.colorbar(scatter_test, ax=ax_test)
    
    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, "all_models_comparison.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {vis_path}")
    plt.close()
    
    # Create metrics comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Silhouette comparison
    ax = axes[0, 0]
    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, results_df['Train_Silhouette'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_Silhouette'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # DBI comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, results_df['Train_DBI'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_DBI'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Calinski-Harabasz comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, results_df['Train_CH'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['Test_CH'], width, label='Test', alpha=0.8)
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Score (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Cluster count comparison
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
    
    # Load best parameters from tuning
    params_dict = load_best_params()
    
    # Train and evaluate all models
    results_df, models_data = train_and_evaluate_all(X_train, X_test, params_dict)
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "final_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n=== Results saved to: {results_path} ===")
    
    # Display results
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS (All Models with Best Parameters)")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Calculate generalization (test/train ratio for silhouette)
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS (Test vs Train)")
    print("="*80)
    for _, row in results_df.iterrows():
        if not np.isnan(row['Train_Silhouette']) and not np.isnan(row['Test_Silhouette']):
            ratio = (row['Test_Silhouette'] / row['Train_Silhouette']) * 100
            print(f"{row['Model']:10s}: Test Silhouette = {ratio:.1f}% of Train (Test={row['Test_Silhouette']:.3f}, Train={row['Train_Silhouette']:.3f})")
    
    # Find best model by individual metrics
    print("\n" + "="*80)
    print("BEST MODEL BY INDIVIDUAL METRIC (TEST DATA)")
    print("="*80)
    best_test_sil = results_df.loc[results_df['Test_Silhouette'].idxmax()]
    best_test_dbi = results_df.loc[results_df['Test_DBI'].idxmin()]
    best_test_ch = results_df.loc[results_df['Test_CH'].idxmax()]
    
    print(f"Best Test Silhouette (higher=better): {best_test_sil['Model']:10s} = {best_test_sil['Test_Silhouette']:.4f}")
    print(f"Best Test DBI (lower=better):         {best_test_dbi['Model']:10s} = {best_test_dbi['Test_DBI']:.4f}")
    print(f"Best Test Calinski-Harabasz:          {best_test_ch['Model']:10s} = {best_test_ch['Test_CH']:.4f}")
    
    # Calculate OVERALL BEST using normalized combined score
    print("\n" + "="*80)
    print("OVERALL BEST MODEL (Combined Score on TEST Data)")
    print("="*80)
    print("Methodology: Normalize all 3 metrics to 0-1 scale, then combine")
    print("  - Silhouette Score (0-1, higher=better): 40% weight")
    print("  - Davies-Bouldin Index (inverted, lower=better): 30% weight")
    print("  - Calinski-Harabasz Score (normalized, higher=better): 30% weight")
    print()
    
    # Normalize metrics to 0-1 scale
    results_df_ranked = results_df.copy()
    
    # Silhouette: already 0-1, keep as is (higher is better)
    sil_min = results_df['Test_Silhouette'].min()
    sil_max = results_df['Test_Silhouette'].max()
    results_df_ranked['Sil_Normalized'] = (results_df['Test_Silhouette'] - sil_min) / (sil_max - sil_min) if sil_max > sil_min else 0
    
    # DBI: lower is better, so invert it (1 - normalized)
    dbi_min = results_df['Test_DBI'].min()
    dbi_max = results_df['Test_DBI'].max()
    results_df_ranked['DBI_Normalized'] = 1 - ((results_df['Test_DBI'] - dbi_min) / (dbi_max - dbi_min)) if dbi_max > dbi_min else 0
    
    # Calinski-Harabasz: normalize to 0-1 (higher is better)
    ch_min = results_df['Test_CH'].min()
    ch_max = results_df['Test_CH'].max()
    results_df_ranked['CH_Normalized'] = (results_df['Test_CH'] - ch_min) / (ch_max - ch_min) if ch_max > ch_min else 0
    
    # Combined score (weighted average)
    results_df_ranked['Combined_Score'] = (
        results_df_ranked['Sil_Normalized'] * 0.40 +
        results_df_ranked['DBI_Normalized'] * 0.30 +
        results_df_ranked['CH_Normalized'] * 0.30
    )
    
    # Sort by combined score
    results_df_ranked = results_df_ranked.sort_values('Combined_Score', ascending=False)
    
    print("RANKING (Best to Worst):")
    print("-" * 100)
    for rank, (idx, row) in enumerate(results_df_ranked.iterrows(), 1):
        print(f"{rank}. {row['Model']:10s} - Combined Score: {row['Combined_Score']:.4f} | "
              f"Sil: {row['Sil_Normalized']:.3f} | DBI: {row['DBI_Normalized']:.3f} | CH: {row['CH_Normalized']:.3f}")
    
    # Save ranked results
    ranked_path = os.path.join(OUTPUT_DIR, "ranked_models.csv")
    results_df_ranked[['Model', 'Params', 'Test_Silhouette', 'Test_DBI', 'Test_CH', 
                       'Sil_Normalized', 'DBI_Normalized', 'CH_Normalized', 'Combined_Score']].to_csv(ranked_path, index=False)
    print(f"\nRanked results saved to: {ranked_path}")
    
    # Declare winner
    winner = results_df_ranked.iloc[0]
    print("\n" + "=" * 40)
    print(f"OVERALL WINNER: {winner['Model']}")
    print(f"Parameters: {winner['Params']}")
    print(f"Combined Score: {winner['Combined_Score']:.4f}")
    print(f"  • Silhouette:         {winner['Test_Silhouette']:.4f} (normalized: {winner['Sil_Normalized']:.3f})")
    print(f"  • Davies-Bouldin:     {winner['Test_DBI']:.4f} (normalized: {winner['DBI_Normalized']:.3f})")
    print(f"  • Calinski-Harabasz:  {winner['Test_CH']:.4f} (normalized: {winner['CH_Normalized']:.3f})")
    print("=" * 40)
    
    # Visualizations
    visualize_comparison(X_train, X_test, models_data, results_df_ranked)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
