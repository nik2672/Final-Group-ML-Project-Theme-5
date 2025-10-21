"""
This program analyzes 5G network performance data to identify natural groups of similar geographic areas.

The analysis categorizes network zones based on performance characteristics:
- High-performance areas (optimal coverage zones)
- Low-performance areas (suboptimal coverage zones)  
- Mixed performance areas (moderate coverage zones)

Two clustering algorithms are implemented:
1. Birch: Memory-efficient hierarchical clustering with tree-based data organization
2. OPTICS: Density-based clustering that identifies clusters of varying sizes and outliers

This analysis enables telecom operators to optimize 5G network infrastructure deployment.
"""

# Import required libraries
import os                          # For file and folder operations
import platform                    # For system architecture detection
import numpy as np                 # For mathematical operations on large datasets
import pandas as pd                # For handling spreadsheet-like data
import matplotlib.pyplot as plt    # For creating charts and graphs
from sklearn.preprocessing import StandardScaler         # To normalize data (make all numbers comparable)
from sklearn.metrics import silhouette_score, davies_bouldin_score  # For clustering quality evaluation
from sklearn.cluster import Birch, OPTICS              # Clustering algorithm implementations
import multiprocessing            # For parallel processing capabilities
from tqdm import tqdm             # For progress bars in terminal

# Set up file locations where data is stored and results will be saved
_HERE = os.path.dirname(os.path.abspath(__file__))      # Current file location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # Main project folder
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')         # Where input data files are stored
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'clustering')  # Results directory location

os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create results folder if it doesn't exist

# Detect computer specifications for optimization
N_CORES = multiprocessing.cpu_count()   # Count how many CPU cores are available
IS_M_SERIES = platform.system() == 'Darwin' and platform.machine() == 'arm64'  # Apple M-series detection

if IS_M_SERIES:
    print(f"M-Series chip detected ({N_CORES} cores) - using optimized settings")
else:
    print(f"Standard architecture detected ({N_CORES} cores)")


def load_clustering_data():
    """
    Load the 5G network performance data from separate train/test files (leakage-safe).
    
    Returns:
        tuple: (train_df, test_df) - Network performance measurements split by time
    """
    print("\nLoading 5G network performance data (train/test split)...")
    
    # Define data file locations - using new improved features
    train_path = os.path.join(DATA_PATH, 'features_for_clustering_train_improved.csv')
    test_path = os.path.join(DATA_PATH, 'features_for_clustering_test_improved.csv')

    # Check if the data files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train data not found at {train_path}. "
            "Please run feature engineering first: python src/features/leakage_safe_feature_engineering.py"
        )
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Please run feature engineering first: python src/features/leakage_safe_feature_engineering.py"
        )

    # Load the data 
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    
    print(f"Successfully loaded TRAIN: {len(train_df):,} measurements with {len(train_df.columns)} metrics")
    print(f"Successfully loaded TEST: {len(test_df):,} measurements with {len(test_df.columns)} metrics")
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Prepare TRAIN and TEST data separately for leakage-safe analysis.
    
    Fits scaler on TRAIN only, then transforms both TRAIN and TEST.
    
    Args:
        train_df (pandas.DataFrame): Training network performance data
        test_df (pandas.DataFrame): Test network performance data
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, feature_names, scaler, train_processed, test_processed)
    """
    print("\nPreparing data for clustering analysis (leakage-safe)...")
    
    # Zone-level aggregation for both train and test
    def aggregate_zones(df, label):
        if 'square_id' in df.columns:
            print(f"Aggregating {len(df):,} {label} measurements by zone (square_id)...")
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
            print(f"Reduced to {len(zone_agg):,} {label} zones")
            return zone_agg
        return df.copy()
    
    train_processed = aggregate_zones(train_df, "train")
    test_processed = aggregate_zones(test_df, "test")
    
    # Use consistent features for team analysis compatibility
    feature_cols = [
        'latitude', 'longitude',
        'avg_latency', 'std_latency', 'total_throughput',
        'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    feature_cols = [col for col in feature_cols if col in train_processed.columns]

    # If expected measurements are missing, use available numeric data
    if len(feature_cols) < 3:
        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols[:8]  # Take the first 8 numeric measurements
        
    print(f"Selected features for clustering: {feature_cols}")
    
    # Extract features from TRAIN and TEST
    X_train = train_processed[feature_cols].fillna(0)
    X_test = test_processed[feature_cols].fillna(0)
    
    # IMPORTANT: Fit scaler on TRAIN only, then transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use same scaler fitted on train
    
    print(f"Prepared TRAIN: {X_train_scaled.shape[0]:,} zones with {X_train_scaled.shape[1]} features")
    print(f"Prepared TEST: {X_test_scaled.shape[0]:,} zones with {X_test_scaled.shape[1]} features")
    
    return X_train_scaled, X_test_scaled, feature_cols, scaler, train_processed, test_processed


def run_birch_clustering(X, max_clusters=6):
    """
    Run Birch clustering algorithm optimized for large datasets.
    
    Birch is good for:
    - Very large datasets (memory efficient)
    - Incremental learning (can add new data)
    - Spherical clusters similar to K-Means
    """
    print("\nRunning Birch clustering...")
    
    # For large datasets, sample for silhouette score calculation (too slow on 2.4M samples)
    sample_size = min(50000, len(X))  # Use max 50k samples for evaluation
    if len(X) > sample_size:
        print(f"Large dataset detected ({len(X):,} samples). Using {sample_size:,} samples for optimization...")
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X
        sample_indices = None
    
    # Test different numbers of clusters
    cluster_range = range(2, max_clusters + 1)
    best_score = -1
    best_k = 3  # Default to 3 clusters
    
    print("Testing different cluster numbers (optimized for speed)...")
    
    # Use tqdm progress bar 
    for k in tqdm(cluster_range, desc="Birch optimization"):
        # Configure Birch parameters
        birch = Birch(
            n_clusters=k,         # Final number of clusters
            threshold=0.3,        # Adjusted for better performance
            branching_factor=50   # Max subclusters in each node
        )
        
        # Use sample for evaluation
        sample_labels = birch.fit_predict(X_sample)
        
        # Quick evaluation using Davies-Bouldin score (faster than silhouette)
        try:
            db_score = davies_bouldin_score(X_sample, sample_labels)
            # Lower Davies-Bouldin is better, so invert for comparison
            eval_score = 1.0 / (1.0 + db_score)
            tqdm.write(f"  k={k}: Score = {eval_score:.3f}")
        except:
            eval_score = 0.1
            tqdm.write(f"  k={k}: Score = 0.100 (evaluation failed)")
        
        if eval_score > best_score:
            best_score = eval_score
            best_k = k
    
    # Fit final Birch model on full dataset
    print(f"\nCreating final model with {best_k} clusters on full dataset...")
    
    with tqdm(total=1, desc="Final Birch clustering") as pbar:
        final_birch = Birch(
            n_clusters=best_k,
            threshold=0.3,
            branching_factor=50
        )
        final_labels = final_birch.fit_predict(X)
        pbar.update(1)
    
    print(f"Birch Results:")
    print(f"  Optimal clusters: {best_k}")
    print(f"  Evaluation score: {best_score:.3f}")
    print(f"  Cluster sizes: {np.bincount(final_labels)}")
    print(f"  Processed {len(X):,} network locations successfully")
    
    return final_labels, final_birch, best_k


def run_optics_clustering(X):
    """
    Execute OPTICS algorithm to identify natural clusters in network performance data.
    
    OPTICS (Ordering Points To Identify Cluster Structure) provides density-based clustering:
    - Identifies densely packed regions representing similar network performance
    - Automatically determines cluster shapes and sizes without predetermined parameters
    - Detects outliers representing locations with anomalous performance characteristics
    
    Unlike partitioning methods, OPTICS discovers natural data patterns automatically
    without requiring specification of cluster count.
    
    Args:
        X (numpy.ndarray): Standardized feature matrix
        
    Returns:
        tuple: Cluster labels, fitted model, number of clusters
    """
    print("\nUsing OPTICS algorithm to discover natural performance groups...")
    
    # Set up OPTICS for performance-based clustering (no geographic features)
    # With 274 zones and performance-only features, these parameters balance:
    # - Finding meaningful clusters (not too strict)
    # - Identifying true outliers (not too permissive)
    optics = OPTICS(
        min_samples=5,           # Relaxed for smaller dataset (5 neighbors required)
        max_eps=2.0,             # Generous distance threshold for performance similarity
        cluster_method='xi',     # Steep boundary detection method
        xi=0.1,                 # Moderate steepness (0.1 = 10% relative density change)
        n_jobs=-1                # Use all available CPU cores
    )
    
    # Analyze the data and assign each location to a group (or mark as outlier)
    print("Analyzing network data to find natural groupings...")
    
    with tqdm(total=1, desc="OPTICS clustering") as pbar:
        labels = optics.fit_predict(X)
        pbar.update(1)
    
    # Count how many groups were found and how many outliers
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 means outlier
    n_noise = np.sum(labels == -1)  # Count outliers
    
    print(f"OPTICS Analysis Results:")
    print(f"  Natural groups discovered: {n_clusters}")
    print(f"  Outlier locations: {n_noise:,} ({n_noise/len(labels)*100:.1f}% of total)")
    
    # Measure the quality of the grouping (only for locations that belong to groups)
    if n_clusters > 1:
        non_noise_mask = labels != -1  # Exclude outliers from quality measurement
        if np.sum(non_noise_mask) > 0:
            sil_score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
            print(f"  Grouping quality score: {sil_score:.3f} (higher is better)")
    
    # Show how many locations are in each group
    if n_clusters > 0:
        print("  Group sizes:")
        unique_labels = set(labels)
        for label in sorted(unique_labels):
            size = np.sum(labels == label)
            if label == -1:
                print(f"    Outliers: {size:,} locations")
            else:
                print(f"    Group {label}: {size:,} locations")
    
    return labels, optics, n_clusters


def plot_birch_results(X, birch_labels, feature_names):
    """
    Generate visualization for Birch clustering results.
    
    Creates scatter plot showing network location clusters identified by Birch algorithm.
    Each color represents a distinct performance zone with similar characteristics.
    
    Args:
        X (numpy.ndarray): Standardized feature data
        birch_labels (numpy.ndarray): Cluster assignments
        feature_names (list): Names of features for axis labels
    """
    if birch_labels is None:
        return
        
    print("Creating Birch clustering visualization...")
    
    plt.figure(figsize=(10, 8))
    
    n_birch = len(set(birch_labels))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=birch_labels, 
                         cmap='Set3', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Birch Cluster')
    
    plt.title(f'Birch Clustering: {n_birch} Performance Zones', fontsize=14, fontweight='bold')
    plt.xlabel(feature_names[0] if len(feature_names) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(feature_names[1] if len(feature_names) > 1 else 'Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save individual Birch chart
    output_path = os.path.join(OUTPUT_PATH, 'birch_clusters.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Birch chart saved as: {output_path}")
    plt.close()


def plot_optics_results(X, optics_labels, feature_names):
    """
    Generate visualization for OPTICS clustering results.
    
    Creates scatter plot displaying natural clusters discovered by OPTICS algorithm.
    Gray markers indicate outlier locations with anomalous performance patterns.
    
    Args:
        X (numpy.ndarray): Standardized feature data
        optics_labels (numpy.ndarray): Cluster assignments with outliers marked as -1
        feature_names (list): Names of features for axis labels
    """
    if optics_labels is None:
        return
        
    print("Creating OPTICS clustering visualization...")
    
    plt.figure(figsize=(10, 8))
    
    n_optics = len(set(optics_labels)) - (1 if -1 in optics_labels else 0)
    
    # Handle outliers and clusters separately
    noise_mask = optics_labels == -1
    cluster_mask = optics_labels != -1
    
    # Plot outliers in gray
    if np.sum(noise_mask) > 0:
        plt.scatter(X[noise_mask, 0], X[noise_mask, 1], 
                   c='lightgray', alpha=0.6, s=50, label='Outliers')
    
    # Plot clusters in colors
    if np.sum(cluster_mask) > 0:
        scatter = plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], 
                            c=optics_labels[cluster_mask], cmap='Set3', 
                            alpha=0.7, s=50)
        plt.colorbar(scatter, label='OPTICS Cluster')
    
    plt.title(f'OPTICS Clustering: {n_optics} Natural Groups', fontsize=14, fontweight='bold')
    plt.xlabel(feature_names[0] if len(feature_names) > 0 else 'Feature 1', fontsize=12)
    plt.ylabel(feature_names[1] if len(feature_names) > 1 else 'Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if np.sum(noise_mask) > 0:
        plt.legend()
    
    # Save individual OPTICS chart
    output_path = os.path.join(OUTPUT_PATH, 'optics_clusters.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"OPTICS chart saved as: {output_path}")
    plt.close()



def save_birch_results(df, birch_labels):
    """Save Birch clustering results to separate CSV file."""
    if birch_labels is None:
        return
    
    print("Saving Birch clustering results...")
    
    results_df = df.copy()
    results_df['birch_cluster'] = birch_labels
    
    output_file = os.path.join(OUTPUT_PATH, 'birch_clusters.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Birch results saved: {output_file}")
    
    # Show Birch summary
    print("\nBirch Method Results:")
    unique, counts = np.unique(birch_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Zone {cluster_id}: {count:,} network locations")


def save_optics_results(df, optics_labels):
    """Save OPTICS clustering results to separate CSV file."""
    if optics_labels is None:
        return
        
    print("Saving OPTICS clustering results...")
    
    results_df = df.copy()
    results_df['optics_cluster'] = optics_labels
    
    output_file = os.path.join(OUTPUT_PATH, 'optics_clusters.csv')
    results_df.to_csv(output_file, index=False)
    print(f"OPTICS results saved: {output_file}")
    
    # Show OPTICS summary  
    print("\nOPTICS Method Results:")
    unique, counts = np.unique(optics_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        if cluster_id == -1:
            print(f"   Outlier locations: {count:,} (need special attention)")
        else:
            print(f"   Natural Group {cluster_id}: {count:,} network locations")


def assign_test_clusters_knn(X_train, train_labels, X_test, n_neighbors=5):
    """
    Assign test data to clusters using K-Nearest Neighbors based on train clusters.
    
    This prevents data leakage by:
    1. Training clustering on TRAIN data only
    2. Using KNN to assign TEST points to nearest train clusters
    
    Args:
        X_train: Scaled training features
        train_labels: Cluster labels from training
        X_test: Scaled test features
        n_neighbors: Number of neighbors to consider
        
    Returns:
        Test cluster labels
    """
    from sklearn.neighbors import NearestNeighbors
    
    print(f"\nAssigning test data to clusters using {n_neighbors}-NN...")
    
    # For OPTICS: handle outliers separately
    if -1 in train_labels:
        # Only use non-outlier points for KNN
        mask = train_labels != -1
        if mask.sum() == 0:
            print("Warning: All train points are outliers, assigning all test as outliers")
            return np.full(len(X_test), -1, dtype=int)
        
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, mask.sum())).fit(X_train[mask])
        indices = nn.kneighbors(X_test, return_distance=False)
        
        # Majority vote from neighbors
        test_labels = []
        train_labels_clean = train_labels[mask]
        for neighbor_idx in indices:
            neighbor_labels = train_labels_clean[neighbor_idx]
            # Remove outliers from voting
            valid_labels = neighbor_labels[neighbor_labels != -1]
            if len(valid_labels) > 0:
                test_labels.append(np.bincount(valid_labels).argmax())
            else:
                test_labels.append(-1)  # All neighbors are outliers
        
        return np.array(test_labels, dtype=int)
    
    else:
        # Standard case (Birch): all points have valid clusters
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
        indices = nn.kneighbors(X_test, return_distance=False)
        
        # Majority vote
        test_labels = []
        for neighbor_idx in indices:
            neighbor_labels = train_labels[neighbor_idx]
            test_labels.append(np.bincount(neighbor_labels).argmax())
        
        return np.array(test_labels, dtype=int)


def main():
    """
    Execute the complete clustering analysis pipeline.
    
    Process steps:
    1. Load network performance data
    2. Prepare features for analysis
    3. Execute Birch and OPTICS clustering algorithms
    4. Generate result visualizations
    5. Save cluster assignments and analysis results
    """
    print("=" * 60)
    print("5G NETWORK PERFORMANCE ZONE ANALYSIS")
    print("Finding natural groups in network coverage areas using two different methods")
    if IS_M_SERIES:
        print(f"Running on Apple M-Series chip with {N_CORES} cores for optimal performance")
    else:
        print(f"Running on {N_CORES} CPU cores")
    print("=" * 60)

    try:
        # Step 1: Load the 5G network measurement data (train/test split)
        print("\n=== STEP 1: LOADING DATA ===")
        train_df, test_df = load_clustering_data()

        # Step 2: Prepare the data for analysis (leakage-safe)
        print("\n=== STEP 2: PREPARING DATA ===")
        X_train_scaled, X_test_scaled, feature_names, scaler, train_processed, test_processed = prepare_features(train_df, test_df)

        # Step 3: Use Birch algorithm to find performance zones (FIT ON TRAIN ONLY)
        print("\n=== STEP 3: BIRCH ANALYSIS ===")
        birch_train_labels, birch_model, n_birch = run_birch_clustering(X_train_scaled)
        
        # Assign test data using KNN
        birch_test_labels = assign_test_clusters_knn(X_train_scaled, birch_train_labels, X_test_scaled)
        print(f"Birch test assignments: {len(birch_test_labels)} zones assigned")

        # Step 4: Use OPTICS algorithm to find natural groupings (FIT ON TRAIN ONLY)
        print("\n=== STEP 4: OPTICS ANALYSIS ===")
        optics_train_labels, optics_model, n_optics = run_optics_clustering(X_train_scaled)
        
        # Assign test data using KNN
        optics_test_labels = assign_test_clusters_knn(X_train_scaled, optics_train_labels, X_test_scaled)
        print(f"OPTICS test assignments: {len(optics_test_labels)} zones assigned")

        # Step 5: Create individual visualizations (using train data)
        print("\n=== STEP 5: CREATING VISUALIZATIONS ===")
        plot_birch_results(X_train_scaled, birch_train_labels, feature_names)
        plot_optics_results(X_train_scaled, optics_train_labels, feature_names)
        
        # Step 6: Save results to separate files (TRAIN and TEST separately)
        print("\n=== STEP 6: SAVING RESULTS ===")
        save_birch_results(train_processed, birch_train_labels)
        save_optics_results(train_processed, optics_train_labels)
        
        # Save TEST results separately
        print("\nSaving TEST results...")
        test_birch_df = test_processed.copy()
        test_birch_df['birch_cluster'] = birch_test_labels
        test_birch_output = os.path.join(OUTPUT_PATH, 'birch_clusters_test.csv')
        test_birch_df.to_csv(test_birch_output, index=False)
        print(f"Birch TEST results saved: {test_birch_output}")
        
        test_optics_df = test_processed.copy()
        test_optics_df['optics_cluster'] = optics_test_labels
        test_optics_output = os.path.join(OUTPUT_PATH, 'optics_clusters_test.csv')
        test_optics_df.to_csv(test_optics_output, index=False)
        print(f"OPTICS TEST results saved: {test_optics_output}")        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print(f"Birch method identified: {n_birch} performance zones")
        print(f"OPTICS method discovered: {n_optics} natural groups")
        print(f"All results and charts saved to: {OUTPUT_PATH}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError occurred during analysis: {str(e)}")
        print("Review the error message and verify data file availability.")
        raise


if __name__ == "__main__":
    main()