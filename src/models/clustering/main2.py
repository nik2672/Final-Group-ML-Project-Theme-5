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
    Load the 5G network performance data from the specified data file.
    
    Returns:
        pandas.DataFrame: Network performance measurements with location and quality metrics
    """
    print("\nLoading 5G network performance data...")
    
    # Define data file location
    input_path = os.path.join(DATA_PATH, 'features_for_clustering.csv')

    # Check if the data file actually exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Data file not found at {input_path}. "
            "Please run feature engineering first: python src/features/feature_engineering.py"
        )

    # Load the data 
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Successfully loaded {len(df):,} network measurements with {len(df.columns)} different metrics")
    return df


def prepare_features(df):
    """
    Prepare the data for analysis by selecting relevant features and standardizing their scales.
    
    Performs feature selection and normalization to ensure comparable measurement scales
    across different network performance metrics for effective clustering analysis.
    
    Args:
        df (pandas.DataFrame): Raw network performance data
        
    Returns:
        tuple: Standardized features, feature names, scaler object, processed dataframe
    """
    print("\nPreparing data for clustering analysis...")
    
    # Zone-level aggregation (consistent with team approach)
    # This reduces 2.4M samples to aggregated zones for efficient processing
    if 'square_id' in df.columns:
        print(f"Aggregating {len(df):,} measurements by zone (square_id)...")
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
        
        print(f"Reduced to {len(zone_agg):,} zones (much faster processing!)")
        df = zone_agg
    
    # Use consistent features for team analysis compatibility
    feature_cols = [
        'latitude', 'longitude',
        'avg_latency', 'std_latency', 'total_throughput',
        'zone_avg_latency', 'zone_avg_upload', 'zone_avg_download'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    # If expected measurements are missing, use available numeric data
    if len(feature_cols) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols[:8]  # Take the first 8 numeric measurements
        
    print(f"Selected features for clustering: {feature_cols}")
    
    # Extract the selected data and fill in any missing values with zeros
    X = df[feature_cols].fillna(0)
    
    # Standardize the data (normalize all measurements for comparison)
    # Converts all metrics to comparable scales for effective clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Prepared data matrix: {X_scaled.shape[0]:,} locations with {X_scaled.shape[1]} measurements each")
    return X_scaled, feature_cols, scaler, df


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
    
    # Set up the OPTICS algorithm with settings optimized for zone-level data
    optics = OPTICS(
        min_samples=5,           # Reduced for smaller zone dataset (was 10)
        max_eps=2.0,             # Increased for zone distances (was 0.5) 
        cluster_method='xi',     # The method for deciding where one group ends and another begins
        xi=0.1,                  # Relaxed boundary detection (was 0.05)
        n_jobs=-1                # Use all available CPU cores for faster processing
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
        # Step 1: Load the 5G network measurement data
        print("\n=== STEP 1: LOADING DATA ===")
        df = load_clustering_data()

        # Step 2: Prepare the data for analysis
        print("\n=== STEP 2: PREPARING DATA ===")
        X_scaled, feature_names, scaler, df_processed = prepare_features(df)

        # Step 3: Use Birch algorithm to find performance zones
        print("\n=== STEP 3: BIRCH ANALYSIS ===")
        birch_labels, birch_model, n_birch = run_birch_clustering(X_scaled)

        # Step 4: Use OPTICS algorithm to find natural groupings
        print("\n=== STEP 4: OPTICS ANALYSIS ===")
        optics_labels, optics_model, n_optics = run_optics_clustering(X_scaled)

        # Step 5: Create individual visualizations
        print("\n=== STEP 5: CREATING VISUALIZATIONS ===")
        plot_birch_results(X_scaled, birch_labels, feature_names)
        plot_optics_results(X_scaled, optics_labels, feature_names)
        
        # Step 6: Save results to separate files
        print("\n=== STEP 6: SAVING RESULTS ===")
        save_birch_results(df_processed, birch_labels)
        save_optics_results(df_processed, optics_labels)        # Step 7: Show final summary
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