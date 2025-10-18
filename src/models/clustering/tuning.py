from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import os

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available. Install with: pip install hdbscan")

# Setup output folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

output_dir = os.path.join(PROJECT_ROOT, "results", "clustering", "tuning")
os.makedirs(output_dir, exist_ok=True)

# Load dataset and perform zone-level aggregation
print("Loading data...")
data_path = os.path.join(PROJECT_ROOT, "data", "features_for_clustering.csv")

print(f"Loading data from: {data_path}")
df_raw = pd.read_csv(data_path)

if "square_id" in df_raw.columns:
    print("Aggregating by square_id for zone-level features...")
    df = df_raw.groupby("square_id").agg({
        "latitude": "mean",
        "longitude": "mean",
        "avg_latency": "mean",
        "std_latency": "mean",
        "total_throughput": "mean",
        "zone_avg_latency": "first",
        "zone_avg_upload": "first",
        "zone_avg_download": "first"
    }).reset_index()
else:
    print("No 'square_id' column found. Using raw data as-is.")
    df = df_raw.copy()

# Drop missing values
df = df.dropna()
print(f"Final dataset shape after aggregation and cleaning: {df.shape}")

features_to_use = [
    "latitude", "longitude",
    "avg_latency", "std_latency", "total_throughput",
    "zone_avg_latency", "zone_avg_upload", "zone_avg_download"
]
features_to_use = [f for f in features_to_use if f in df.columns]

X = StandardScaler().fit_transform(df[features_to_use])
print(f"Scaled {len(features_to_use)} features for clustering...\n")

# Initialise results list
results = []

# KMeans tuning
for k in tqdm(range(2, 10), desc="KMeans tuning"):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        results.append({"Model": "KMeans", "Param": f"k={k}", "Silhouette": sil, "DBI": dbi})

# DBSCAN tuning
for eps in tqdm([0.2, 0.4, 0.6, 0.8], desc="DBSCAN tuning"):
    for min_samp in [3, 5, 7]:
        model = DBSCAN(eps=eps, min_samples=min_samp)
        labels = model.fit_predict(X)
        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            results.append({"Model": "DBSCAN", "Param": f"eps={eps},min={min_samp}", "Silhouette": sil, "DBI": dbi})

# Birch tuning
for k in tqdm(range(2, 10), desc="Birch tuning"):
    for thresh in [0.3, 0.5, 0.7]:
        model = Birch(n_clusters=k, threshold=thresh)
        labels = model.fit_predict(X)
        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            results.append({"Model": "Birch", "Param": f"k={k},thr={thresh}", "Silhouette": sil, "DBI": dbi})

# OPTICS tuning
for min_samp in tqdm([3, 5, 10], desc="OPTICS tuning"):
    for xi in [0.01, 0.05, 0.1]:
        model = OPTICS(min_samples=min_samp, xi=xi)
        labels = model.fit_predict(X)
        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            results.append({"Model": "OPTICS", "Param": f"min={min_samp},xi={xi}", "Silhouette": sil, "DBI": dbi})

# HDBSCAN tuning
if HDBSCAN_AVAILABLE:
    for min_cluster_size in tqdm([5, 8, 10, 12, 15], desc="HDBSCAN tuning"):
        for min_samples in [3, 5, 8, None]:
            try:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.0,
                    cluster_selection_method="eom",
                    core_dist_n_jobs=-1
                )
                labels = model.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    # Calculate silhouette excluding noise points
                    mask = labels != -1
                    if mask.sum() > 0 and len(set(labels[mask])) > 1:
                        sil = silhouette_score(X[mask], labels[mask])
                        dbi = davies_bouldin_score(X[mask], labels[mask])
                        min_samp_str = "None" if min_samples is None else str(min_samples)
                        results.append({"Model": "HDBSCAN", "Param": f"mcs={min_cluster_size},ms={min_samp_str}", "Silhouette": sil, "DBI": dbi})
            except Exception as e:
                print(f"HDBSCAN failed with min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}")
else:
    print("Skipping HDBSCAN tuning - package not available")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Combined Score (weighted balance)
sil_norm = (results_df["Silhouette"] - results_df["Silhouette"].min()) / (
    results_df["Silhouette"].max() - results_df["Silhouette"].min()
)
dbi_norm = (results_df["DBI"].max() - results_df["DBI"]) / (
    results_df["DBI"].max() - results_df["DBI"].min()
)
results_df["CombinedScore"] = 0.7 * sil_norm + 0.3 * dbi_norm

# Save all results
all_path = os.path.join(output_dir, "hyperparameter_tuning_all.csv")
results_df.to_csv(all_path, index=False)
print(f"\nSaved full results to {all_path}")

# Display top configurations
top_sil = results_df.sort_values("Silhouette", ascending=False).head(10)
print("\nTop 10 configurations by Silhouette Score:")
print(top_sil[["Model", "Param", "Silhouette", "DBI"]])

top_dbi = results_df.sort_values("DBI", ascending=True).head(10)
print("\nTop 10 configurations by Davies–Bouldin Index:")
print(top_dbi[["Model", "Param", "Silhouette", "DBI"]])

top_combined = results_df.sort_values("CombinedScore", ascending=False).head(10)
print("\nTop 10 configurations by Combined Score (Silhouette + DBI):")
print(top_combined[["Model", "Param", "Silhouette", "DBI", "CombinedScore"]])

# Best parameters per model
def best_per_model(df, metric, ascending=False):
    return (
        df.sort_values(metric, ascending=ascending)
        .groupby("Model")
        .first()
        .reset_index()
        .sort_values(metric, ascending=not ascending)
    )

best_sil = best_per_model(results_df, "Silhouette", ascending=False)
best_dbi = best_per_model(results_df, "DBI", ascending=True)
best_combined = best_per_model(results_df, "CombinedScore", ascending=False)

# Save summaries
best_sil.to_csv(os.path.join(output_dir, "best_per_model_silhouette.csv"), index=False)
best_dbi.to_csv(os.path.join(output_dir, "best_per_model_dbi.csv"), index=False)
best_combined.to_csv(os.path.join(output_dir, "best_per_model_combined.csv"), index=False)

print("\nBest parameters per model (Silhouette):")
print(best_sil[["Model", "Param", "Silhouette", "DBI"]])

print("\nBest parameters per model (DBI):")
print(best_dbi[["Model", "Param", "Silhouette", "DBI"]])

print("\nBest parameters per model (Combined Score):")
print(best_combined[["Model", "Param", "Silhouette", "DBI", "CombinedScore"]])