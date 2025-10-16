import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_data_with_imputation.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df["avg_latency"] = df[["svr1", "svr2", "svr3", "svr4"]].mean(axis=1)
df["std_latency"] = df[["svr1", "svr2", "svr3", "svr4"]].std(axis=1)
df["total_throughput"] = df["upload_bitrate_mbits/sec"] + df["download_bitrate_rx_mbits/sec"]

features = [
    "latitude", "longitude",
    "avg_latency", "std_latency",
    "total_throughput",
    "upload_bitrate_mbits/sec",
    "download_bitrate_rx_mbits/sec"
]

df_features = df[features].dropna()
print(f"Filtered dataset shape for clustering: {df_features.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Run K-Means
optimal_k = 4
print(f"Running final K-Means with k={optimal_k} ...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_features["cluster"] = kmeans.fit_predict(X_scaled)

# Merge cluster labels back into main dataframe
df = df.merge(df_features[["latitude", "longitude", "cluster"]], on=["latitude", "longitude"], how="left")

# Save results
summary = df_features.groupby("cluster")[features].mean().round(2)
summary_path = os.path.join(RESULTS_DIR, f"eda_cluster_summary_k{optimal_k}.csv")
summary.to_csv(summary_path, index=True)

plt.figure(figsize=(8,6))
plt.scatter(df_features["longitude"], df_features["latitude"], c=df_features["cluster"], cmap="tab10", s=10)
plt.title(f"K-Means Clusters (k={optimal_k}) by Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, f"cluster_map_k{optimal_k}.png"), dpi=300)
plt.close()

print(f"Cluster summary and map saved to {RESULTS_DIR}")