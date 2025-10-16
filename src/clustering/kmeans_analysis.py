# src/clustering/eda_kmeans_analysis.py
# Author: Finn Porter
# Purpose: Exploratory K-Means clustering analysis on 5G network data (EDA stage)
# Note: Runs K-Means for multiple k values (2–7) on a 25% random sample to generate Elbow and Silhouette plots.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_data_with_imputation.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Compute basic descriptive metrics
df["avg_latency"] = df[["svr1", "svr2", "svr3", "svr4"]].mean(axis=1)
df["std_latency"] = df[["svr1", "svr2", "svr3", "svr4"]].std(axis=1)
df["total_throughput"] = df["upload_bitrate_mbits/sec"] + df["download_bitrate_rx_mbits/sec"]

# Select features for clustering
features = [
    "latitude", "longitude",
    "avg_latency", "std_latency",
    "total_throughput", "upload_bitrate_mbits/sec", "download_bitrate_rx_mbits/sec"
]

# Ensure features exist
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in dataset: {missing}")

df_features = df[features].dropna()
print(f"Using {len(features)} features for EDA clustering: {features}")
print(f"Full dataset shape before sampling: {df_features.shape}")

# Use a 25% random sample for faster analysis
df_sample = df_features.sample(frac=0.25, random_state=42)
print(f"Using 25% random sample: {df_sample.shape[0]} rows")

# Scale features
print(f"Scaling features")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)

# Explore different k values (Elbow + Silhouette)
K = range(2, 8)
results = []

print("\nRunning K-Means for k = 2–7 ...")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    results.append({"k": k, "inertia": inertia, "silhouette": silhouette})
    print(f"  k={k}: inertia={inertia:.2f}, silhouette={silhouette:.3f}")

# Save results table
results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, "kmeans_scores.csv")
results_df.to_csv(results_path, index=False)
print(f"\nSaved score table to: {results_path}")

# Elbow plot
plt.figure(figsize=(8, 4))
plt.plot(results_df["k"], results_df["inertia"], "o-", color="blue")
plt.title("Elbow Method (EDA Sampled 25%)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "elbow_plot_eda_sampled.png"), dpi=300)
plt.close()

# Silhouette plot
plt.figure(figsize=(8, 4))
plt.plot(results_df["k"], results_df["silhouette"], "o-", color="green")
plt.title("Silhouette Scores vs k (EDA Sampled 25%)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "silhouette_plot_eda_sampled.png"), dpi=300)
plt.close()

print(f"Elbow and Silhouette plots saved in: {RESULTS_DIR}")