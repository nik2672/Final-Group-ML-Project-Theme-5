"""
HDBSCAN Clustering for 5G Network Zones
- Loads features_for_clustering.csv
- Aggregates by square_id
- Scales features
- Small param sweep for HDBSCAN
- Prints results and saves plots/CSVs (consistent with main.py/main2.py/comparison.py)

Requires:
    pip install hdbscan
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import multiprocessing

try:
    import hdbscan
except ImportError as e:
    raise SystemExit(
        "HDBSCAN is not installed.\n"
        "Install it with: pip install hdbscan\n"
        f"Original error: {e}"
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "clustering")
os.makedirs(OUTPUT_PATH, exist_ok=True)

N_CORES = multiprocessing.cpu_count()
IS_M_SERIES = platform.system() == "Darwin" and platform.machine() == "arm64"

print("=" * 70)
print("5G NETWORK PERFORMANCE: HDBSCAN CLUSTERING")
print(f"Mode: {'M-Series' if IS_M_SERIES else 'Standard'} ({N_CORES} cores)")
print("=" * 70)


def load_and_prepare():
    print("\nLoading 5G clustering features...")
    input_path = os.path.join(DATA_PATH, "features_for_clustering.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Data file not found at {input_path}\n"
            "Please run feature engineering first: python src/features/feature_engineering.py"
        )

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} measurements with {len(df.columns)} columns")

    # Aggregate to zone-level
    if "square_id" in df.columns:
        print("Aggregating by square_id to zone-level...")
        zone_agg = df.groupby("square_id").agg({
            "latitude": "mean",
            "longitude": "mean",
            "avg_latency": "mean",
            "std_latency": "mean",
            "total_throughput": "mean",
            "zone_avg_latency": "first",
            "zone_avg_upload": "first",
            "zone_avg_download": "first"
        }).reset_index()
        print(f"Aggregated to {len(zone_agg):,} zones")
        df = zone_agg

    # Choose features 
    feature_cols = [
        "latitude", "longitude",
        "avg_latency", "std_latency", "total_throughput",
        "zone_avg_latency", "zone_avg_upload", "zone_avg_download"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Selected features: {feature_cols}")

    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Prepared matrix: {X_scaled.shape[0]} zones x {X_scaled.shape[1]} features")

    return df, X_scaled, feature_cols


def safe_silhouette(X, labels):
    """ilhouette score excluding noise -1. Returns None if not computable"""
    mask = labels != -1
    if mask.sum() <= 1:
        return None
    if len(np.unique(labels[mask])) < 2:
        return None
    try:
        return silhouette_score(X[mask], labels[mask])
    except Exception:
        return None


def count_clusters(labels):
    """Number of non-noise clusters and number of outliers."""
    n_outliers = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, n_outliers


def hdbscan_param_sweep(X):
    """
    Small grid search for HDBSCAN.
    Returns best (labels, model, stats) and a DataFrame of all trials.
    """
    print("\nHDBSCAN parameter sweep...")
    # Reasonable small grid for zones ~300
    min_cluster_sizes = [5, 8, 10, 12, 15]
    min_samples_list = [None, 3, 5, 8]

    trials = []
    best = {
        "score": -np.inf,
        "labels": None,
        "model": None,
        "params": None,
        "n_clusters": 0,
        "n_outliers": 0
    }

    total = len(min_cluster_sizes) * len(min_samples_list)
    idx = 0
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            idx += 1
            desc = f"  Trial {idx:02d}/{total}: min_cluster_size={mcs}, min_samples={ms}"
            print(desc + "...", end=" ")

            model = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=0.0,  # kept simple
                cluster_selection_method="eom",
                core_dist_n_jobs=-1  # multi-core
            )
            labels = model.fit_predict(X)
            n_clusters, n_outliers = count_clusters(labels)
            sil = safe_silhouette(X, labels)

            # Preference higher silhouette if None, treat as very low
            score = -1.0 if sil is None else sil

            # Tiebreakers (1) fewer outliers, (2) more clusters (but not degenerate)
            tie = False
            if score == best["score"]:
                tie = True

            improved = score > best["score"]
            if tie:
                if n_outliers < best["n_outliers"]:
                    improved = True
                elif n_outliers == best["n_outliers"] and n_clusters > best["n_clusters"]:
                    improved = True

            trials.append({
                "min_cluster_size": mcs,
                "min_samples": -1 if ms is None else ms,
                "silhouette_excl_noise": sil if sil is not None else np.nan,
                "n_clusters": n_clusters,
                "n_outliers": n_outliers
            })

            print(f"RM (sil excl noise)={sil:.3f}" if sil is not None else "RM (sil excl noise)=N/A",
                  f"| clusters={n_clusters}, outliers={n_outliers}")

            if improved:
                best.update({
                    "score": score,
                    "labels": labels,
                    "model": model,
                    "params": {"min_cluster_size": mcs, "min_samples": ms},
                    "n_clusters": n_clusters,
                    "n_outliers": n_outliers
                })

    trials_df = pd.DataFrame(trials)
    return best, trials_df.sort_values(
        by=["silhouette_excl_noise", "n_outliers", "n_clusters"],
        ascending=[False, True, False]
    )


# plotting & Saving 
def plot_hdbscan(X, labels, feature_names):
    print("Generating plot: hdbscan_clusters.png")
    plt.figure(figsize=(10, 8))

    # Handle noise vs clusters
    noise_mask = labels == -1
    cluster_mask = labels != -1

    # Outliers in light gray
    if np.sum(noise_mask) > 0:
        plt.scatter(X[noise_mask, 0], X[noise_mask, 1],
                    c="lightgray", alpha=0.6, s=40, label="Outliers")

    # Colored clusters
    if np.sum(cluster_mask) > 0:
        scatter = plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                              c=labels[cluster_mask], cmap="Set3", alpha=0.8, s=50)
        plt.colorbar(scatter, label="HDBSCAN Cluster")

    n_clusters, n_outliers = count_clusters(labels)
    title = f"HDBSCAN: {n_clusters} clusters, {n_outliers} outliers"
    plt.title(title, fontsize=14)
    plt.xlabel(feature_names[0] if len(feature_names) > 0 else "Feature 1")
    plt.ylabel(feature_names[1] if len(feature_names) > 1 else "Feature 2")
    plt.grid(True, alpha=0.3)
    if np.sum(noise_mask) > 0:
        plt.legend()

    out_path = os.path.join(OUTPUT_PATH, "hdbscan_clusters.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


def save_cluster_csv(df_zones, labels):
    print("Saving HDBSCAN cluster assignments...")
    out_df = df_zones.copy()
    out_df["hdbscan_cluster"] = labels
    out_file = os.path.join(OUTPUT_PATH, "hdbscan_clusters.csv")
    out_df.to_csv(out_file, index=False)
    print(f"[OK] Saved: {out_file}")


def save_trials_csv(trials_df):
    out_file = os.path.join(OUTPUT_PATH, "hdbscan_trials.csv")
    trials_df.to_csv(out_file, index=False)
    print(f"[OK] Saved grid search table: {out_file}")



def main():
    try:
        df_zones, X_scaled, feature_names = load_and_prepare()

        # Param sweep
        best, trials_df = hdbscan_param_sweep(X_scaled)
        save_trials_csv(trials_df)

        labels = best["labels"]
        n_clusters, n_outliers = best["n_clusters"], best["n_outliers"]
        sil = safe_silhouette(X_scaled, labels)

        print("\nHDBSCAN BEST CONFIGURATION")
        print("-" * 50)
        print(f"min_cluster_size : {best['params']['min_cluster_size']}")
        print(f"min_samples      : {best['params']['min_samples']}")
        print(f"clusters         : {n_clusters}")
        print(f"outliers         : {n_outliers} ({n_outliers / len(labels) * 100:.1f}%)")
        print(f"silhouette excl noise : {sil:.3f}" if sil is not None else
              "silhouette excl noise : N/A")

        # Output artifacts
        plot_hdbscan(X_scaled, labels, feature_names)
        save_cluster_csv(df_zones, labels)

        # Quick cluster distribution summary
        print("\nHDBSCAN Cluster Distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(dist)

        print("\n" + "=" * 70)
        print("[OK] HDBSCAN Clustering Complete!")
        print(f"Results saved to: {OUTPUT_PATH}")
        print("=" * 70)

    except Exception as e:
        print("\n[ERROR] HDBSCAN run failed.")
        print(str(e))
        raise


if __name__ == "__main__":
    main()
