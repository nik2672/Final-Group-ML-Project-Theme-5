# nik:hdbscan 
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

try:
    import hdbscan
except Exception as e:
    raise SystemExit("please: pip install hdbscan\n" + str(e))

here = os.getcwd()
out_dir = os.path.join(here, "results", "clustering_hdbscan")
os.makedirs(out_dir, exist_ok=True)

def find_paths():
    env_tr = os.environ.get("HDB_TRAIN_CSV", "").strip()
    env_te = os.environ.get("HDB_TEST_CSV", "").strip()
    if env_tr and env_te and os.path.exists(env_tr) and os.path.exists(env_te):
        return env_tr, env_te, "env"

    drive_root = "/content/drive/MyDrive"
    tr_drive = os.path.join(drive_root, "DATA-NEW", "features_for_clustering_train.csv")
    te_drive = os.path.join(drive_root, "DATA-NEW", "features_for_clustering_test.csv")
    if os.path.exists(tr_drive) and os.path.exists(te_drive):
        return tr_drive, te_drive, "drive/DATA-NEW"

    # Try improved features first (leakage-safe pipeline)
    proj_data = os.path.join(here, "data")
    tr_repo_improved = os.path.join(proj_data, "features_for_clustering_train_improved.csv")
    te_repo_improved = os.path.join(proj_data, "features_for_clustering_test_improved.csv")
    if os.path.exists(tr_repo_improved) and os.path.exists(te_repo_improved):
        return tr_repo_improved, te_repo_improved, "repo/data (improved)"
    
    # Fall back to old naming convention
    tr_repo = os.path.join(proj_data, "features_for_clustering_train.csv")
    te_repo = os.path.join(proj_data, "features_for_clustering_test.csv")
    if os.path.exists(tr_repo) and os.path.exists(te_repo):
        return tr_repo, te_repo, "repo/data"

    raise FileNotFoundError("cant find clustering train/test csvs. set HDB_* env or put in drive/DATA-NEW or run: python src/features/leakage_safe_feature_engineering.py")

def read_csv(p):
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def agg_by_square(df):
    if "square_id" not in df.columns:
        return df.copy()
    cols = {
        "latitude": "mean","longitude": "mean",
        "avg_latency": "mean","std_latency": "mean","total_throughput": "mean",
        "zone_avg_latency": "first","zone_avg_upload": "first","zone_avg_download": "first",
    }
    cols = {k:v for k,v in cols.items() if k in df.columns}
    return df.groupby("square_id").agg(cols).reset_index()

def pick_feature_cols(df):
    base = ["latitude","longitude","avg_latency","std_latency","total_throughput",
            "zone_avg_latency","zone_avg_upload","zone_avg_download"]
    feats = [c for c in base if c in df.columns]
    if not feats:
        raise ValueError("no usable feature cols found")
    return feats

def safe_sil(X, labels):
    m = labels != -1
    if m.sum() <= 1 or len(np.unique(labels[m])) < 2:
        return None
    try:
        return silhouette_score(X[m], labels[m])
    except Exception:
        return None

def count_clusters(labels):
    n_out = int(np.sum(labels == -1))
    k = len(set(labels)) - (1 if -1 in labels else 0)
    return k, n_out

def param_sweep(X):
    mcs_list, ms_list = [5,8,10,12,15], [None,3,5,8]
    trials = []
    best = {"score": -1e9, "labels": None, "model": None,
            "params": None, "n_clusters": 0, "n_outliers": 0}
    for mcs in mcs_list:
        for ms in ms_list:
            model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                    cluster_selection_method="eom", core_dist_n_jobs=-1)
            labels = model.fit_predict(X)
            k, out = count_clusters(labels)
            sil = safe_sil(X, labels)
            score = -1.0 if sil is None else sil
            trials.append({
                "min_cluster_size": mcs,
                "min_samples": -1 if ms is None else ms,
                "silhouette_excl_noise": np.nan if sil is None else sil,
                "n_clusters": k, "n_outliers": out
            })
            better = (score > best["score"] or
                      (np.isclose(score,best["score"]) and (out < best["n_outliers"] or
                       (out == best["n_outliers"] and k > best["n_clusters"]))))
            if better:
                best.update({"score": score, "labels": labels, "model": model,
                             "params": {"min_cluster_size": mcs, "min_samples": ms},
                             "n_clusters": k, "n_outliers": out})
    trials = pd.DataFrame(trials).sort_values(
        ["silhouette_excl_noise","n_outliers","n_clusters"], ascending=[False,True,False]
    )
    return best, trials

def pca2d(X):
    Xc = X - X.mean(0, keepdims=True)
    _,_,vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T

def plot_2d(X2, labels, fname, note):
    plt.figure(figsize=(10,8))
    noise = labels == -1
    if noise.any():
        plt.scatter(X2[noise,0], X2[noise,1], c="lightgray", s=35, alpha=0.6, label="noise")
    m = ~noise
    if m.any():
        sc = plt.scatter(X2[m,0], X2[m,1], c=labels[m], cmap="Set3", s=45, alpha=0.9)
        plt.colorbar(sc, label="cluster")
    k,out = count_clusters(labels)
    plt.title(f"hdbscan {note} | clusters={k} outliers={out}")
    plt.grid(alpha=.3); plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150); plt.close()
    print("[saved]", path)

def assign_test_by_knn(Xtr, lab_tr, Xte, n_neighbors=5):
    m = lab_tr != -1
    if m.sum() == 0:
        return np.full(len(Xte), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, m.sum())).fit(Xtr[m])
    idx = nn.kneighbors(Xte, return_distance=False)
    out = []
    pool = lab_tr[m]
    for row in pool[idx]:
        vals, cnt = np.unique(row[row!=-1], return_counts=True)
        out.append(int(vals[np.argmax(cnt)]) if len(vals) else -1)
    return np.array(out, dtype=int)

def main():
    tr_path, te_path, mode = find_paths()
    print("paths:", mode)
    print(" train:", tr_path)
    print(" test :", te_path)
    print(" save ->", out_dir)

    df_tr_raw, df_te_raw = read_csv(tr_path), read_csv(te_path)
    df_tr, df_te = agg_by_square(df_tr_raw), agg_by_square(df_te_raw)
    print("zones: train=", len(df_tr), " test=", len(df_te))

    feats = pick_feature_cols(df_tr)
    Xtr = df_tr[feats].fillna(0).values
    Xte = df_te[feats].fillna(0).values

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    best, trials = param_sweep(Xtr_s)
    trials_path = os.path.join(out_dir, "hdbscan_trials.csv")
    trials.to_csv(trials_path, index=False); print("[saved]", trials_path)

    labels_tr = best["labels"]
    sil = safe_sil(Xtr_s, labels_tr)
    print("best params:", best["params"], "| clusters:", best["n_clusters"],
          "outliers:", best["n_outliers"], "| sil(excl noise):", "N/A" if sil is None else f"{sil:.3f}")

    Xtr_2 = pca2d(Xtr_s)
    plot_2d(Xtr_2, labels_tr, "hdbscan_clusters_train.png", "train")

    out_tr = df_tr.copy(); out_tr["hdbscan_cluster"] = labels_tr
    tr_csv = os.path.join(out_dir, "hdbscan_clusters_train.csv")
    out_tr.to_csv(tr_csv, index=False); print("[saved]", tr_csv)

    labels_te = assign_test_by_knn(Xtr_s, labels_tr, Xte_s, 5)
    Xte_2 = pca2d(Xte_s)
    plot_2d(Xte_2, labels_te, "hdbscan_clusters_test.png", "test(knn)")

    out_te = df_te.copy(); out_te["hdbscan_cluster"] = labels_te
    te_csv = os.path.join(out_dir, "hdbscan_clusters_test.csv")
    out_te.to_csv(te_csv, index=False); print("[saved]", te_csv)

    u,c = np.unique(labels_tr, return_counts=True); print("train dist:", dict(zip(map(int,u), map(int,c))))
    u2,c2 = np.unique(labels_te, return_counts=True); print("test  dist:", dict(zip(map(int,u2), map(int,c2))))

if __name__ == "__main__":
    main()