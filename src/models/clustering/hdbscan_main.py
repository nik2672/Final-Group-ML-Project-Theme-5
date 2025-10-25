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

def agg_by_square(df):#lots of rows for teh same zone (square_id) hence we group rows by (square_id) sumamrise them into one row per zone (take the mean latency, mean throguhput etc)
    if "square_id" not in df.columns:
        return df.copy()#identify if square_id is there or not 
    cols = {#how to sqaush many rows into one row for each square_id: latiude:mean, logitude: mean | avg_latency: mean, std_latency: mean, total_throghput: mean | zone_avg_latency: first, zong_avg_upload: first, zone_avg_downlaod: first (just picking the first alue seen in group as it is assumed to be similar for all rows in that zone (hene simplifying data)
        "latitude": "mean","longitude": "mean",
        "avg_latency": "mean","std_latency": "mean","total_throughput": "mean",
        "zone_avg_latency": "first","zone_avg_upload": "first","zone_avg_download": "first",
    }#
    cols = {k:v for k,v in cols.items() if k in df.columns}#keeping only entries for columns that ACTUALLY exist in the dataframe (dosnt cause errors)
    return df.groupby("square_id").agg(cols).reset_index()#result: one row per zone with the agrgeagated featuers needed for clustering | group all rows by 'square id' turn group labels back into a nroaml column with reset_index() 

def pick_feature_cols(df):#says which columns to use as features after aggregation - create list of columsn to feed into teh model (mean data under the original name)
    base = ["latitude","longitude","avg_latency","std_latency","total_throughput",
            "zone_avg_latency","zone_avg_upload","zone_avg_download"]
    feats = [c for c in base if c in df.columns]#to prevent erorrs (columns missing) buidling the actual LIST of FEATURES keeping only those COLLLUMNS 
    if not feats:
        raise ValueError("no ffeature cols found")#raise error if feature column ins empty 
    return feats#return final list of feautres names to use for clsutering 

def safe_sil(X, labels):#helper that will compute silhoutte score 
    m = labels != -1#it is -1 becuase clser IDs are (0,1,2) -1 indicates 'noise (point didnt belong to any cluster) | boolean mask that keeps only real cluster points (true) and drops noise (false) because noise would mess up score (outleirs) 
    if m.sum() <= 1 or len(np.unique(labels[m])) < 2:#after removign noise (-1) there are 0 or 1 points left you cant judge "cluster quality" with <1 so stop | <2 silhouette compares between clusers, so with fewer than 2 clusters its not defined
        return None#so stop no score keep going 
    try:
        return silhouette_score(X[m], labels[m])#try compute silhoette score only on the NON noise points and return that number 
    except Exception:
        return None#if there is NOT ENOGH POINTS weird geometry catch it and return nothign instead of crashing

def count_clusters(labels):#helper COUNTS HOW MANY CLUSTERS and hwo many NOISE POINTS ()
    n_out = int(np.sum(labels == -1))#count how many outleris/noisepoints  (-1)label
    k = len(set(labels)) - (1 if -1 in labels else 0)#set(labels) - set all unique labels (-1,0,1) |Len () - how many unique labels are there | subtract 1 if -1 is present because -1 is not a real cluster | k=results -> number of actaul clusters (0,1,2) excluding noise
    return k, n_out#return number of (number_of clusters, number_of_noise_points)
    #example: labels = [-1,1,1,0,0] -> n_out = 1 (one-1) | unique labels {-1,0,1}=  unique minus 1 for -1 -> k= 2 clusters returns -> {2,1,}


def param_sweep(X):
    mcs_list, ms_list = [5,8,10,12,15], [None,3,5,8]#PERAMETER LIST: min_cluster_size & min_samples if |NONE meaning default to HDBSCANE default 
    trials = []#empty list -> store resullts of each perameter 
    best = {"score": -1e9, "labels": None, "model": None,
            "params": None, "n_clusters": 0, "n_outliers": 0}#set bad scores so first real run will be autimatically better & will fill in once a better run is made
    for mcs in mcs_list:
        for ms in ms_list:#loop over each combo of MIN_CLUSTER_SIZE, MIN_SAMPLES 
            model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                    cluster_selection_method="eom", core_dist_n_jobs=-1)#BUILD HDBSCAN with PERMETER COMBO: Cluster_selection_method ="eom" robust way HDBSCAN choosen stable cluster|core_dist_n_jobs = -1 use all cpu cores 
            labels = model.fit_predict(X)
            k, out = count_clusters(labels)
            sil = safe_sil(X, labels)
            score = -1.0 if sil is None else sil#train HDBSCAN and get each lablle for each row (X), 0,1,2 cluster ID | -1 (outlier)| k= how many real clusters (ignore -1)|out = how many outliers |sil = compute silhouete score (how well-seprated the clusters are excluding outliers )| score = make a SINGLE score for comparing each trail with -1 as penalty so the rank ranks worse than a valid one 
            trials.append({#log trial for the CSV 
                "min_cluster_size": mcs,
                "min_samples": -1 if ms is None else ms,
                "silhouette_excl_noise": np.nan if sil is None else sil,
                "n_clusters": k, "n_outliers": out
            })
            better = (score > best["score"] or#decide if trail is BETTER than the current BEST (HGIHER SILHOUETTE wins| tie silhourette looks at 1. fewer outliers win or more clusters wins )
                      (np.isclose(score,best["score"]) and (out < best["n_outliers"] or
                       (out == best["n_outliers"] and k > best["n_clusters"]))))
            if better:#if thi scombo is better then REMMEBER modle (labels, perams and stats)
                best.update({"score": score, "labels": labels, "model": model,
                             "params": {"min_cluster_size": mcs, "min_samples": ms},
                             "n_clusters": k, "n_outliers": out})
    trials = pd.DataFrame(trials).sort_values(#convert to dataframe SORT for easy reading (Highest silhouette first -> hten fewest outliers -> then more clusters )
        ["silhouette_excl_noise","n_outliers","n_clusters"], ascending=[False,True,False]
    )
    return best, trials#best = winning modle + metadata -> trails -> the full leaderbaord table 

def pca2d(X):#center the data: subtrat the collumn MEAN from each collumn so every features has a MEAN of roughly zero (PA needes centred data) 
    Xc = X - X.mean(0, keepdims=True)
    _,_,vt = np.linalg.svd(Xc, full_matrices=False)#run SVD on centred data (vt hlds priniple direction (each row is a direction of maximum ariance ))
    return Xc @ vt[:2].T
#DRAW A 2D SCATTER OF PCA POINTS, show clsuter in coloru, noise ingray annotate with counts 
def plot_2d(X2, labels, fname, note):
    plt.figure(figsize=(10,8))#scatter plot from 2D data and their clsuter labells| start resonably large figure (10,8) | make mask for noise points (-1)  
    noise = labels == -1#make mask (array liek true, flase, true, true being outlier false being other) for noise points 
    if noise.any():#any noise points plot a LIGHT GRAY 
        plt.scatter(X2[noise,0], X2[noise,1], c="lightgray", s=35, alpha=0.6, label="noise")
    m = ~noise#mask for NON noise points (actual clusters) 
    if m.any():#plot the CLUSTERED POINTS cooruign them 
        sc = plt.scatter(X2[m,0], X2[m,1], c=labels[m], cmap="Set3", s=45, alpha=0.9)
        plt.colorbar(sc, label="cluster")
    k,out = count_clusters(labels)#K=how many clusters and OUTLIERS (out)
    plt.title(f"hdbscan {note} | clusters={k} outliers={out}")#NOTE wheahther tain or tets disply counts 
    plt.grid(alpha=.3); plt.tight_layout()#add faint grid 
    path = os.path.join(out_dir, fname)#store PLOTS in RESULTS folder
    plt.savefig(path, dpi=150); plt.close()
    print("[saved]", path)

def assign_test_by_knn(Xtr, lab_tr, Xte, n_neighbors=5):#labels TEST POINTS by looking at the nearest labelled TRAIN points IGNROING outleir (-1) | Xtr: train features (scaled) -> lab_tr: train labels from HDBSCAN (e.g 0,1, or -1 noise ) -> Xte: test features (scaled) | n_eighbours: how many enarest train points to use (5) 
    m = lab_tr != -1#mask to keep custers only remove neighbors that consider (-1) outliers 
    if m.sum() == 0:#if all trian points were noise we ant assign anythign return -1 for every test point 
        return np.full(len(Xte), -1, dtype=int)#^
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, m.sum())).fit(Xtr[m])#fit a K nearest neighbors model only on clusters only train points, if there are fewer clean train points than n_neighbors, reduce that K value accordingly 
    idx = nn.kneighbors(Xte, return_distance=False)#for each test point find idices of its K nearest lcean train points 
    out = []
    pool = lab_tr[m]#prepare a output list:pool is the array of train labels for the clusters only train points 
    for row in pool[idx]:#for each test point look up the labels of its K nearest neigborurs (row)|ignore any -1 | take majority label among the neightbours (cluster id with highest ocunt) & if there are no valid points then identify -1 
        vals, cnt = np.unique(row[row!=-1], return_counts=True)
        out.append(int(vals[np.argmax(cnt)]) if len(vals) else -1)
    return np.array(out, dtype=int)#return all test albels as a numPy array 

def main():
    tr_path, te_path, mode = find_paths()#locate train/test CSVS print in results folder 
    print("paths:", mode)
    print(" train:", tr_path)
    print(" test :", te_path)
    print(" save ->", out_dir)

    df_tr_raw, df_te_raw = read_csv(tr_path), read_csv(te_path)#read sv aggregate by square_id -> one row per zone (means/firsts)
    df_tr, df_te = agg_by_square(df_tr_raw), agg_by_square(df_te_raw)
    print("zones: train=", len(df_tr), " test=", len(df_te))#pint how many zones are in train/test

    feats = pick_feature_cols(df_tr)#choose which columsn to use as features (extrat those oclumsn to addays and fill missing values with 0 )
    Xtr = df_tr[feats].fillna(0).values
    Xte = df_te[feats].fillna(0).values
#STANDARD SCALAR Standard Scalar on train apply to train and test such that there is 0 variae | try as many HDBSCNA setting and pcik the best 
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
#RUN a perameter sweep (min_cluster_size) / min_samples -> keep the best modle (by silhouette; tie breaks by fewer outliers, then more clusters) SAVE CSV of all trails 
    best, trials = param_sweep(Xtr_s)
    trials_path = os.path.join(out_dir, "hdbscan_trials.csv")
    trials.to_csv(trials_path, index=False); print("[saved]", trials_path)
#grab CLuster labels on train, compute the silhouette (excl noise) and print PERAMS/ CLUSTERS OUTLIERS 
    labels_tr = best["labels"]
    sil = safe_sil(Xtr_s, labels_tr)
    print("best params:", best["params"], "| clusters:", best["n_clusters"],
          "outliers:", best["n_outliers"], "| sil(excl noise):", "N/A" if sil is None else f"{sil:.3f}")
#make a 2D picture fo train clusters and save them | do PCA 2d projection (just for visualization) | plot: clusters in coloru, noise in grey save PNG  
    Xtr_2 = pca2d(Xtr_s)
    plot_2d(Xtr_2, labels_tr, "hdbscan_clusters_train.png", "train")
#save the train labels next to the zone table | add HDBSAN_CLUSTER columns to teh zone table and save a CSV 
    out_tr = df_tr.copy(); out_tr["hdbscan_cluster"] = labels_tr
    tr_csv = os.path.join(out_dir, "hdbscan_clusters_train.csv")
    out_tr.to_csv(tr_csv, index=False); print("saved", tr_csv)
#for each test zone, find the NEAREST train zones (ignoring oultier) and copy teh amjority cluster ID | make a test plot PCA 
    labels_te = assign_test_by_knn(Xtr_s, labels_tr, Xte_s, 5)
    Xte_2 = pca2d(Xte_s)
    plot_2d(Xte_2, labels_te, "hdbscan_clusters_test.png", "test(knn)")
#save the test labels with teh zone table 
    out_te = df_te.copy(); out_te["hdbscan_cluster"] = labels_te
    te_csv = os.path.join(out_dir, "hdbscan_clusters_test.csv")
    out_te.to_csv(te_csv, index=False); print("[saved]", te_csv)
#print label counts (train & test)| quci histogram: how amny in each cluser and how many -1 noise
    u,c = np.unique(labels_tr, return_counts=True); print("train dist:", dict(zip(map(int,u), map(int,c))))
    u2,c2 = np.unique(labels_te, return_counts=True); print("test  dist:", dict(zip(map(int,u2), map(int,c2))))
#entry run main when executign script 
if __name__ == "__main__":
    main()