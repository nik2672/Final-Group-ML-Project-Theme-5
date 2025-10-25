#auth: nik
# lean-ish GRU that works with the new train/test csv
"""
Before windowing:

X_tr: [T_train, n_features]

y_tr: [T_train] (after scaling, 1D)

After windowing in a batch:

xb: [batch_size, lookback=32, n_features] (float32, on device)

yb: [batch_size, 1] (float32, on device)

GRU output:

out: [batch_size, lookback, hidden=48]

last = out[:, -1, :]: [batch_size, 48]

head(last): [batch_size, 1]
"""

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# paths
_here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
data_dir = os.path.join(project_root, "data")
out_dir  = os.path.join(project_root, "results", "forecasting_gru_lean_v2")
os.makedirs(out_dir, exist_ok=True)

train_csv = os.path.join(data_dir, "features_for_forecasting_train.csv")
test_csv  = os.path.join(data_dir, "features_for_forecasting_test.csv")
engineered_csv = os.path.join(data_dir, "features_engineered.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

# knobs
seed = 42
np.random.seed(seed); torch.manual_seed(seed)

lookback = 32#how may past time step per training sample 
epochs   = 15#passes over thr training data - 1000 samples 10 times in total trained, making 10000 trainin g steps, if too many model overfit 
hidden   = 48# gru hiddnen unit (size ) - track 48 different featues or patterns simultaneous can learn to recognise different aspects of data, eahc 48 units can learn to recognizee differtn aspects of data  
n_layers = 1#gru layers (stack depth )- onelayer of processing 
dropout  = 0.1#pruen 10% of neurons 
lr       = 1e-3#learnign rate for adam 
batch0   = 128#processe 18- sampels at once during training instead of one at a time 
num_workers = 0
#KPI - key perfroamnce indicators in clustering measure how ell algo groups simialr data points together simialr to silhouette score (how tight and sperated the clusters are) . 
# kpi in forecasting - ki measur eprediction accraucy using metrics like RMSE (how far predictions are off, MAE (avreage prediction eror and MAPe ( PERCENTAGE ERORR) HELP COMPARE AND HCOOSE THE BEST MODEL 
use_only_kpis = False
max_features  = 16#MAX FEATUERS 

# utils- FEATURE PCIKIGN, CLEARS UP MESSY NAMES ADN  
def _norm_names(df: pd.DataFrame) -> pd.DataFrame:#LOWER CASE HEADERS aliases messy anmes ot clea one 
    m = {c: c.lower() for c in df.columns}
    df = df.rename(columns=m)
    ren = {
        "upload_bitrate_mbits/sec": "upload_bitrate",
        "download_bitrate_rx_mbytes": "download_bitrate",
        "avg_latency_lag1": "avg_latency_lag1",
        "avg_latency_lag_1": "avg_latency_lag1",
        "avg_latencylag1": "avg_latency_lag1",
        "upload_bitrate_mbits/sec_lag1": "upload_bitrate_lag1",
        "upload_bitrate_lag_1": "upload_bitrate_lag1",
        "download_bitrate_rx_mbytes_lag1": "download_bitrate_lag1",
        "download_bitrate_lag_1": "download_bitrate_lag1",
        "avg_latencylag_1": "avg_latency_lag1",
        "avg_latency": "avg_latency",
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def _pick_kpis(df: pd.DataFrame):#returns [avg_latency, upload_bitrate, dowload_bitrate]
    return [c for c in ["avg_latency","upload_bitrate","download_bitrate"] if c in df.columns]

def _select_feats(df: pd.DataFrame, target: str):#takes all numeric columsn except teh target optionally resitricts to kpis + lag 1 when and caps max features 
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num if c != target]
    if use_only_kpis:
        kpi = _pick_kpis(df)
        feats = [c for c in feats if c in kpi or c.endswith("_lag1")] or kpi
    if max_features and len(feats) > max_features:
        feats = feats[:max_features]
    return feats

def _downcast(df: pd.DataFrame, cols):# coorces selected columsn to a float 32 
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df

# dataset
class WinDS(Dataset):#stores c,y arrays plsu w = lookback, len : number of pssible windows = (len (x) - w)| _getitem_(i) returns the window X[i; i+ w] and the target at tiem i+w (ie predict next step )
    def __init__(self, X, y, w):
        self.X = X
        self.y = y.reshape(-1,1)
        self.w = w
    def __len__(self): 
        return max(0, len(self.X) - self.w)
    def __getitem__(self, i):
        j = i + self.w
        xw = self.X[j-self.w:j]
        return torch.from_numpy(xw), torch.from_numpy(self.y[j])

def _make_loaders_from_arrays(X_tr, y_tr, X_te, y_te):#uses the 15% train as validation | wraps each split in windDS to get windows of length lookback = 32 | builds DataLoaders (no shuffle -> preserves order) | whyy no shuffle to keep order to avoid leaking the futrue into teh past 
    n = len(X_tr)
    n_va = int(0.15 * n)
    X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
    X_tr, y_tr = X_tr[:-n_va], y_tr[:-n_va]

    ds_tr = WinDS(X_tr, y_tr, lookback)
    ds_va = WinDS(X_va, y_va, lookback)
    ds_te = WinDS(X_te, y_te, lookback)

    if min(len(ds_tr), len(ds_va), len(ds_te)) == 0:
        raise ValueError("not enough rows after windowing. reduce lookback maybe.")

    batch = min(batch0, max(16, len(ds_tr)//100))
    pin = torch.cuda.is_available()

    ld_tr = DataLoader(ds_tr, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    ld_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    ld_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return ld_tr, ld_va, ld_te

# model
#GRU: a recurrrent layer with gates tha tlearn what ot keep/forget from recent timesteps 
#MLP head: small fully ocnnnected network tha tmaps the GRU last hidden state ot the target 
class GRUReg(nn.Module):#gru process teh 32 x n_feature sequence and outputs a vector per time step- it takes the last tiem step (ut [:, -1, :]) -> feed into a small MLP head -> one scalar: 
    def __init__(self, n_features, hidden=hidden, n_layers=n_layers, dropout=dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)

# data loading for the new split
def load_split_or_fallback():
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        tr = pd.read_csv(train_csv, low_memory=False)
        te = pd.read_csv(test_csv, low_memory=False)
        tr, te = _norm_names(tr), _norm_names(te)
        return tr.reset_index(drop=True), te.reset_index(drop=True), "split"
    if not os.path.exists(engineered_csv):
        raise FileNotFoundError(f"missing: {train_csv} / {test_csv} and also {engineered_csv}")
    df = pd.read_csv(engineered_csv, low_memory=False)
    df = _norm_names(df)
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df); n_tr = int(0.8*n)
    tr, te = df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()
    return tr, te, "fallback"

def build_arrays(tr_df, te_df, target):#1. select featuers with _select_feats downcast feauters + target to flaot 32 3. fits scales on train only (xsc = standard scaler(), ys = MinMaxScaler())
    feats = _select_feats(tr_df, target)
    cols = feats + [target]
    tr_df = _downcast(tr_df, cols)
    te_df = _downcast(te_df, cols)

    xsc = StandardScaler().fit(tr_df[feats].values.astype(np.float32))
    ysc = MinMaxScaler().fit(tr_df[target].values.reshape(-1,1).astype(np.float32))

    X_tr = xsc.transform(tr_df[feats].values.astype(np.float32)).astype(np.float32)
    y_tr = ysc.transform(tr_df[target].values.reshape(-1,1).astype(np.float32)).ravel()

    X_te = xsc.transform(te_df[feats].values.astype(np.float32)).astype(np.float32)
    y_te = ysc.transform(te_df[target].values.reshape(-1,1).astype(np.float32)).ravel()
#transform trian/test with those scalers  + return arrays and the yscaler (used later to invert predicitosn to real units, plus feature count and names )
    return X_tr, y_tr, X_te, y_te, ysc, len(feats), feats
#standard scaler: features -weise (x-mean) /std 
# train/eval one target | Why best val snapshot? -> prevent saving a late pech tha toverfit; you restore the bet performing weights 
def run_one(tr_df, te_df, target):#training loop (per target) |1. build array & loaders: build_arrays -> _make_laoders_from_arrays 2. create mdoel + adam (learnign rate), MSE LOSS.3. 15 epochs  _ train phase for eah batch window xb and label yb-> compute preds & MSE -> backward with scaler -> optimiser step 4. no grad compute val MSE accross teh val loader | keep weights 
    X_tr, y_tr, X_te, y_te, ysc, n_feats, feats = build_arrays(tr_df, te_df, target)
    ld_tr, ld_va, ld_te = _make_loaders_from_arrays(X_tr, y_tr, X_te, y_te)

    model = GRUReg(n_features=n_feats).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf"); best = None
#test evalation and plotting 
    pbar = tqdm(range(1, epochs+1), desc=f"[{target}] epochs", ncols=110)
    for _ in pbar:
        model.train()
        tr_sum, tr_n = 0.0, 0
        for xb, yb in ld_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb); loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_sum += loss.item() * len(xb); tr_n += len(xb)
        tr_mse = tr_sum / max(tr_n, 1)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for xb, yb in ld_va:
                xb, yb = xb.to(device), yb.to(device)
                va_sum += loss_fn(model(xb), yb).item() * len(xb)
                va_n += len(xb)
        va_mse = va_sum / max(va_n, 1)
        pbar.set_postfix({"train_mse": f"{tr_mse:.5f}", "val_mse": f"{va_mse:.5f}"})

        if va_mse < best_val:
            best_val = va_mse
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best.items()})

    model.eval()
    preds, trues = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for xb, yb in ld_te:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
            trues.append(yb.numpy())
    y_hat = np.vstack(preds).ravel()
    y_true = np.vstack(trues).ravel()

    y_hat = ysc.inverse_transform(y_hat.reshape(-1,1)).ravel()
    y_true = ysc.inverse_transform(y_true.reshape(-1,1)).ravel()

    mae  = mean_absolute_error(y_true, y_hat)
    rmse = math.sqrt(mean_squared_error(y_true, y_hat))
    r2   = r2_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan

    safe = target.replace("/", "_")
    _plot_pred(y_true, y_hat, f"GRU-lean v2 – {target}",
               os.path.join(out_dir, f"gru_lean_v2_{safe}.png"))
    _plot_res(y_true, y_hat, target,
              os.path.join(out_dir, f"gru_lean_v2_{safe}_residuals.png"))

    return {
        "target": target, "model": "GRU-lean-v2",
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Seq": lookback, "Epochs": epochs, "Batch": ld_tr.batch_size,
        "n_features": n_feats
    }

def _plot_pred(y_true, y_pred, title, outp):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label="actual", alpha=0.85)
    plt.plot(y_pred, label="pred", linestyle="--")
    plt.title(title); plt.xlabel("time"); plt.ylabel("value")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(outp, dpi=150); plt.close()

def _plot_res(y_true, y_pred, label, outp):
    r = y_true - y_pred
    plt.figure(figsize=(12,4))
    plt.plot(r, alpha=0.85)
    plt.title(f"residuals – {label}"); plt.xlabel("time"); plt.ylabel("error")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(outp, dpi=150); plt.close()

def main():
    print(f"GRU-lean v2 | device={device.type} | out={out_dir}")
    tr_df, te_df, mode = load_split_or_fallback()
    print(f"data mode: {mode}")

    targets = [c for c in ["avg_latency","upload_bitrate","download_bitrate"] if c in tr_df.columns]
    if not targets:
        raise ValueError("no KPI targets found in train csv")

    results = []
    for t in targets:
        print(f"\n--- {t} ---")
        res = run_one(tr_df, te_df, t)
        print(f"  test: mae={res['MAE']:.3f} rmse={res['RMSE']:.3f} r2={res['R2']:.3f}")
        results.append(res)

    out_csv = os.path.join(out_dir, "model_comparison_gru_lean_v2.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nsaved summary: {out_csv}")
    print(f"plots dir: {out_dir}")

if __name__ == "__main__":
    main()
