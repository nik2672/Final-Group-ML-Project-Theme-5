# -*- coding: utf-8 -*-
# auth: nik
# gru-lean v2 that reads new train/test csvs and saves to /content/results in colab

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

IN_COLAB = "/content" in os.getcwd() or "COLAB_GPU" in os.environ
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# where outputs go
OUT_DIR = "/content/results/forecasting_gru_lean_v2" if IN_COLAB \
          else os.path.join(PROJECT_ROOT, "results", "forecasting_gru_lean_v2")
os.makedirs(OUT_DIR, exist_ok=True)

def _find_split_csvs():
    """look for train/test csvs in env, repo data/, or Drive DATA-NEW"""
    tr_env = os.environ.get("GRU_TRAIN_CSV", "").strip()
    te_env = os.environ.get("GRU_TEST_CSV", "").strip()
    if tr_env and te_env and os.path.exists(tr_env) and os.path.exists(te_env):
        return tr_env, te_env, "env"

    tr_repo = os.path.join(PROJECT_ROOT, "data", "features_for_forecasting_train.csv")
    te_repo = os.path.join(PROJECT_ROOT, "data", "features_for_forecasting_test.csv")
    if os.path.exists(tr_repo) and os.path.exists(te_repo):
        return tr_repo, te_repo, "repo-data"

    tr_drive = "/content/drive/MyDrive/DATA-NEW/features_for_forecasting_train.csv"
    te_drive = "/content/drive/MyDrive/DATA-NEW/features_for_forecasting_test.csv"
    if os.path.exists(tr_drive) and os.path.exists(te_drive):
        return tr_drive, te_drive, "drive-data-new"

    return None, None, None

ENGINEERED_CSV = os.path.join(PROJECT_ROOT, "data", "features_engineered.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

seed = 42
np.random.seed(seed); torch.manual_seed(seed)

lookback = 32
epochs   = 15
hidden   = 48
n_layers = 1
dropout  = 0.1
lr       = 1e-3
batch0   = 128
num_workers = 0

use_only_kpis = False
max_features  = 16

def _norm_names(df: pd.DataFrame) -> pd.DataFrame:
    """normalize column names and fix common variants"""
    m = {c: c.lower() for c in df.columns}
    df = df.rename(columns=m)
    ren = {
        "upload_bitrate_mbits/sec": "upload_bitrate",
        "download_bitrate_rx_mbytes": "download_bitrate",
        "avg_latency_lag_1": "avg_latency_lag1",
        "upload_bitrate_mbits/sec_lag1": "upload_bitrate_lag1",
        "upload_bitrate_lag_1": "upload_bitrate_lag1",
        "download_bitrate_rx_mbytes_lag1": "download_bitrate_lag1",
        "download_bitrate_lag_1": "download_bitrate_lag1",
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def _pick_kpis(df: pd.DataFrame):
    return [c for c in ["avg_latency", "upload_bitrate", "download_bitrate"] if c in df.columns]

def _select_feats(df: pd.DataFrame, target: str):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num if c != target]
    if use_only_kpis:
        kpi = _pick_kpis(df)
        feats = [c for c in feats if c in kpi or c.endswith("_lag1")] or kpi
    if max_features and len(feats) > max_features:
        feats = feats[:max_features]
    return feats

def _downcast(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df

class WinDS(Dataset):
    """windowed dataset that slices on the fly"""
    def __init__(self, X, y, w):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.w = w
    def __len__(self):
        return max(0, len(self.X) - self.w)
    def __getitem__(self, i):
        j = i + self.w
        xw = self.X[j - self.w : j]
        return torch.from_numpy(xw), torch.from_numpy(self.y[j])

def _make_loaders_from_arrays(X_tr, y_tr, X_te, y_te):
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

class GRUReg(nn.Module):
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

def load_split_or_fallback():
    """use provided train/test split if found, else split features_engineered.csv chronologically"""
    tr_csv, te_csv, mode = _find_split_csvs()
    if tr_csv and te_csv:
        tr = pd.read_csv(tr_csv, low_memory=False)
        te = pd.read_csv(te_csv, low_memory=False)
        tr, te = _norm_names(tr), _norm_names(te)
        return tr.reset_index(drop=True), te.reset_index(drop=True), mode
    if not os.path.exists(ENGINEERED_CSV):
        raise FileNotFoundError("cant find train/test csvs or features_engineered.csv")
    df = pd.read_csv(ENGINEERED_CSV, low_memory=False)
    df = _norm_names(df)
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df); n_tr = int(0.8 * n)
    tr, te = df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()
    return tr, te, "fallback"

def build_arrays(tr_df, te_df, target):
    feats = _select_feats(tr_df, target)
    cols = feats + [target]
    tr_df = _downcast(tr_df, cols)
    te_df = _downcast(te_df, cols)

    xsc = StandardScaler().fit(tr_df[feats].values.astype(np.float32))
    ysc = MinMaxScaler().fit(tr_df[target].values.reshape(-1, 1).astype(np.float32))

    X_tr = xsc.transform(tr_df[feats].values.astype(np.float32)).astype(np.float32)
    y_tr = ysc.transform(tr_df[target].values.reshape(-1, 1).astype(np.float32)).ravel()

    X_te = xsc.transform(te_df[feats].values.astype(np.float32)).astype(np.float32)
    y_te = ysc.transform(te_df[target].values.reshape(-1, 1).astype(np.float32)).ravel()

    return X_tr, y_tr, X_te, y_te, ysc, len(feats), feats

def run_one(tr_df, te_df, target):
    X_tr, y_tr, X_te, y_te, ysc, n_feats, feats = build_arrays(tr_df, te_df, target)
    ld_tr, ld_va, ld_te = _make_loaders_from_arrays(X_tr, y_tr, X_te, y_te)

    model = GRUReg(n_features=n_feats).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf"); best = None

    pbar = tqdm(range(1, epochs + 1), desc=f"[{target}] epochs", ncols=110)
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

    y_hat = ysc.inverse_transform(y_hat.reshape(-1, 1)).ravel()
    y_true = ysc.inverse_transform(y_true.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_hat)
    rmse = math.sqrt(mean_squared_error(y_true, y_hat))
    r2   = r2_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan

    safe = target.replace("/", "_")
    _plot_pred(y_true, y_hat, f"gru-lean v2 – {target}",
               os.path.join(OUT_DIR, f"gru_lean_v2_{safe}.png"))
    _plot_res(y_true, y_hat, target,
              os.path.join(OUT_DIR, f"gru_lean_v2_{safe}_residuals.png"))

    return {
        "target": target, "model": "GRU-lean-v2",
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Seq": lookback, "Epochs": epochs, "Batch": ld_tr.batch_size,
        "n_features": n_feats
    }

def _plot_pred(y_true, y_pred, title, outp):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="actual", alpha=0.85)
    plt.plot(y_pred, label="pred", linestyle="--")
    plt.title(title); plt.xlabel("time"); plt.ylabel("value")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(outp, dpi=150); plt.close()

def _plot_res(y_true, y_pred, label, outp):
    r = y_true - y_pred
    plt.figure(figsize=(12, 4))
    plt.plot(r, alpha=0.85)
    plt.title(f"residuals – {label}"); plt.xlabel("time"); plt.ylabel("error")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(outp, dpi=150); plt.close()

def main():
    print(f"GRU-lean v2 | device={device.type} | out={OUT_DIR}")
    tr_df, te_df, mode = load_split_or_fallback()
    print(f"data mode: {mode}")
    print(f"rows: train={len(tr_df):,}  test={len(te_df):,}")

    targets = [c for c in ["avg_latency", "upload_bitrate", "download_bitrate"] if c in tr_df.columns]
    if not targets:
        raise ValueError("no KPI targets found in train csv")

    results = []
    for t in targets:
        print(f"\n{t}")
        res = run_one(tr_df, te_df, t)
        print(f"  test: mae={res['MAE']:.3f} rmse={res['RMSE']:.3f} r2={res['R2']:.3f}")
        results.append(res)

    out_csv = os.path.join(OUT_DIR, "model_comparison_gru_lean_v2.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nsaved summary: {out_csv}")
    print(f"plots dir: {OUT_DIR}")

if __name__ == "__main__":
    main()