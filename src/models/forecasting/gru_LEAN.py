# -*- coding: utf-8 -*-
"""
auth-nik
leaned out GRU forecasting for large 5G datasets (Colab-friendly).

Key 
- lazywindowing Dataset (no giant 3D arrays)
- optional resampling to reduce rows
- mixed precision on GPU
- rruned features plus float32

"""

import os, math, warnings
from typing import List, Optional
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

_IN_COLAB = "/content" in os.getcwd()
BASE_DIR = "/content" if _IN_COLAB else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results", "forecasting_gru_lean")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PRIMARY   = os.path.join(DATA_DIR, "features_engineered.csv")
CSV_FALLBACK  = "/content/drive/MyDrive/FEATURE/features_engineered.csv"
OVERRIDE_CSV  = os.environ.get("GRU_DATA_CSV", "").strip() or None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # mixed precision only if GPU present

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# Model / data knobs (tune to fit Colab RAM/VRAM)
LOOKBACK     = 32
EPOCHS       = 15
HIDDEN       = 48         # smaller model
N_LAYERS     = 1          # fewer layers => less VRAM
DROPOUT      = 0.1
LR           = 1e-3
BATCH        = 128        # start smaller; script auto-downscales if needed
RESAMPLE     = "15min"    # set to None to disable; try "5min" or "H" for hourly
USE_ONLY_KPIS = True      # use only KPI columns as features (lowest memory)
MAX_FEATURES  = 8         # cap feature count if not using only KPIs
NUM_WORKERS   = 0         # Colab: avoid memory-copy workers


def find_csv() -> str:
    for p in [OVERRIDE_CSV, CSV_PRIMARY, CSV_FALLBACK]:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "features_engineered.csv not found.\n"
        f"Tried:\n - {CSV_PRIMARY}\n - {CSV_FALLBACK}\n"
        "or set GRU_DATA_CSV=/full/path/to/csv"
    )

def load_df() -> pd.DataFrame:
    path = find_csv()
    df = pd.read_csv(path, low_memory=False)
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in features_engineered.csv")

    # datetime index + sort
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Optional: resample to reduce rows massively
    if RESAMPLE:
        # Use means for numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns
        df = df.set_index("timestamp")
        df = df[num_cols].resample(RESAMPLE).mean().dropna().reset_index()
    return df

def select_features(df: pd.DataFrame, target: str) -> List[str]:
    # Keep only numeric and drop the target & 'time'
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in (target, "time")]
    if USE_ONLY_KPIS:
        kpis = [c for c in ["avg_latency", "upload_bitrate", "download_bitrate"] if c in feats]
        feats = kpis or feats
    if MAX_FEATURES and len(feats) > MAX_FEATURES:
        feats = feats[:MAX_FEATURES]
    return feats

def to_float32(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # downcast aggressively
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df


# ------------------------------ Lazy window dataset ------------------------------
class WindowedDataset(Dataset):
    """Slices windows on-the-fly to avoid building huge (N,T,F) arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.lookback = lookback

    def __len__(self):
        return max(0, len(self.X) - self.lookback)

    def __getitem__(self, i):
        j = i + self.lookback
        xw = self.X[j - self.lookback : j]     # (T, F) slice (small copy only)
        yw = self.y[j]
        return torch.from_numpy(xw), torch.from_numpy(yw)


def build_splits(df: pd.DataFrame, target: str):
    feats = select_features(df, target)
    cols = feats + [target]
    # ensure float32
    df = to_float32(df, cols)

    X_all = df[feats].values.astype(np.float32)
    y_all = df[target].values.astype(np.float32)

    # scalers
    x_scaler = StandardScaler()
    X_all = x_scaler.fit_transform(X_all).astype(np.float32)
    y_scaler = MinMaxScaler()
    y_all = y_scaler.fit_transform(y_all.reshape(-1, 1)).astype(np.float32).ravel()

    # temporal split
    n = len(X_all)
    n_tr = int(0.7 * n); n_va = int(0.85 * n)

    X_tr, X_va, X_te = X_all[:n_tr], X_all[n_tr:n_va], X_all[n_va:]
    y_tr, y_va, y_te = y_all[:n_tr], y_all[n_tr:n_va], y_all[n_va:]

    ds_tr = WindowedDataset(X_tr, y_tr, LOOKBACK)
    ds_va = WindowedDataset(X_va, y_va, LOOKBACK)
    ds_te = WindowedDataset(X_te, y_te, LOOKBACK)

    if min(len(ds_tr), len(ds_va), len(ds_te)) == 0:
        raise ValueError(
            f"Not enough samples to build {LOOKBACK}-step windows for target '{target}'. "
            "Reduce LOOKBACK or adjust RESAMPLE to keep more rows."
        )

    # adaptive batch to fit memory
    batch = min(BATCH, max(16, len(ds_tr)//100))  # roughly ≤1% of training windows
    pin = torch.cuda.is_available()

    ld_tr = DataLoader(ds_tr, batch_size=batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    ld_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    ld_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    return ld_tr, ld_va, ld_te, y_scaler, len(feats)


# ------------------------------ Model ------------------------------
class GRUReg(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT):
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
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out, _ = self.gru(x)       # (B, T, H)
        last = out[:, -1, :]       # (B, H)
        return self.head(last)


# ------------------------------ Train / Eval ------------------------------
def train_one(df: pd.DataFrame, target: str):
    loaders = build_splits(df, target)
    ld_tr, ld_va, ld_te, y_scaler, n_feats = loaders

    model = GRUReg(n_features=n_feats).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val = float("inf")
    best = None

    pbar = tqdm(range(1, EPOCHS+1), desc=f"[{target}] epochs", ncols=110)
    for ep in pbar:
        # train 
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for xb, yb in ld_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss_sum += loss.item() * len(xb)
            tr_n += len(xb)
        tr_mse = tr_loss_sum / max(tr_n, 1)

        # val 
        model.eval()
        va_loss_sum, va_n = 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
            for xb, yb in ld_va:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                pred = model(xb)
                va_loss_sum += loss_fn(pred, yb).item() * len(xb)
                va_n += len(xb)
        va_mse = va_loss_sum / max(va_n, 1)

        pbar.set_postfix({"train_mse": f"{tr_mse:.5f}", "val_mse": f"{va_mse:.5f}"})

        if va_mse < best_val:
            best_val = va_mse
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.items()})

    # collect only test preds to save memory
    model.eval()
    preds, trues = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
        for xb, yb in ld_te:
            xb = xb.to(DEVICE, non_blocking=True)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    y_hat = np.vstack(preds).ravel()
    y_true = np.vstack(trues).ravel()

    # inversescale
    y_hat = y_scaler.inverse_transform(y_hat.reshape(-1,1)).ravel()
    y_true = y_scaler.inverse_transform(y_true.reshape(-1,1)).ravel()

    # metrics
    mae = mean_absolute_error(y_true, y_hat)
    rmse = math.sqrt(mean_squared_error(y_true, y_hat))
    r2 = r2_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan

    # plots
    save = target.replace("/", "_")
    plot_predictions(y_true, y_hat, f"GRU (lean) – {target}",
                     os.path.join(OUT_DIR, f"gru_lean_{save}.png"))
    plot_residuals(y_true, y_hat, target,
                   os.path.join(OUT_DIR, f"gru_lean_{save}_residuals.png"))

    print(f"\nTest metrics for {target}:  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"Target": target, "Model": "GRU-lean",
            "Test_MAE": mae, "Test_RMSE": rmse, "Test_R2": r2,
            "SeqLen": LOOKBACK, "Epochs": EPOCHS, "Batch": ld_tr.batch_size,
            "n_features": n_feats, "Resample": RESAMPLE or "none"}


def plot_predictions(y_true, y_pred, title, out_path):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label="Actual", alpha=0.85)
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.title(title); plt.xlabel("Time steps"); plt.ylabel("Value")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def plot_residuals(y_true, y_pred, label, out_path):
    r = y_true - y_pred
    plt.figure(figsize=(12,4))
    plt.plot(r, alpha=0.85)
    plt.title(f"Residuals – {label}"); plt.xlabel("Time steps"); plt.ylabel("Error")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def main():
    print(f"GRU (lean)  | device={DEVICE.type} | AMP={USE_AMP} | out={OUT_DIR}")
    print(f"Settings    | LOOKBACK={LOOKBACK}  EPOCHS={EPOCHS}  HIDDEN={HIDDEN}  LAYERS={N_LAYERS}")
    print(f"Data        | RESAMPLE={RESAMPLE}  ONLY_KPIS={USE_ONLY_KPIS}  MAX_FEATURES={MAX_FEATURES}")

    df = load_df()
    targets = [c for c in ["avg_latency", "upload_bitrate", "download_bitrate"] if c in df.columns]
    if not targets:
        raise ValueError("No KPI targets found (avg_latency/upload_bitrate/download_bitrate).")

    results = []
    for t in targets:
        print(f"\ntarget: {t}")
        res = train_one(df, t)
        results.append(res)

    out_csv = os.path.join(OUT_DIR, "model_comparison_gru_lean.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n[OK] Saved summary: {out_csv}")
    print(f"[OK] Plots in: {OUT_DIR}")

if __name__ == "__main__":
    main()
