#auth: nik
# lean-ish GRU that works with the new train/test csv


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
def _norm_names(df: pd.DataFrame) -> pd.DataFrame:#LOWER CASE HEADERS aliases messy anmes ot clea ones| nnames on left rename to on the right 
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

def _pick_kpis(df: pd.DataFrame):#returns [avg_latency, upload_bitrate, dowload_bitrate]- these are kPI identifed in the dataframe 
    return [c for c in ["avg_latency","upload_bitrate","download_bitrate"] if c in df.columns]

def _select_feats(df: pd.DataFrame, target: str):
    num = df.select_dtypes(include=[np.number]).columns.tolist()#takes all numeric column names| text date columsn ingored 
    feats = [c for c in num if c != target]
    if use_only_kpis:#drop the target so it doesnt leak the label into inputs 
        kpi = _pick_kpis(df)#keep only kpi bases (avg_latency, upload_bitrate, download_bitrate )
        feats = [c for c in feats if c in kpi or c.endswith("_lag1")] or kpi
    if max_features and len(feats) > max_features:#MAX FEATURES set to 16 hence keep to max features only 16 features 
        feats = feats[:max_features]
    return feats

def _downcast(df: pd.DataFrame, cols):# turn the columns into numbers - non numeric or unparasabe entires become NaN 
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)#down case numeric data to float32 isntead fo 64 which will halves memory 
    return df

# dataset
#turns a logn tiem series into overlapping windows of length w (lookback) for each window x[i : i + 2] te allnel is the next point y[i+w] so the model learns " given the last w steps, predict the next step "
class WinDS(Dataset):#stores c,y arrays plsu w = lookback, len : number of pssible windows = (len (x) - w)| _getitem_(i) returns the window X[i; i+ w] and the target at tiem i+w (ie predict next step )
    def __init__(self, X, y, w):#w = lookback 
        self.X = X
        self.y = y.reshape(-1,1)
        self.w = w
    def __len__(self): #number of windws to slide over. | ex t=100 and w=32 you get 100-32 training samples 
        return max(0, len(self.X) - self.w)
    def __getitem__(self, i):#for index i build a window fo previous w rows and pairs it with the next point 
        j = i + self.w#compute ('next' index after a window of size w)
        xw = self.X[j-self.w:j]#the last w rows ending at j-1
        return torch.from_numpy(xw), torch.from_numpy(self.y[j])
#expalple (w=3 ) | i = 0 -> window = X[0;3] (rows 0,1,2), label = y[3] | i = 1 -> window = X[1:4] (rows 1,2,3), label = y[4]
def _make_loaders_from_arrays(X_tr, y_tr, X_te, y_te):#uses the 15% train as validation | wraps each split in windDS to get windows of length lookback = 32 | builds DataLoaders (no shuffle -> preserves order) | whyy no shuffle to keep order to avoid leaking the futrue into teh past 
    n = len(X_tr)
    n_va = int(0.15 * n)#uses 15% fo trainign period as validation to mimic future data | keeps tmeroral order (no random split )
    X_va, y_va = X_tr[-n_va:], y_tr[-n_va:]
    X_tr, y_tr = X_tr[:-n_va], y_tr[:-n_va]

    ds_tr = WinDS(X_tr, y_tr, lookback)#turns squences int overlaping windows of length lookback (e.g 32) with the next point as the label 
    ds_va = WinDS(X_va, y_va, lookback)
    ds_te = WinDS(X_te, y_te, lookback)

    if min(len(ds_tr), len(ds_va), len(ds_te)) == 0:
        raise ValueError("not enough rows after windowing. reduce lookback maybe.")#if squence is too short for lookback; fail early with error 

    batch = min(batch0, max(16, len(ds_tr)//100))#lower bound 16, scaled with daaset size, but capped at batch 0 (e.g 128 )
    pin = torch.cuda.is_available()#if gpu is there pseed up cpu -> gpu moves 

    ld_tr = DataLoader(ds_tr, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    ld_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    ld_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return ld_tr, ld_va, ld_te#shuffle=false prevent leaking into past prevent shuffling 

# model
#GRU: a recurrrent layer with gates tha tlearn what ot keep/forget from recent timesteps 
#MLP head: small fully ocnnnected network tha tmaps the GRU last hidden state ot the target
class GRUReg(nn.Module):#gru process teh 32 x n_feature sequence and outputs a vector per time step- it takes the last tiem step (ut [:, -1, :]) -> feed into a small MLP head -> one scalar: 
    #reads a window of shape [batch 32, n_features] -> GRU encodes teh sequence into hidden states -> take the last time step (summary of thewindow) -> small MLP head turns into one scalar prediction (the next value)
    def __init__(self, n_features, hidden=hidden, n_layers=n_layers, dropout=dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,#how many features per time step 
            hidden_size=hidden,#size of GRU (e.g 48) 
            num_layers=n_layers,#stack depth (1 here)
            batch_first=True,#x is [batch, seq, feat]
            dropout=dropout if n_layers > 1 else 0.0,#Gru own drop out layer only if layers are more than 1 whcih they arnt 
        )#out shape [batch, seq_len, hidden]
        self.head = nn.Sequential(#MLP HEAD (MULTI LAYER PERCEPTION - FEED FORWARD NEURAL NETWORK - STACKED LINEAR (DENSE) LAYERS TAKES MULTIPLE INPUTS AND THEN OOUTPUTS THE PREDICTION 
            nn.Linear(hidden, 64),#dense layer (compress/expand features )
            nn.ReLU(),#nonlinearity: helps modle learn the curves not just linesidentify complex reationships 
            nn.Dropout(dropout),#regularization: randomly zeros units to reduce overfitting 
            nn.Linear(64, 1),#final dense: map to one scalar (the prediction) [the next value fo the target]
        )
    def forward(self, x):#run sequence throguh gru 
        out, _ = self.gru(x)#take the GRU output at the last time step out has shape [batch, seq_len, n_feature] - | n_features -> is the number of of input clumsn (16) FEATS are hte list of input columns deied to use after filtering 
        last = out[:, -1, :]#pass the vector thoguht eh MLP head last = out [:, -1,:]
        return self.head(last)#[batch, hidden] -> [batch, 1 ]

# data loading for the new split
#1. prefer both trian_csv and test_csv, clean columsn with _norm_names()
#2.if csv missing raise erorr |if exists read and normalize column nmaes | if theres a 'time' column convert it to a proper timestamp - drop invalid orws, sort by time, and resset index | PRESENRVES TEMPORAL ORDER FOR A PROPER CHRONLOGICAL SPLIT
#do 80/20 time ordered split | TR = FIRST 80% TE = LAST 20% 
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
    #1. Standadize x: help GRU trian stably 
    #2. MinMax y: keeps target in compact range: easy to invert later 
    #3. fit on trian only: prevent test set leakage 
    #4. float32 everywhere: faster and smaller, matches toch tensors 
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
