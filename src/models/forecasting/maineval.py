"""
Unified Forecasting Evaluation (with tuning) for 5G performance.
- Chronological split + leakage guards
- Naive baseline
- Residual diagnostics (mean/std, lag-1 ACF, Durbin–Watson)
- XGBoost with time-aware hyperparameter tuning (blocked validation + early stopping)
- ARIMA small grid search (AIC) + test metrics
Outputs:
- results/forecasting/model_comparison.csv
- residual plots per model/target
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys, io, random
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import xgboost as xgb  # for callback compatibility across versions
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import multiprocessing

# ---------- Windows-safe stdout ----------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", errors="replace")
    except Exception:
        pass
# ----------------------------------------

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'forecasting')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Env
N_CORES = multiprocessing.cpu_count()
IS_M_SERIES = platform.system() == 'Darwin' and platform.machine() == 'arm64'
print(f"{'[OK] M-Series' if IS_M_SERIES else 'Running on standard architecture'} ({N_CORES} cores)")
print("=" * 70)
print("5G Network Performance Forecasting Analysis (Eval + Tuning)")
print("=" * 70)

# ========================= Config toggles =========================
TIME_COLS = ('timestamp','day_id','hour')  # we auto-create day_id if missing
TARGETS = {
    'avg_latency': 'Avg Latency (ms)',
    'upload_bitrate': 'Upload (Mbps)',
    'download_bitrate': 'Download (Mbps)'
}
TEST_FRACTION = 0.20
XGB_TUNE_MAX_TRAIN = 500_000
XGB_N_TRIALS = 15
XGB_EARLY_STOPPING_ROUNDS = 50
ARIMA_SAMPLE = 50_000
ARIMA_P = [1, 2, 3]
ARIMA_D = [0, 1]
ARIMA_Q = [0, 1, 2]
# ================================================================


# ----------------------- Utility & Guards ------------------------

def load_forecasting_data() -> pd.DataFrame:
    """Load forecasting features and ensure a time column exists."""
    print("\nLoading forecasting features...")
    path = os.path.join(DATA_PATH, 'features_for_forecasting.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Forecasting features not found at {path}. "
            "Please create this file (see feature engineering step)."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} samples with {len(df.columns)} features")

    time_candidates = [c for c in TIME_COLS if c in df.columns]
    if not time_candidates:
        print("[WARN] No time column found; assuming current row order is chronological and creating 'day_id'.")
        df = df.reset_index(drop=True)
        df['day_id'] = np.arange(len(df))
    else:
        if 'timestamp' in df.columns:
            try:
                _ = pd.to_datetime(df['timestamp'])
            except Exception:
                print("[WARN] 'timestamp' present but not parseable; proceeding with raw order.")
    return df

def require_time_col(df: pd.DataFrame) -> str:
    time_col = next((c for c in TIME_COLS if c in df.columns), None)
    if not time_col:
        raise ValueError(f"No time column found. Add one of {TIME_COLS} to ensure valid evaluation.")
    return time_col

def chronological_sort(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    return df.sort_values(time_col, kind='stable').reset_index(drop=True)

def chronological_split(df: pd.DataFrame, time_col: str, test_fraction: float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = chronological_sort(df, time_col)
    n = len(df); split = int(n * (1 - test_fraction))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    assert train[time_col].iloc[-1] <= test[time_col].iloc[0], "Time leakage: train extends beyond test start."
    return train, test

def naive_persistence(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    if len(y_test) == 0: return np.array([])
    baseline = np.empty_like(y_test, dtype=float)
    baseline[0] = y_train[-1] if len(y_train) else y_test[0]
    if len(y_test) > 1: baseline[1:] = y_test[:-1]
    return baseline

def residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_true - y_pred
    r_mean = float(np.mean(resid)) if len(resid) else np.nan
    r_std = float(np.std(resid)) if len(resid) else np.nan
    if len(resid) >= 2:
        r = np.corrcoef(resid[1:], resid[:-1])[0,1]
        lag1 = float(r) if np.isfinite(r) else np.nan
    else:
        lag1 = np.nan
    dw = float(durbin_watson(resid)) if len(resid) > 0 else np.nan
    return {"mean": r_mean, "std": r_std, "lag1": lag1, "dw": dw, "resid": resid}

def plot_residuals(residuals: np.ndarray, target: str, prefix: str):
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(residuals)), residuals, alpha=0.7)
    plt.axhline(0, linestyle='--', linewidth=1.5, color='red')
    plt.title(f"{prefix.upper()} Residuals Over Time - {target}")
    plt.xlabel("Test Time Index"); plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    out = os.path.join(OUTPUT_PATH, f"{prefix}_residuals_time_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

# --------------------- Feature Engineering ----------------------

def add_time_features(df: pd.DataFrame, target_col: str, time_col: str) -> pd.DataFrame:
    """Add lag and rolling stats (no leakage)."""
    df = chronological_sort(df, time_col).copy()
    for L in [1, 2, 3, 24]:
        df[f"{target_col}_lag{L}"] = df[target_col].shift(L)
    for W in [6, 24, 72]:
        df[f"{target_col}_rollmean_{W}"] = df[target_col].shift(1).rolling(W, min_periods=max(3, W//3)).mean()
        df[f"{target_col}_rollstd_{W}"]  = df[target_col].shift(1).rolling(W, min_periods=max(3, W//3)).std()
    if 'timestamp' in df.columns:
        tt = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = tt.dt.hour
        df['dow']  = tt.dt.dayofweek
    return df

def build_feature_matrix(df: pd.DataFrame, target_col: str, time_col: str) -> Tuple[pd.DataFrame, List[str]]:
    exclude_exact = {target_col, time_col, 'square_id', 'day', 'day_id', 'timestamp'}
    other_targets = set(TARGETS.keys()) - {target_col}
    drop_cols = set()
    for c in df.columns:
        if c in exclude_exact or c in other_targets:
            drop_cols.add(c)
    safe_cols = [c for c in df.columns if c not in drop_cols]
    X = df[safe_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    X = X.dropna(axis=1, how='all')
    features = list(X.columns)
    if not any(('lag' in f or 'roll' in f) for f in features):
        print("WARNING: No lag/roll features detected; predictions may underperform the naive baseline.")
    return X, features

# -------------------- XGBoost (Tuned) ---------------------------

def xgb_param_space() -> Dict[str, List]:
    return {
        "n_estimators": [400, 600, 800, 1000],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "min_child_weight": [1, 5, 10]
    }

def sample_params(space: Dict[str, List]) -> Dict:
    return {k: random.choice(v) for k, v in space.items()}

def fit_xgb_version_safe(model, X, y, eval_set, early_stopping_rounds, verbose=False):
    """
    Version-proof XGBoost fit:
    - Prefer early_stopping_rounds kwarg.
    - Fallback to callbacks API.
    - Final fallback: fit without early stopping.
    Also sets eval_metric on the model (works in old APIs that reject it in fit()).
    """
    # Ensure eval_metric is set at model level for old versions
    try:
        model.set_params(eval_metric='rmse')
    except Exception:
        pass

    # Attempt 1: full modern signature
    try:
        return model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='rmse',
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds
        )
    except TypeError:
        pass

    # Attempt 2: early stopping but without eval_metric kwarg
    try:
        return model.fit(
            X, y,
            eval_set=eval_set,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds
        )
    except TypeError:
        pass

    # Attempt 3: callbacks API
    try:
        cb = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        return model.fit(
            X, y,
            eval_set=eval_set,
            verbose=verbose,
            callbacks=cb
        )
    except Exception:
        pass

    # Final fallback: plain fit
    return model.fit(
        X, y,
        eval_set=eval_set,
        verbose=verbose
    )

def train_xgb_with_tuning(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          target_name: str) -> Tuple[XGBRegressor, np.ndarray, Dict]:
    # Downsample most-recent block of train to keep tuning fast
    if len(X_train) > XGB_TUNE_MAX_TRAIN:
        start = len(X_train) - XGB_TUNE_MAX_TRAIN
        X_tr_small = X_train[start:]
        y_tr_small = y_train[start:]
    else:
        X_tr_small = X_train
        y_tr_small = y_train

    # inner val split (last 10% of small train)
    split = int(0.9 * len(X_tr_small))
    X_tr_inner, X_val_inner = X_tr_small[:split], X_tr_small[split:]
    y_tr_inner, y_val_inner = y_tr_small[:split], y_tr_small[split:]

    space = xgb_param_space()
    best = {"rmse": float("inf"), "params": None}

    print(f"\n[Hyperparam Tuning] Trials: {XGB_N_TRIALS} (target={target_name})")
    for i in range(1, XGB_N_TRIALS+1):
        params = sample_params(space)
        model = XGBRegressor(
            n_jobs=N_CORES,
            random_state=42,
            tree_method='hist' if IS_M_SERIES else 'auto',
            **params
        )
        fit_xgb_version_safe(
            model,
            X_tr_inner, y_tr_inner,
            eval_set=[(X_val_inner, y_val_inner)],
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            verbose=False
        )
        y_val_pred = model.predict(X_val_inner)
        rmse = float(np.sqrt(mean_squared_error(y_val_inner, y_val_pred)))
        print(f"  Trial {i:02d}: RMSE={rmse:.4f} | {params}")
        if rmse < best["rmse"]:
            best = {"rmse": rmse, "params": params, "best_iteration": getattr(model, "best_iteration", None)}

    print(f"[Best Params] {best['params']} | val_RMSE={best['rmse']:.4f}")

    # Refit best on full train, eval on test
    best_model = XGBRegressor(
        n_jobs=N_CORES,
        random_state=42,
        tree_method='hist' if IS_M_SERIES else 'auto',
        **best["params"]
    )
    # early stopping on last 10% of training
    split_full = int(0.9 * len(X_train))
    X_tr_full, X_es = X_train[:split_full], X_train[split_full:]
    y_tr_full, y_es = y_train[:split_full], y_train[split_full:]

    fit_xgb_version_safe(
        best_model,
        X_tr_full, y_tr_full,
        eval_set=[(X_es, y_es)],
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        verbose=False
    )
    y_pred_test = best_model.predict(X_test)

    # Metrics
    train_pred = best_model.predict(X_train)
    metrics = {
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, y_pred_test),
    }
    return best_model, y_pred_test, metrics

# -------------------- ARIMA (tuned small grid) -------------------

def prepare_arima_series(df: pd.DataFrame, target_col: str, time_col: str, sample_size: int) -> np.ndarray:
    df = df.dropna(subset=[target_col]).copy()
    df = chronological_sort(df, time_col)
    if len(df) > sample_size:
        df = df.iloc[-sample_size:].copy()  # contiguous recent block
    return df[target_col].to_numpy()

def arima_grid_search(ts: np.ndarray,
                      p_list=ARIMA_P, d_list=ARIMA_D, q_list=ARIMA_Q,
                      verbose=True) -> Tuple[Tuple[int,int,int], Dict]:
    best = {"aic": float("inf"), "order": None, "fit": None}
    for p in p_list:
        for d in d_list:
            for q in q_list:
                order = (p,d,q)
                try:
                    model = ARIMA(ts[:int(0.8*len(ts))], order=order)
                    fit = model.fit()
                    aic = float(fit.aic)
                    if verbose: print(f"  ARIMA{order} AIC={aic:.2f}")
                    if aic < best["aic"]:
                        best = {"aic": aic, "order": order, "fit": fit}
                except Exception:
                    continue
    if verbose and best["order"] is not None:
        print(f"[Best ARIMA] order={best['order']} | AIC={best['aic']:.2f}")
    return best["order"], best

def arima_eval(ts: np.ndarray, order: Tuple[int,int,int], target_name: str) -> Dict:
    n = len(ts)
    split = int(0.8*n)
    train, test = ts[:split], ts[split:]
    model = ARIMA(train, order=order)
    fit = model.fit()
    steps = len(test)
    fc = fit.forecast(steps=steps)
    mae = mean_absolute_error(test, fc)
    rmse = np.sqrt(mean_squared_error(test, fc))
    rstats = residual_stats(test, fc)
    # Naive baseline on same horizon
    naive = np.empty_like(test, dtype=float)
    if steps > 0:
        naive[0] = train[-1]
        if steps > 1: naive[1:] = test[:-1]
    n_mae = mean_absolute_error(test, naive)
    n_rmse = np.sqrt(mean_squared_error(test, naive))
    plot_residuals(rstats["resid"], target_name, prefix="arima")
    print(f"  ARIMA{order} -> MAE: {mae:.3f}, RMSE: {rmse:.3f} | Resid mean={rstats['mean']:.3f} lag1={rstats['lag1']:.3f} DW={rstats['dw']:.3f}")
    print(f"  Naive baseline -> MAE: {n_mae:.3f}, RMSE: {n_rmse:.3f}")
    return {
        "mae": mae, "rmse": rmse,
        "resid_mean": rstats["mean"], "resid_lag1": rstats["lag1"], "resid_dw": rstats["dw"],
        "naive_mae": n_mae, "naive_rmse": n_rmse
    }

# --------------------------- Orchestration -----------------------

def run_xgb_block(df: pd.DataFrame, target_col: str, target_desc: str, time_col: str) -> Dict:
    print(f"\n--- {target_desc} ---")
    df_t = df.dropna(subset=[target_col]).copy()
    df_t = add_time_features(df_t, target_col, time_col)
    lag_roll_cols = [c for c in df_t.columns if ('lag' in c or 'roll' in c)]
    df_t = df_t.dropna(subset=lag_roll_cols)

    train_df, test_df = chronological_split(df_t, time_col, TEST_FRACTION)
    X_train_full, feats = build_feature_matrix(train_df, target_col, time_col)
    X_test_full, _ = build_feature_matrix(test_df, target_col, time_col)
    y_train = train_df[target_col].to_numpy()
    y_test  = test_df[target_col].to_numpy()

    print(f"  Using time column '{time_col}' (train={len(train_df):,}, test={len(test_df):,})")
    print(f"  Features: {len(feats)} (lags/rolls included)")

    model, y_pred, mets = train_xgb_with_tuning(
        X_train_full.values, y_train,
        X_test_full.values, y_test,
        target_desc
    )

    rstats = residual_stats(y_test, y_pred)
    plot_residuals(rstats["resid"], target_desc, prefix="xgb")

    y_naive = naive_persistence(y_train, y_test)
    naive_mae = mean_absolute_error(y_test, y_naive)
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_naive))
    print(f"  Naive baseline -> MAE: {naive_mae:.3f}, RMSE: {naive_rmse:.3f}")

    print("\n  XGB (tuned) Results:")
    print(f"    Train MAE: {mets['train_mae']:.3f} | Test MAE: {mets['test_mae']:.3f}")
    print(f"    Train RMSE: {mets['train_rmse']:.3f} | Test RMSE: {mets['test_rmse']:.3f}")
    print(f"    Train R^2: {mets['train_r2']:.3f} | Test R^2: {mets['test_r2']:.3f}")
    print(f"    Residuals (test) mean={rstats['mean']:.3f}, lag1={rstats['lag1']:.3f}, DW={rstats['dw']:.3f}")

    return {
        "Target": target_desc,
        "Model": "XGBoost(tuned)",
        "MAE": mets["test_mae"],
        "RMSE": mets["test_rmse"],
        "R2": mets["test_r2"],
        "Naive_MAE": naive_mae,
        "Naive_RMSE": naive_rmse,
        "Resid_Mean": rstats["mean"],
        "Resid_Lag1ACF": rstats["lag1"],
        "Resid_DW": rstats["dw"]
    }

def run_arima_block(df: pd.DataFrame, target_col: str, target_desc: str, time_col: str) -> Dict:
    print(f"\n--- {target_desc} ---")
    ts = prepare_arima_series(df, target_col, time_col, sample_size=ARIMA_SAMPLE)
    if len(ts) < 1000:
        print("  Series too short for ARIMA grid; skipping.")
        return {
            "Target": target_desc, "Model": "ARIMA", "MAE": np.nan, "RMSE": np.nan, "R2": None,
            "Naive_MAE": np.nan, "Naive_RMSE": np.nan, "Resid_Mean": np.nan, "Resid_Lag1ACF": np.nan, "Resid_DW": np.nan
        }
    order, _ = arima_grid_search(ts, verbose=True)
    if order is None:
        print("  ARIMA grid failed; skipping.")
        return {
            "Target": target_desc, "Model": "ARIMA", "MAE": np.nan, "RMSE": np.nan, "R2": None,
            "Naive_MAE": np.nan, "Naive_RMSE": np.nan, "Resid_Mean": np.nan, "Resid_Lag1ACF": np.nan, "Resid_DW": np.nan
        }
    res = arima_eval(ts, order, target_desc)
    return {
        "Target": target_desc, "Model": f"ARIMA{order}",
        "MAE": res["mae"], "RMSE": res["rmse"], "R2": None,
        "Naive_MAE": res["naive_mae"], "Naive_RMSE": res["naive_rmse"],
        "Resid_Mean": res["resid_mean"], "Resid_Lag1ACF": res["resid_lag1"], "Resid_DW": res["resid_dw"]
    }

def main():
    df = load_forecasting_data()
    available = {k: v for k, v in TARGETS.items() if k in df.columns}
    if not available:
        raise ValueError("No expected targets found in data.")

    time_col = require_time_col(df)

    results = []

    print("\n" + "=" * 70)
    print("XGBoost (tuned) – Time-aware with lags/rolling")
    print("=" * 70)
    for tcol, tdesc in available.items():
        out = run_xgb_block(df, tcol, tdesc, time_col)
        results.append(out)

    print("\n" + "=" * 70)
    print("ARIMA – Small grid (contiguous, sorted series)")
    print("=" * 70)
    for tcol, tdesc in available.items():
        out = run_arima_block(df, tcol, tdesc, time_col)
        results.append(out)

    comp = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_PATH, 'model_comparison.csv')
    comp.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved comparison: {out_csv}")

    print("\nPerformance Summary:")
    for row in results:
        print(f"\n{row['Model']} - {row['Target']}:")
        print(f"  MAE: {row['MAE']:.3f} | RMSE: {row['RMSE']:.3f} | R^2: {row['R2'] if row['R2'] is not None else '—'}")
        print(f"  Naive -> MAE: {row['Naive_MAE']:.3f} | RMSE: {row['Naive_RMSE']:.3f}")
        print(f"  Residuals -> mean: {row['Resid_Mean']:.3f} | lag1_ACF: {row['Resid_Lag1ACF']:.3f} | DW: {row['Resid_DW']:.3f}")

if __name__ == "__main__":
    main()
