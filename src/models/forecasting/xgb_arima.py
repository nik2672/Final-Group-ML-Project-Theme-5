import os
import warnings
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "xgbarima_results")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "features_for_forecasting_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "features_for_forecasting_test.csv")

IS_M_SERIES = platform.system() == "Darwin" and platform.machine() == "arm64"

def _norm_names(df: pd.DataFrame) -> pd.DataFrame:#lower case header fix and noramlises messalases  so theyre conssitant 
    m = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=m)
    ren = {
        "upload_bitrate_mbits/sec": "upload_bitrate",
        "download_bitrate_rx_mbytes": "download_bitrate",
        "avg_latency_lag_1": "avg_latency_lag1",
        "avg_latencylag1": "avg_latency_lag1",
        "upload_bitrate_mbits/sec_lag1": "upload_bitrate_lag1",
        "upload_bitrate_lag_1": "upload_bitrate_lag1",
        "download_bitrate_rx_mbytes_lag1": "download_bitrate_lag1",
        "download_bitrate_lag_1": "download_bitrate_lag1",
    }
    for k, v in ren.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def _clean_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feats):#replace +inf with nan, putes trian median per features, then fils nans with both train/test -> returns numpy arrays X_tr, X_te 
    trX = train_df[feats].replace([np.inf, -np.inf], np.nan)
    med = trX.median()
    X_tr = trX.fillna(med).values
    X_te = test_df[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
    return X_tr, X_te

def _clean_targets(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):#coorces target to a float tuns Inf -> NAN, builds bulian masks m_tr, m_te indicating on NAN targets | reuturns y_tr, y_te (with nans dopped + the masks )
    y_tr = train_df[target].astype(float).replace([np.inf, -np.inf], np.nan)#WHY? metrics need real target vlaeu; test rows with missing target cant be scored, so you dorp them cleanly 
    y_te = test_df[target].astype(float).replace([np.inf, -np.inf], np.nan)
    # Drop NaN targets independently per split
    m_tr = y_tr.notna().values
    m_te = y_te.notna().values
    return y_tr.values[m_tr], y_te.values[m_te], m_tr, m_te

def _align_after_target_mask(X_tr, X_te, m_tr, m_te):#what is does: applies target masks to teh feature matricies, so X and y stay aligned row by row; why: when you drop target NANs you msut drop the same rows from features
    return X_tr[m_tr], X_te[m_te]

def run_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):#core training/eval | 1. copy splits tr,te | 2. feature slist feats =keep only columsn that exsit in both splits and are not the target | 3. categorical encoding -> concatinte train + test column as a single categorical then takes codes for each split with a shared category set , why ensures (IDS square_id, hour, dayy map t teh same integrers in train and test )
    # Encode categoricals consistently across splits
    tr = train_df.copy()
    te = test_df.copy()

    # Choose features: keep numerics + encoded categoricals, exclude target
    feats = [c for c in tr.columns if c != target and c in te.columns]

    # Consistent category encoding (only for object dtypes)
    for c in feats:
        if tr[c].dtype == "object" or te[c].dtype == "object":
            both = pd.Categorical(pd.concat([tr[c].astype("object"), te[c].astype("object")], axis=0))
            tr[c] = pd.Categorical(tr[c].astype("object"), categories=both.categories).codes
            te[c] = pd.Categorical(te[c].astype("object"), categories=both.categories).codes

    # Replace inf/NaN; impute features by train median only
    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)

    # Make sure features exist
    feats = [c for c in feats if c in tr.columns and c in te.columns]

    # Clean targets + align features by target masks (drop rows where target is NaN)
    y_tr, y_te, m_tr, m_te = _clean_targets(tr, te, target)
    # Impute features using train medians
    X_tr_full, X_te_full = _clean_features(tr, te, feats)
    X_tr, X_te = _align_after_target_mask(X_tr_full, X_te_full, m_tr, m_te)

    if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
        raise ValueError(f"No data left after cleaning for target='{target}'")

    params = dict(#peramsn are good fro tabular forecasting 
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=os.cpu_count() or 4,
        tree_method="hist" if IS_M_SERIES else "auto",
    )

    model = XGBRegressor(**params)#best perams 
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)#mean absoluee errro, rmse penalizes big erors r^2 variance exlaned non safe when y vairance is 0 
    rmse = np.sqrt(((y_te - y_pred) ** 2).mean())
    r2   = r2_score(y_te, y_pred) if len(np.unique(y_te)) > 1 else np.nan

    _plot_xgb_scatter(y_te, y_pred, target)
    _plot_xgb_feature_importance(model, feats, target)

    return {
        "Target": target,
        "Model": "XGBoost",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2) if r2 == r2 else None,  # handle nan
    }

def _plot_xgb_scatter(y_true, y_pred, target):#scatter of actual vs predicted and dashed y = x line. tight clustering on the line = good | residuals vs predicted with a dashed zzero line. a flat band aroudn 0 means low bias, patterns (U sharep) suggest under/overfit in ranges 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.5, s=12)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.8)
    axes[0].set_title(f"XGBoost: Actual vs Predicted – {target}")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].grid(alpha=0.3)

    resid = y_true - y_pred
    axes[1].scatter(y_pred, resid, alpha=0.5, s=12)
    axes[1].axhline(0.0, color="r", linestyle="--", lw=1.8)
    axes[1].set_title(f"XGBoost: Residuals – {target}")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fp = os.path.join(OUT_DIR, f"xgb_{target}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_xgb_feature_importance(model, feats, target, top_n=20):#gain based imporatnce bar chart; show sthe top contributers
    importances = model.feature_importances_
    n = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:n]
    plt.figure(figsize=(10, 6))
    plt.bar(range(n), importances[idx])
    plt.xticks(range(n), [feats[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature Importance – {target}")
    plt.ylabel("gain")
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, f"feature_importance_{target}.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()

def run_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, order=(2,1,2), max_forecast=None):
    # Build a single series with boundary at len(train)
    s_tr = (train_df[target].astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill"))
    s_te = (test_df[target].astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill"))

    train = s_tr.values
    test  = s_te.values
    if len(train) < 10 or len(test) == 0:
        return {"Target": target, "Model": "ARIMA", "MAE": None, "RMSE": None, "R2": None}

    try:
        model = ARIMA(train, order=order)
        fit = model.fit()
        steps = len(test) if max_forecast is None else min(max_forecast, len(test))
        fc = fit.forecast(steps=steps)
        test_subset = test[:steps]

        mae  = mean_absolute_error(test_subset, fc)
        rmse = np.sqrt(((test_subset - fc) ** 2).mean())
        r2   = None  # R2 not meaningful for pure forecast horizon in this simple setup

        _plot_arima_series(train, test, fc, target)
        return {"Target": target, "Model": "ARIMA", "MAE": float(mae), "RMSE": float(rmse), "R2": r2}
    except Exception as e:
        print(f"[ARIMA:{target}] failed: {e}")
        _plot_arima_series(train, test, np.array([]), target)
        return {"Target": target, "Model": "ARIMA", "MAE": None, "RMSE": None, "R2": None}

def _plot_arima_series(train, test, fc, target):
    n = len(train)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(range(0, n), train, label="Train", alpha=0.8)
    plt.plot(range(n, n + len(test)), test, label="Actual (test)", alpha=0.8)
    if len(fc):
        plt.plot(range(n, n + len(fc)), fc, label="Forecast", linestyle="--", lw=2)
    plt.title(f"ARIMA – {target}")
    plt.xlabel("time"); plt.ylabel(target)
    plt.grid(alpha=0.3); plt.legend()
    fp = os.path.join(OUT_DIR, f"arima_{target}.png")
    plt.tight_layout(); plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)

def main():
    print(f"XGB+ARIMA | reading:\n  train={TRAIN_CSV}\n  test ={TEST_CSV}\n  out  ={OUT_DIR}")

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError("Train/Test CSVs not found. Check paths.")

    tr = pd.read_csv(TRAIN_CSV, low_memory=False)
    te = pd.read_csv(TEST_CSV,  low_memory=False)
    tr, te = _norm_names(tr), _norm_names(te)

    targets = [c for c in ["avg_latency", "upload_bitrate", "download_bitrate"] if c in tr.columns]
    if not targets:
        num = tr.select_dtypes(include=[np.number]).columns.tolist()
        if not num:
            raise ValueError("No numeric target found.")
        targets = [num[0]]

    all_rows = []

    for t in targets:
        print(f"\n=== XGBoost: {t} ===")
        try:
            res_xgb = run_xgboost(tr, te, t)
            print(f"  MAE={res_xgb['MAE']:.3f}  RMSE={res_xgb['RMSE']:.3f}  R2={res_xgb['R2'] if res_xgb['R2'] is not None else 'NA'}")
            all_rows.append(res_xgb)
        except Exception as e:
            print(f"[XGBoost:{t}] failed: {e}")

    for t in targets:
        print(f"\n=== ARIMA: {t} ===")
        res_arima = run_arima(tr, te, t, order=(2,1,2), max_forecast=None)
        if res_arima["MAE"] is not None:
            print(f"  MAE={res_arima['MAE']:.3f}  RMSE={res_arima['RMSE']:.3f}")
        else:
            print("  (no metrics)")
        all_rows.append(res_arima)

    out_csv = os.path.join(OUT_DIR, "model_comparison_xgbarima.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(f"[plots] {OUT_DIR}")

if __name__ == "__main__":
    main()
