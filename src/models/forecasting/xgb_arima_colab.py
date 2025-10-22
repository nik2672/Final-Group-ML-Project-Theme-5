import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import platform

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

IN_COLAB = "/content" in os.getcwd()
OUT_DIR = "/content/results/xgbarima_results" if IN_COLAB else os.path.join(os.getcwd(), "results", "xgbarima_results")
os.makedirs(OUT_DIR, exist_ok=True)

IS_M_SERIES = platform.system() == "Darwin" and platform.machine() == "arm64"

def _find_split_csvs():

    tr_env = os.environ.get("XGB_TRAIN_CSV", "").strip()
    te_env = os.environ.get("XGB_TEST_CSV", "").strip()
    if tr_env and te_env and os.path.exists(tr_env) and os.path.exists(te_env):
        return tr_env, te_env, "env"

    drive = "/content/drive/MyDrive"
    tr = os.path.join(drive, "DATA-NEW", "features_for_forecasting_train.csv")
    te = os.path.join(drive, "DATA-NEW", "features_for_forecasting_test.csv")
    if os.path.exists(tr) and os.path.exists(te):
        return tr, te, "drive/DATA-NEW"

    tr_repo = "/content/data/features_for_forecasting_train.csv"
    te_repo = "/content/data/features_for_forecasting_test.csv"
    if os.path.exists(tr_repo) and os.path.exists(te_repo):
        return tr_repo, te_repo, "repo(/content/data)"

    raise FileNotFoundError(
        "Couldn't find train/test CSVs.\n"
        "Mount Drive and place files in MyDrive/DATA-NEW/ (non-improved names), or set env vars "
        "XGB_TRAIN_CSV / XGB_TEST_CSV."
    )

def _norm_names(df: pd.DataFrame) -> pd.DataFrame:
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

def forecast_metrics(y_true, y_pred, train_series=None, compute_r2=True):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if compute_r2 and np.std(y_true) > 0 else None

    skill = None
    mase  = None
    if train_series is not None and len(train_series) >= 2 and len(y_true) == len(y_pred):
        insample_naive = np.abs(np.diff(train_series))
        denom = float(insample_naive.mean()) if insample_naive.size else np.nan

        naive = np.r_[train_series[-1], y_true[:-1]]  # persistence baseline on test
        ss_model = float(np.sum((y_true - y_pred) ** 2))
        ss_naive = float(np.sum((y_true - naive) ** 2))
        if ss_naive > 0:
            skill = 1.0 - ss_model / ss_naive
        if denom and denom > 0:
            mase = float(np.mean(np.abs(y_true - y_pred)) / denom)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Skill": skill, "MASE": mase}

def _clean_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feats):
    trX = train_df[feats].replace([np.inf, -np.inf], np.nan)
    med = trX.median()
    X_tr = trX.fillna(med).values
    X_te = test_df[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
    return X_tr, X_te

def _clean_targets(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    y_tr = train_df[target].astype(float).replace([np.inf, -np.inf], np.nan)
    y_te = test_df[target].astype(float).replace([np.inf, -np.inf], np.nan)
    m_tr = y_tr.notna().values
    m_te = y_te.notna().values
    return y_tr.values[m_tr], y_te.values[m_te], m_tr, m_te

def _align_after_target_mask(X_tr, X_te, m_tr, m_te):
    return X_tr[m_tr], X_te[m_te]

def run_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    tr = train_df.copy()
    te = test_df.copy()

    feats = [c for c in tr.columns if c != target and c in te.columns]

    for c in feats:
        if tr[c].dtype == "object" or te[c].dtype == "object":
            both = pd.Categorical(pd.concat([tr[c].astype("object"), te[c].astype("object")], axis=0))
            tr[c] = pd.Categorical(tr[c].astype("object"), categories=both.categories).codes
            te[c] = pd.Categorical(te[c].astype("object"), categories=both.categories).codes

    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)

    y_tr, y_te, m_tr, m_te = _clean_targets(tr, te, target)
    X_tr_full, X_te_full = _clean_features(tr, te, feats)
    X_tr, X_te = _align_after_target_mask(X_tr_full, X_te_full, m_tr, m_te)

    if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
        raise ValueError(f"No data left after cleaning for target='{target}'")

    params = dict(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=os.cpu_count() or 4,
        tree_method="hist" if (IS_M_SERIES or IN_COLAB) else "auto",
    )
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    met = forecast_metrics(y_true=y_te, y_pred=y_pred, train_series=y_tr, compute_r2=True)

    _plot_xgb_scatter(y_te, y_pred, target)
    _plot_xgb_feature_importance(model, feats, target)

    return {
        "Target": target, "Model": "XGBoost",
        "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"],
        "Skill": met["Skill"], "MASE": met["MASE"]
    }

def _plot_xgb_scatter(y_true, y_pred, target):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.5, s=12)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.6)
    axes[0].set_title(f"XGBoost: Actual vs Predicted – {target}")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].grid(alpha=0.3)

    resid = y_true - y_pred
    axes[1].scatter(y_pred, resid, alpha=0.5, s=12)
    axes[1].axhline(0.0, color="r", linestyle="--", lw=1.6)
    axes[1].set_title(f"XGBoost: Residuals – {target}")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"xgb_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_xgb_feature_importance(model, feats, target, top_n=20):
    importances = model.feature_importances_
    n = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:n]
    plt.figure(figsize=(10, 6))
    plt.bar(range(n), importances[idx])
    plt.xticks(range(n), [feats[i] for i in idx], rotation=45, ha="right")
    plt.title(f"Feature Importance – {target}")
    plt.ylabel("importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"feature_importance_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close()

def run_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, order=(2,1,2), max_train=50000):
    # Clean series (simple interpolation/back/forward fill)
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
        return {"Target": target, "Model": "ARIMA", "MAE": None, "RMSE": None, "R2": None, "Skill": None, "MASE": None}

    if max_train and len(train) > max_train:
        train = train[-max_train:]

    try:
        model = ARIMA(train, order=order)
        fit = model.fit()
        steps = len(test)
        fc = fit.forecast(steps=steps)

        met = forecast_metrics(y_true=test, y_pred=fc, train_series=train, compute_r2=True)

        _plot_arima_series(train, test, fc, target)
        return {
            "Target": target, "Model": "ARIMA",
            "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"],
            "Skill": met["Skill"], "MASE": met["MASE"]
        }
    except Exception as e:
        print(f"[ARIMA:{target}] failed: {e}")
        _plot_arima_series(train, test, np.array([]), target)
        return {"Target": target, "Model": "ARIMA", "MAE": None, "RMSE": None, "R2": None, "Skill": None, "MASE": None}

def _plot_arima_series(train, test, fc, target):
    n = len(train)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(range(0, n), train, label("Train"), alpha=0.8)   # <-- label typo fixed below
    plt.plot(range(n, n + len(test)), test, label="Actual (test)", alpha=0.8)
    if len(fc):
        plt.plot(range(n, n + len(fc)), fc, label="Forecast", linestyle="--", lw=2)
    plt.title(f"ARIMA – {target}")
    plt.xlabel("time"); plt.ylabel(target)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"arima_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    print(f"XGB+ARIMA (Colab) → out: {OUT_DIR}")
    tr_path, te_path, mode = _find_split_csvs()
    print("paths:", mode)
    print(" train:", tr_path)
    print(" test :", te_path)

    tr = pd.read_csv(tr_path, low_memory=False)
    te = pd.read_csv(te_path, low_memory=False)
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
            r = run_xgboost(tr, te, t)
            all_rows.append(r)
            r2_str = f"{r['R2']:.3f}" if r["R2"] is not None else "NA"
            print(f"  MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R2={r2_str}  Skill={r['Skill'] if r['Skill'] is not None else 'NA'}  MASE={r['MASE'] if r['MASE'] is not None else 'NA'}")
        except Exception as e:
            print(f"[XGBoost:{t}] failed: {e}")

    for t in targets:
        print(f"\n=== ARIMA: {t} ===")
        r = run_arima(tr, te, t, order=(2,1,2), max_train=50000)
        all_rows.append(r)
        r2_str = f"{r['R2']:.3f}" if r["R2"] is not None else "NA"
        if r["MAE"] is not None:
            print(f"  MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R2={r2_str}  Skill={r['Skill'] if r['Skill'] is not None else 'NA'}  MASE={r['MASE'] if r['MASE'] is not None else 'NA'}")
        else:
            print("  (no metrics)")

    out_csv = os.path.join(OUT_DIR, "model_comparison_xgbarima.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(f"[plots] {OUT_DIR}")

if __name__ == "__main__":
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
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"arima_{target}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    main()
