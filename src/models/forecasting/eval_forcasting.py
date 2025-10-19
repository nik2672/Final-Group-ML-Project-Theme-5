# src/models/forecasting/eval_forecasting.py
# Unified forecasting evaluation (chronological split + optional rolling-origin CV)
# Models: NaiveLast, SeasonalHourly, LinearRegression, optional XGBoost, optional ARIMA
# Metrics: MAE, RMSE, MAPE, sMAPE, R2
# Saves: CSV summary and plots into results/forecasting_eval/

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ---------- toggles ----------
ENABLE_XGBOOST = True
ENABLE_ARIMA = False          # leave off unless needed (slow)
DO_ROLLING_CV = True          # set False for a quick single split run

# ---------- paths ----------
_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "forecasting_eval")
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset():
    cand = [
        os.path.join(DATA_DIR, "features_for_forecasting.csv"),
        os.path.join(DATA_DIR, "features_engineered.csv"),
        os.path.join(DATA_DIR, "clean_data_with_imputation.csv"),
    ]
    for p in cand:
        if os.path.exists(p):
            print(f"using dataset: {p}")
            df = pd.read_csv(p, low_memory=False)
            break
    else:
        raise FileNotFoundError("no dataset found in /data. expected one of features_for_forecasting.csv, features_engineered.csv, clean_data_with_imputation.csv")

    # add a timestamp-like column if missing (best effort)
    if "timestamp" in df.columns:
        tcol = "timestamp"
    elif "time" in df.columns:
        tcol = "time"
    else:
        # fabricate an index-based chronological id
        df = df.reset_index(drop=False).rename(columns={"index": "day_id"})
        tcol = "day_id"

    # if numeric epoch-like, keep; else, just use rank order
    try:
        df = df.sort_values(tcol)
    except Exception:
        df = df.reset_index(drop=True)

    # try to build hour for seasonal baseline
    if "hour" not in df.columns:
        if "timestamp" in df.columns:
            # sometimes timestamp is epoch seconds
            try:
                dt = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
                if dt.notna().any():
                    df["hour"] = dt.dt.hour
            except Exception:
                pass

    return df

# ---------- metrics ----------
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denom))

def all_metrics(y_true, y_pred):
    return dict(
        MAE = float(mean_absolute_error(y_true, y_pred)),
        RMSE = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAPE = float(mape(y_true, y_pred)),
        sMAPE = float(smape(y_true, y_pred)),
        R2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    )

# ---------- baselines ----------
def naive_last_forecast(series, horizon=1):
    # predict each step as the last observed value
    preds = np.roll(series, 1)
    preds[0] = series[0]
    return preds

def seasonal_hourly_forecast(df, y, horizon=1):
    # if hour available, use the historical mean by hour as prediction
    if "hour" not in df.columns:
        return None
    hour_means = df.groupby("hour")[y].mean()
    preds = df["hour"].map(hour_means).values
    return preds

# ---------- feature builder ----------
def make_xy(df, y, feature_cols=None):
    # default features: numeric columns excluding the target
    if feature_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols if c != y]
    X = df[feature_cols].copy().fillna(0.0)
    yv = df[y].values.astype(float)
    return X, yv, feature_cols

# ---------- chronological split ----------
def chrono_split(n_rows, test_frac=0.2):
    split = int(n_rows * (1 - test_frac))
    idx_train = np.arange(0, split)
    idx_test = np.arange(split, n_rows)
    return idx_train, idx_test

# ---------- rolling origin splits ----------
def rolling_origin_indices(n, n_folds=3, test_frac=0.2, min_train_frac=0.4):
    folds = []
    base_split = int(n * (1 - test_frac))
    min_train = int(n * min_train_frac)
    step = max((base_split - min_train) // max(n_folds - 1, 1), 1)
    for i in range(n_folds):
        end_train = min_train + i * step
        if end_train >= base_split:
            end_train = base_split
        train_idx = np.arange(0, end_train)
        test_idx = np.arange(end_train, n)
        if len(test_idx) == 0 or len(train_idx) < 10:
            continue
        folds.append((train_idx, test_idx))
    return folds

# ---------- optional xgboost ----------
def fit_xgboost(X_tr, y_tr, X_te):
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42,
        n_jobs=os.cpu_count()
    )
    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te)

# ---------- optional arima ----------
def fit_arima(y_train, steps=50, order=(2,1,2)):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(y_train, order=order)
    fit = model.fit()
    return fit.forecast(steps=steps)

# ---------- plotting ----------
def plot_actual_pred(y_true, y_pred, title, out_png):
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="actual", lw=2, alpha=0.8)
    plt.plot(y_pred, label="pred", lw=2, alpha=0.8)
    plt.title(title)
    plt.xlabel("time steps")
    plt.ylabel("value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

def evaluate_one_target(df, yname, tag="eval", do_cv=True):
    rows = []
    n = len(df)
    if n < 50:
        print(f"skip {yname}: not enough rows")
        return rows

    # build X and y
    X, y, feats = make_xy(df, yname)

    # main chrono split
    tr_idx, te_idx = chrono_split(n, test_frac=0.2)
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # baselines
    yhat_naive = naive_last_forecast(y)
    naive_metrics = all_metrics(yte, yhat_naive[te_idx])
    rows.append(dict(target=yname, model="NaiveLast", split="chrono", **naive_metrics))

    yhat_seas = seasonal_hourly_forecast(df, yname)
    if yhat_seas is not None:
        seas_metrics = all_metrics(yte, yhat_seas[te_idx])
        rows.append(dict(target=yname, model="SeasonalHourly", split="chrono", **seas_metrics))

    # linear regression baseline (simple, fast)
    lr = LinearRegression()
    lr.fit(Xtr, ytr)
    yhat_lr = lr.predict(Xte)
    lr_metrics = all_metrics(yte, yhat_lr)
    rows.append(dict(target=yname, model="Linear", split="chrono", **lr_metrics))

    # xgboost (optional)
    if ENABLE_XGBOOST:
        yhat_xgb = fit_xgboost(Xtr, ytr, Xte)
        xgb_metrics = all_metrics(yte, yhat_xgb)
        rows.append(dict(target=yname, model="XGBoost", split="chrono", **xgb_metrics))

    # arima (optional, univariate, slow on big series)
    if ENABLE_ARIMA:
        steps = len(yte)
        yhat_ar = fit_arima(ytr, steps=steps, order=(2,1,2))
        ar_metrics = all_metrics(yte[:len(yhat_ar)], yhat_ar)
        rows.append(dict(target=yname, model="ARIMA(2,1,2)", split="chrono", **ar_metrics))

    # plot for chrono split: best non-naive model if present
    best = None
    for r in rows:
        if r["target"] == yname and r["split"] == "chrono" and r["model"] != "NaiveLast":
            if best is None or r["RMSE"] < best["RMSE"]:
                best = r
    if best is not None:
        if best["model"] == "Linear":
            yplot = yhat_lr
        elif best["model"] == "XGBoost":
            yplot = yhat_xgb
        elif best["model"].startswith("ARIMA"):
            yplot = yhat_ar
        else:
            yplot = yhat_naive[te_idx]
        plot_actual_pred(
            yte, yplot,
            f"{yname} - {best['model']} (chrono split)",
            os.path.join(OUT_DIR, f"{yname}_{tag}_chrono.png")
        )

    # rolling-origin CV
    if do_cv:
        folds = rolling_origin_indices(n, n_folds=3, test_frac=0.2, min_train_frac=0.4)
        for fi, (tri, tei) in enumerate(folds):
            Xtr2, Xte2 = X.iloc[tri], X.iloc[tei]
            ytr2, yte2 = y[tri], y[tei]

            # naive
            yhat_n2 = naive_last_forecast(y)
            rows.append(dict(target=yname, model="NaiveLast", split=f"cv{fi+1}",
                             **all_metrics(yte2, yhat_n2[tei])))

            # seasonal
            if yhat_seas is not None:
                rows.append(dict(target=yname, model="SeasonalHourly", split=f"cv{fi+1}",
                                 **all_metrics(yte2, yhat_seas[tei])))

            # linear
            lr2 = LinearRegression()
            lr2.fit(Xtr2, ytr2)
            yhat_lr2 = lr2.predict(Xte2)
            rows.append(dict(target=yname, model="Linear", split=f"cv{fi+1}",
                             **all_metrics(yte2, yhat_lr2)))

            # xgb
            if ENABLE_XGBOOST:
                yhat_x2 = fit_xgboost(Xtr2, ytr2, Xte2)
                rows.append(dict(target=yname, model="XGBoost", split=f"cv{fi+1}",
                                 **all_metrics(yte2, yhat_x2)))

            # arima
            if ENABLE_ARIMA:
                yhat_a2 = fit_arima(ytr2, steps=len(yte2), order=(2,1,2))
                rows.append(dict(target=yname, model="ARIMA(2,1,2)", split=f"cv{fi+1}",
                                 **all_metrics(yte2[:len(yhat_a2)], yhat_a2)))

    return rows

def main():
    print("=" * 68)
    print("Forecasting evaluation (chronological, plus optional rolling CV)")
    print("=" * 68)

    df = load_dataset()
    targets = [c for c in ["avg_latency", "upload_bitrate", "download_bitrate"] if c in df.columns]
    if not targets:
        # pick any numeric column as a fallback
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            targets = [num_cols[0]]
            print(f"warning: defaulting target to {targets[0]}")
        else:
            raise ValueError("no numeric targets found")

    all_rows = []
    for y in targets:
        print(f"\n--- evaluating target: {y} ---")
        rows = evaluate_one_target(df, y, tag="eval", do_cv=DO_ROLLING_CV)
        all_rows.extend(rows)

    # save summary
    df_out = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, "forecast_eval_summary.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nsaved summary: {out_csv}")

    # print compact leaderboard (chrono split only)
    print("\nLeaderboard (chrono split):")
    leaderboard = (
        df_out[df_out["split"] == "chrono"]
        .sort_values(["target", "RMSE"])
    )
    for t in leaderboard["target"].unique():
        print(f"\n  target: {t}")
        tmp = leaderboard[leaderboard["target"] == t][["model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"]]
        print(tmp.to_string(index=False))

if __name__ == "__main__":
    main()
