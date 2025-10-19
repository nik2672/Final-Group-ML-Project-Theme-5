# src/models/forecasting/forecast_eval.py
# quick n dirty forecasting eval for theme 5
# models: naive last, seasonal by hour (if we have time), linear, xgboost
# split: chronological (last 20% is test)
# metrics: mae, rmse, r2, mape (safe), smape

import os
import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# paths
_here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
data_dir = os.path.join(project_root, "data")
out_dir = os.path.join(project_root, "results", "forecasting_eval")
os.makedirs(out_dir, exist_ok=True)

data_file = os.path.join(data_dir, "features_for_forecasting.csv")

# config
test_fraction = 0.20          # last 20% for test
use_lags_for_tabular = True   # add lag1/2/4 of target to linear/xgb (helps a lot)
lag_list = [1, 2, 4]

xgb_params = dict(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="auto",
    random_state=42,
    n_jobs=0
)

targets = ["avg_latency", "upload_bitrate", "download_bitrate"]

# metrics
def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def metric_pack(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    return dict(MAE=mae, RMSE=rmse, MAPE=safe_mape(y_true, y_pred), sMAPE=smape(y_true, y_pred), R2=r2)

# helpers
def try_parse_time(df):
    # try to get a timestamp; if none, we just keep index order
    df = df.copy()
    has_ts = False

    if "timestamp" in df.columns:
        try:
            pd.to_datetime(df["timestamp"])
            has_ts = True
        except Exception:
            has_ts = False

    if not has_ts and "time" in df.columns:
        s = df["time"]
        try:
            ts = pd.to_datetime(s, unit="s", errors="coerce")  # epoch seconds
            if ts.notna().sum() >= len(df) * 0.5:
                df["timestamp"] = ts
                has_ts = True
        except Exception:
            pass

        if not has_ts:
            ts2 = pd.to_datetime(s, errors="coerce")  # generic parse
            if ts2.notna().sum() >= len(df) * 0.5:
                df["timestamp"] = ts2
                has_ts = True

    if has_ts:
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["hour"] = df["timestamp"].dt.hour
    else:
        df = df.reset_index(drop=True)

    return df, has_ts

def chrono_split_idx(n_rows, test_frac=test_fraction):
    test_n = max(1, int(n_rows * test_frac))
    return n_rows - test_n

def naive_last_forecast(train_y, test_y):
    last_val = train_y.iloc[-1]
    return pd.Series([last_val] * len(test_y), index=test_y.index)

def seasonal_hourly_forecast(df, target_col):
    # needs 'hour' in both train and test; uses train median per hour
    if "hour" not in df.columns:
        return None
    n = len(df)
    split = chrono_split_idx(n)
    train = df.iloc[:split]
    test = df.iloc[split:]
    if "hour" not in test.columns:
        return None
    med_map = train.groupby("hour")[target_col].median()
    preds = test["hour"].map(med_map).fillna(train[target_col].median())
    return preds

def build_feature_matrix(df, target_col, use_lags=True):
    df = df.copy()
    exclude = {"square_id", "day", "day_id", "timestamp", "hour", target_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.Categorical(X[c]).codes
    X = X.fillna(0)

    y = df[target_col].copy()

    if use_lags:
        for L in lag_list:
            df[f"{target_col}_lag{L}"] = y.shift(L)
        lag_cols = [f"{target_col}_lag{L}" for L in lag_list]
        X = pd.concat([X, df[lag_cols]], axis=1)

    valid = ~X.isna().any(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    return X, y

def fit_linear(x_tr, y_tr, x_te):
    m = LinearRegression()
    m.fit(x_tr, y_tr)
    return m.predict(x_te), m

def fit_xgb(x_tr, y_tr, x_te):
    m = XGBRegressor(**xgb_params)
    m.fit(x_tr, y_tr, verbose=False)
    return m.predict(x_te), m

def main():
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found")
    print(data_file)

    df = pd.read_csv(data_file, low_memory=False)
    df, has_ts = try_parse_time(df)

    avail = [t for t in targets if t in df.columns]
    if not avail:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("no numeric columns to eval on")
        avail = [num_cols[0]]

    rows = []

    for tgt in avail:
        print(f"\nevaluating target: {tgt}")

        dft = df.dropna(subset=[tgt]).copy()
        if len(dft) < 10:
            print("  not enough rows, skipping")
            continue

        n = len(dft)
        split = chrono_split_idx(n)
        train = dft.iloc[:split]
        test = dft.iloc[split:]

        # naive last
        y_pred_naive = naive_last_forecast(train[tgt], test[tgt])
        rows.append(dict(target=tgt, model="NaiveLast", **metric_pack(test[tgt], y_pred_naive)))

        # seasonal hourly (if possible)
        if has_ts and "hour" in dft.columns:
            y_pred_season = seasonal_hourly_forecast(dft, tgt)
            if y_pred_season is not None and len(y_pred_season) == len(test):
                rows.append(dict(target=tgt, model="SeasonalHourly", **metric_pack(test[tgt], y_pred_season)))

        # tabular models with optional lags
        X, y = build_feature_matrix(dft, tgt, use_lags=use_lags_for_tabular)
        if len(X) < 10:
            print("  not enough rows after feature build, skip linear/xgb")
            continue

        split2 = chrono_split_idx(len(X))
        X_tr, X_te = X.iloc[:split2], X.iloc[split2:]
        y_tr, y_te = y.iloc[:split2], y.iloc[split2:]

        # linear
        y_pred_lin, _ = fit_linear(X_tr, y_tr, X_te)
        rows.append(dict(target=tgt, model="Linear", **metric_pack(y_te, y_pred_lin)))

        # xgboost
        y_pred_xgb, _ = fit_xgb(X_tr, y_tr, X_te)
        rows.append(dict(target=tgt, model="XGBoost", **metric_pack(y_te, y_pred_xgb)))

    # writecsv
    out_csv = os.path.join(out_dir, "forecast_eval_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nsaved summary: {out_csv}")

    # tiny leaderboard per target
    if rows:
        print("\nleaderboard (chrono):\n")
        dfres = pd.DataFrame(rows)
        for tgt in dfres["target"].unique():
            sub = dfres[dfres["target"] == tgt].copy()
            sub = sub.sort_values(by=["RMSE", "MAE", "R2"], ascending=[True, True, False])
            print(f"target: {tgt}")
            print(sub[["model", "MAE", "RMSE", "MAPE", "sMAPE", "R2"]].to_string(index=False))
            print("")

if __name__ == "__main__":
    main()
