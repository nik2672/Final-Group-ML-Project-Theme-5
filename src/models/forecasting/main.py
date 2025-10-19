"""
Unified Forecasting for 5G network performance prediction.
Implements XGBoost and ARIMA for network performance forecasting.
Uses a chronological split for fair time-series evaluation.
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
import multiprocessing

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# -------- Paths --------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'forecasting')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -------- System info --------
N_CORES = multiprocessing.cpu_count()
IS_M_SERIES = platform.system() == 'Darwin' and platform.machine() == 'arm64'

if IS_M_SERIES:
    print(f"M-Series chip detected ({N_CORES} cores) - Using optimized algorithms")
else:
    print(f"Running on standard architecture ({N_CORES} cores)")


def load_forecasting_data():
    """Load forecasting features CSV."""
    print("\nLoading forecasting features...")
    input_path = os.path.join(DATA_PATH, 'features_for_forecasting.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Forecasting features not found at {input_path}. "
            "Please run feature engineering first: python src/features/feature_engineering.py"
        )

    if IS_M_SERIES:
        file_size = os.path.getsize(input_path)
        chunk_size = 100000
        print(f"Reading data (~{file_size / (1024**2):.1f} MB)...")
        chunks = []
        with tqdm(unit='rows', desc="Loading CSV") as pbar:
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
    return df


def prepare_xgboost_data(df, target_col='avg_latency'):
    """
    Prepare data for XGBoost forecasting using a chronological split.
    - Sort by available time columns (timestamp/time/day_id/day)
    - Encode categorical features
    - 80/20 time-based split (no shuffling)
    """
    print(f"\nPreparing XGBoost data for: {target_col}")

    # Remove NaN targets
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"  Rows with target: {len(df_clean):,}")

    # Build sorting keys (use what's available)
    sort_cols = []
    if 'square_id' in df_clean.columns:
        sort_cols.append('square_id')  # optional grouping if present

    for cand in ['timestamp', 'time', 'datetime', 'day_id', 'day', 'hour']:
        if cand in df_clean.columns and cand not in sort_cols:
            sort_cols.append(cand)

    if sort_cols:
        df_clean = df_clean.sort_values(sort_cols)
    else:
        # fallback: keep current order
        df_clean = df_clean.reset_index(drop=True)

    # Feature columns: everything except the target and pure IDs we don't want to leak
    exclude = {target_col, 'square_id', 'day', 'day_id'}
    feature_cols = [c for c in df_clean.columns if c not in exclude]

    X = df_clean[feature_cols].copy()

    # Encode categoricals
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes

    X = X.fillna(0)
    y = df_clean[target_col].astype(float)

    print(f"  Features: {X.shape[1]} | Rows: {X.shape[0]:,}")

    # Chronological split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Chrono split -> Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, list(X.columns)


def train_xgboost(X_train, X_test, y_train, y_test, target_name='avg_latency'):
    """Train XGBoost regressor and report metrics."""
    print(f"\nTraining XGBoost for {target_name}...")

    params = dict(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=N_CORES
    )

    if IS_M_SERIES:
        params.update(dict(tree_method='hist', enable_categorical=True, max_bin=256))
    else:
        params.update(dict(tree_method='auto'))

    model = XGBRegressor(**params)

    with tqdm(total=100, desc="  XGBoost training") as pbar:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        pbar.update(100)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("  XGBoost Results:")
    print(f"    Train MAE:  {train_mae:.3f}")
    print(f"    Test  MAE:  {test_mae:.3f}")
    print(f"    Train RMSE: {train_rmse:.3f}")
    print(f"    Test  RMSE: {test_rmse:.3f}")
    print(f"    Train R2:   {train_r2:.3f}")
    print(f"    Test  R2:   {test_r2:.3f}")

    return model, y_pred_test


def plot_xgboost_results(y_test, y_pred, target_name='avg_latency'):
    """Plot Actual vs Predicted and Residuals for XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], 'r--', lw=2)
    axes[0].set_xlabel(f'Actual {target_name}')
    axes[0].set_ylabel(f'Predicted {target_name}')
    axes[0].set_title('XGBoost: Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel(f'Predicted {target_name}')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('XGBoost: Residuals')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_PATH, f'xgboost_{target_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, target_name='avg_latency', top_n=15):
    """Plot XGBoost feature importance."""
    importances = model.feature_importances_
    n = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:n]

    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance - {target_name}')
    plt.bar(range(n), importances[idx])
    plt.xticks(range(n), [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()

    out = os.path.join(OUTPUT_PATH, f'feature_importance_{target_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


def prepare_arima_data(df, target_col='avg_latency', sample_size=50000):
    """
    Prepare a univariate time series for ARIMA:
    - sort by available time columns
    - optionally sample the most recent N rows for speed
    """
    print(f"\nPreparing ARIMA data for: {target_col}")
    df_clean = df.dropna(subset=[target_col]).copy()

    sort_cols = []
    if 'square_id' in df_clean.columns:
        sort_cols.append('square_id')
    for cand in ['timestamp', 'time', 'datetime', 'day_id', 'day', 'hour']:
        if cand in df_clean.columns and cand not in sort_cols:
            sort_cols.append(cand)

    if sort_cols:
        df_clean = df_clean.sort_values(sort_cols)
    else:
        df_clean = df_clean.sort_index()

    if len(df_clean) > sample_size:
        print(f"  Using most recent {sample_size:,} rows for ARIMA")
        df_clean = df_clean.iloc[-sample_size:]

    ts = df_clean[target_col].astype(float).values
    print(f"  Time series length: {len(ts):,}")
    return ts


def train_arima(ts, target_name='avg_latency', order=(2, 1, 2), forecast_steps=50):
    """Fit ARIMA on the training portion and forecast into the test horizon."""
    print(f"\nTraining ARIMA for {target_name}...")

    train_size = int(len(ts) * 0.8)
    train_data = ts[:train_size]
    test_data = ts[train_size:]

    with tqdm(total=1, desc="  ARIMA fitting") as pbar:
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        pbar.update(1)

    print(f"  AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}")

    steps = min(forecast_steps, len(test_data)) if len(test_data) else 0
    forecast = model_fit.forecast(steps=steps) if steps > 0 else np.array([])

    if steps > 0:
        test_subset = test_data[:steps]
        mae = mean_absolute_error(test_subset, forecast)
        rmse = np.sqrt(mean_squared_error(test_subset, forecast))
        print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    return model_fit, forecast, train_data, test_data


def plot_arima_results(train_data, test_data, forecast, target_name='avg_latency'):
    """Plot ARIMA forecast vs actual."""
    plt.figure(figsize=(14, 6))

    plt.plot(range(len(train_data)), train_data, label='Training', color='blue', alpha=0.7)

    test_start = len(train_data)
    test_end = test_start + len(test_data)
    plt.plot(range(test_start, test_end), test_data, label='Actual', color='green', alpha=0.7)

    if len(forecast) > 0:
        forecast_start = len(train_data)
        forecast_end = forecast_start + len(forecast)
        plt.plot(range(forecast_start, forecast_end), forecast, label='Forecast',
                 color='red', linestyle='--', linewidth=2)

    plt.xlabel('Time Steps')
    plt.ylabel(target_name)
    plt.title(f'ARIMA Forecast - {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_PATH, f'arima_{target_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


def save_model_comparison(rows):
    """Save aggregated comparison metrics to CSV."""
    df_results = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_PATH, 'model_comparison.csv')
    df_results.to_csv(out, index=False)
    print(f"\nSaved comparison: {out}")


def main():
    print("=" * 70)
    print("5G Network Performance Forecasting Analysis")
    if IS_M_SERIES:
        print(f"M-Series Optimized Mode ({N_CORES} cores)")
    print("=" * 70)

    df = load_forecasting_data()

    targets = {
        'avg_latency': 'Avg Latency (ms)',
        'upload_bitrate': 'Upload (Mbps)',
        'download_bitrate': 'Download (Mbps)'
    }
    available = {k: v for k, v in targets.items() if k in df.columns}
    if not available:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            available = {num_cols[0]: num_cols[0]}

    results = []

    print("\n" + "=" * 70)
    print("XGBoost Forecasting (chronological split)")
    print("=" * 70)

    for target_col, target_desc in available.items():
        print(f"\n--- {target_desc} ---")
        X_train, X_test, y_train, y_test, feats = prepare_xgboost_data(df, target_col)
        xgb_model, y_pred = train_xgboost(X_train, X_test, y_train, y_test, target_col)
        plot_xgboost_results(y_test, y_pred, target_col)
        plot_feature_importance(xgb_model, feats, target_col)

        results.append({
            'Target': target_desc,
            'Model': 'XGBoost',
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        })

    print("\n" + "=" * 70)
    print("ARIMA Forecasting")
    print("=" * 70)

    for target_col, target_desc in available.items():
        print(f"\n--- {target_desc} ---")
        try:
            ts = prepare_arima_data(df, target_col, sample_size=50000)
            arima_model, forecast, train_data, test_data = train_arima(ts, target_col)

            plot_arima_results(train_data, test_data, forecast, target_col)

            if len(forecast) > 0:
                test_subset = test_data[:len(forecast)]
                results.append({
                    'Target': target_desc,
                    'Model': 'ARIMA',
                    'MAE': mean_absolute_error(test_subset, forecast),
                    'RMSE': np.sqrt(mean_squared_error(test_subset, forecast)),
                    'R2': None
                })
            else:
                results.append({
                    'Target': target_desc,
                    'Model': 'ARIMA',
                    'MAE': None,
                    'RMSE': None,
                    'R2': None
                })

        except Exception as e:
            print(f"  ARIMA failed: {e}")

    save_model_comparison(results)

    print("\n" + "=" * 70)
    print("Forecasting Complete")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)

    print("\nPerformance Summary:")
    for r in results:
        print(f"\n{r['Model']} - {r['Target']}:")
        if r['MAE'] is not None:
            print(f"  MAE: {r['MAE']:.3f}")
            print(f"  RMSE: {r['RMSE']:.3f}")
        else:
            print("  (no metrics)")
        if r['R2'] is not None:
            print(f"  R2: {r['R2']:.3f}")


if __name__ == "__main__":
    main()
