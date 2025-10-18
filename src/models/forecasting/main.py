"""
Unified Forecasting for 5G network performance prediction.
Implements XGBoost and ARIMA for network performance forecasting.
Auto-detects M-series chips for optimization.
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
import multiprocessing

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Config paths
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'forecasting')

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Detect M-series chip
N_CORES = multiprocessing.cpu_count()
IS_M_SERIES = platform.system() == 'Darwin' and platform.machine() == 'arm64'

if IS_M_SERIES:
    print(f"✓ M-Series chip detected ({N_CORES} cores) - Using optimized algorithms")
else:
    print(f"Running on standard architecture ({N_CORES} cores)")


def load_forecasting_data():
    """Load forecasting features with optional progress bar."""
    print("\nLoading forecasting features...")
    input_path = os.path.join(DATA_PATH, 'features_for_forecasting.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Forecasting features not found at {input_path}. "
            "Please run feature engineering first: python src/features/feature_engineering.py"
        )

    # M-series optimized chunked loading
    if IS_M_SERIES:
        file_size = os.path.getsize(input_path)
        chunk_size = 100000
        print(f"Reading data ({file_size / (1024**2):.1f} MB)...")

        chunks = []
        with tqdm(unit='rows', desc="Loading CSV") as pbar:
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded {len(df):,} samples with {len(df.columns)} features")
    return df


def prepare_xgboost_data(df, target_col='avg_latency'):
    """Prepare data for XGBoost forecasting."""
    print(f"\nPreparing XGBoost data for: {target_col}")

    # Remove NaN targets
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"  Dataset size: {len(df_clean):,} rows")

    # Select features
    exclude_cols = [target_col, 'square_id', 'day', 'day_id']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    X = df_clean[feature_cols].copy()

    # Encode categorical
    if IS_M_SERIES:
        for col in tqdm(X.columns, desc="  Encoding features", leave=False):
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
    else:
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

    X = X.fillna(0)
    y = df_clean[target_col]

    print(f"  Features: {X.shape[1]} columns, {X.shape[0]:,} rows")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, list(X.columns)


def train_xgboost(X_train, X_test, y_train, y_test, target_name='avg_latency'):
    """Train XGBoost model."""
    print(f"\nTraining XGBoost for {target_name}...")

    # M-series optimized configuration
    if IS_M_SERIES:
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            tree_method='hist',  # Optimized for M-series
            n_jobs=N_CORES,
            random_state=42,
            enable_categorical=True,
            max_bin=256,
        )
    else:
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            tree_method='auto',
            n_jobs=N_CORES,
            random_state=42
        )

    # Train with progress
    with tqdm(total=100, desc="  XGBoost training") as pbar:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
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

    print(f"\n  XGBoost Results:")
    print(f"    Training MAE: {train_mae:.3f}")
    print(f"    Testing MAE: {test_mae:.3f}")
    print(f"    Training RMSE: {train_rmse:.3f}")
    print(f"    Testing RMSE: {test_rmse:.3f}")
    print(f"    Training R²: {train_r2:.3f}")
    print(f"    Testing R²: {test_r2:.3f}")

    return model, y_pred_test


def plot_xgboost_results(y_test, y_pred, target_name='avg_latency'):
    """Plot XGBoost results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel(f'Actual {target_name}', fontsize=12)
    axes[0].set_ylabel(f'Predicted {target_name}', fontsize=12)
    axes[0].set_title(f'XGBoost: Actual vs Predicted', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel(f'Predicted {target_name}', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'XGBoost: Residuals', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_PATH, f'xgboost_{target_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, target_name='avg_latency', top_n=15):
    """Plot feature importance."""
    importances = model.feature_importances_
    n_features = min(top_n, len(importances))
    indices = np.argsort(importances)[::-1][:n_features]

    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance - {target_name}', fontsize=14)
    plt.bar(range(n_features), importances[indices])
    plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_PATH, f'feature_importance_{target_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def prepare_arima_data(df, target_col='avg_latency', sample_size=50000):
    """Prepare ARIMA data (with sampling for speed)."""
    print(f"\nPreparing ARIMA data for: {target_col}")

    df_clean = df.dropna(subset=[target_col]).copy()

    sort_cols = []
    if 'square_id' in df_clean.columns:
        sort_cols.append('square_id')
    for candidate in ['timestamp', 'time', 'datetime', 'day_id', 'day', 'hour']:
        if candidate in df_clean.columns and candidate not in sort_cols:
            sort_cols.append(candidate)

    if sort_cols:
        df_clean = df_clean.sort_values(sort_cols)
    else:
        df_clean = df_clean.sort_index()

    if len(df_clean) > sample_size:
        print(f"  Using the most recent {sample_size:,} rows for ARIMA...")
        df_clean = df_clean.iloc[-sample_size:]

    ts = df_clean[target_col].values
    print(f"  Time series length: {len(ts):,}")
    return ts


def train_arima(ts, target_name='avg_latency', order=(2, 1, 2), forecast_steps=50):
    """Train ARIMA model."""
    print(f"\nTraining ARIMA for {target_name}...")

    train_size = int(len(ts) * 0.8)
    train_data = ts[:train_size]
    test_data = ts[train_size:]

    # Train ARIMA
    with tqdm(total=1, desc="  ARIMA fitting") as pbar:
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        pbar.update(1)

    print(f"  AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}")

    forecast_steps = min(forecast_steps, len(test_data))
    forecast = model_fit.forecast(steps=forecast_steps)

    if len(test_data) > 0:
        test_subset = test_data[:forecast_steps]
        mae = mean_absolute_error(test_subset, forecast)
        rmse = np.sqrt(mean_squared_error(test_subset, forecast))
        print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    return model_fit, forecast, train_data, test_data


def plot_arima_results(train_data, test_data, forecast, target_name='avg_latency'):
    """Plot ARIMA forecast."""
    plt.figure(figsize=(14, 6))

    plt.plot(range(len(train_data)), train_data, label='Training', color='blue', alpha=0.7)

    test_start = len(train_data)
    test_end = test_start + len(test_data)
    plt.plot(range(test_start, test_end), test_data, label='Actual', color='green', alpha=0.7)

    forecast_start = len(train_data)
    forecast_end = forecast_start + len(forecast)
    plt.plot(range(forecast_start, forecast_end), forecast, label='Forecast',
             color='red', linestyle='--', linewidth=2)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel(target_name, fontsize=12)
    plt.title(f'ARIMA Forecast - {target_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_PATH, f'arima_{target_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_model_comparison(results_dict):
    """Save comparison results."""
    df_results = pd.DataFrame(results_dict)
    output_path = os.path.join(OUTPUT_PATH, 'model_comparison.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Saved comparison: {output_path}")


def main():
    """Main execution flow."""
    print("=" * 70)
    print("5G Network Performance Forecasting Analysis")
    if IS_M_SERIES:
        print(f"M-Series Optimized Mode ({N_CORES} cores)")
    print("=" * 70)

    # Load data
    df = load_forecasting_data()

    # Available targets
    targets = {
        'avg_latency': 'Avg Latency (ms)',
        'upload_bitrate': 'Upload (Mbps)',
        'download_bitrate': 'Download (Mbps)'
    }

    available_targets = {k: v for k, v in targets.items() if k in df.columns}

    if not available_targets:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            available_targets = {numeric_cols[0]: numeric_cols[0]}

    results_summary = []

    # XGBoost forecasting
    print("\n" + "=" * 70)
    print("XGBoost Forecasting")
    print("=" * 70)

    for target_col, target_desc in available_targets.items():
        print(f"\n--- {target_desc} ---")

        X_train, X_test, y_train, y_test, features = prepare_xgboost_data(df, target_col)
        xgb_model, y_pred = train_xgboost(X_train, X_test, y_train, y_test, target_col)

        plot_xgboost_results(y_test, y_pred, target_col)
        plot_feature_importance(xgb_model, features, target_col)

        results_summary.append({
            'Target': target_desc,
            'Model': 'XGBoost',
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        })

    # ARIMA forecasting
    print("\n" + "=" * 70)
    print("ARIMA Forecasting")
    print("=" * 70)

    for target_col, target_desc in available_targets.items():
        print(f"\n--- {target_desc} ---")

        try:
            ts = prepare_arima_data(df, target_col, sample_size=50000)
            arima_model, forecast, train_data, test_data = train_arima(ts, target_col)

            plot_arima_results(train_data, test_data, forecast, target_col)

            test_subset = test_data[:len(forecast)]
            results_summary.append({
                'Target': target_desc,
                'Model': 'ARIMA',
                'MAE': mean_absolute_error(test_subset, forecast),
                'RMSE': np.sqrt(mean_squared_error(test_subset, forecast)),
                'R2': None
            })

        except Exception as e:
            print(f"  ARIMA failed: {e}")

    # Save results
    save_model_comparison(results_summary)

    print("\n" + "=" * 70)
    print("✓ Forecasting Complete!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)

    print("\nPerformance Summary:")
    for result in results_summary:
        print(f"\n{result['Model']} - {result['Target']}:")
        print(f"  MAE: {result['MAE']:.3f}")
        print(f"  RMSE: {result['RMSE']:.3f}")
        if result['R2'] is not None:
            print(f"  R²: {result['R2']:.3f}")


if __name__ == "__main__":
    main()
