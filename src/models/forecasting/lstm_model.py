import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# Config paths
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'forecasting')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Data preparation
def load_forecasting_data():
    """Load forecasting dataset."""
    print("\nLoading forecasting features...")
    input_path = os.path.join(DATA_PATH, 'features_for_forecasting.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Forecasting features not found at {input_path}. "
            "Please run feature engineering first."
        )

    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} samples with {len(df.columns)} features")
    return df

def prepare_lstm_data(df, target_col='avg_latency', lookback=10, train_ratio=0.8):
    """Prepare sequential, leakage-free data for LSTM forecasting."""
    print(f"\nPreparing LSTM data for: {target_col}")

    df_clean = df.dropna(subset=[target_col]).copy()
    if df_clean.empty:
        raise ValueError(f"No rows remaining after dropping NaNs for target '{target_col}'.")

    entity_col = 'square_id' if 'square_id' in df_clean.columns else None

    sort_candidates = [
        'timestamp', 'time', 'datetime', 'day_id', 'day', 'Date', 'hour', 'HOUR', 'minute', 'min', 'sec'
    ]
    sort_cols = [col for col in sort_candidates if col in df_clean.columns]

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {target_col, 'time', 'timestamp'}
    if entity_col:
        exclude_cols.add(entity_col)
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    if not feature_cols:
        raise ValueError("No suitable numeric feature columns available for LSTM after excluding identifiers.")

    groups = []
    train_feature_blocks = []
    train_target_blocks = []

    grouped = [df_clean] if entity_col is None else [
        grp for _, grp in df_clean.groupby(entity_col, dropna=False)
    ]

    for grp in grouped:
        grp_sorted = grp.sort_values(sort_cols) if sort_cols else grp.sort_index()
        if len(grp_sorted) <= lookback:
            continue

        features = grp_sorted[feature_cols].values
        targets = grp_sorted[[target_col]].values

        split_idx = int(len(grp_sorted) * train_ratio)
        split_idx = max(split_idx, lookback)
        split_idx = min(split_idx, len(grp_sorted) - 1)

        if split_idx <= lookback:
            continue

        train_feature_blocks.append(features[:split_idx])
        train_target_blocks.append(targets[:split_idx])

        groups.append({
            "features": features,
            "targets": targets,
            "split_idx": split_idx
        })

    if not groups:
        raise ValueError(
            f"Not enough data to create {lookback}-step sequences for target '{target_col}'. "
            "Consider lowering lookback or adjusting preprocessing."
        )

    X_train_rows = np.vstack(train_feature_blocks)
    y_train_rows = np.vstack(train_target_blocks)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(X_train_rows)
    y_scaler.fit(y_train_rows)

    X_train_seq, y_train_seq = [], []
    X_test_seq, y_test_seq = [], []

    for info in groups:
        features_scaled = x_scaler.transform(info["features"])
        targets_scaled = y_scaler.transform(info["targets"])
        split_idx = info["split_idx"]

        for idx in range(lookback, len(features_scaled)):
            window = features_scaled[idx - lookback:idx]
            target_value = targets_scaled[idx]
            if idx < split_idx:
                X_train_seq.append(window)
                y_train_seq.append(target_value)
            else:
                X_test_seq.append(window)
                y_test_seq.append(target_value)

    if not X_train_seq or not X_test_seq:
        raise ValueError(
            "Unable to create both training and testing sequences with the current configuration. "
            "Adjust train_ratio or lookback."
        )

    X_train = np.array(X_train_seq)
    X_test = np.array(X_test_seq)
    y_train = np.array(y_train_seq)
    y_test = np.array(y_test_seq)

    print(f"  Sequences: {X_train.shape[0] + X_test.shape[0]:,} total, lookback={lookback}")
    print(f"  Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
    print(f"  Features per timestep: {X_train.shape[2]}")

    return X_train, X_test, y_train, y_test, y_scaler

# Model training
def train_lstm(X_train, X_test, y_train, y_test, y_scaler, target_name='avg_latency'):
    """Train LSTM model."""
    print(f"\nTraining LSTM for {target_name}...")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    with tqdm(total=20, desc="  LSTM training") as pbar:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=64,
            verbose=0,
            callbacks=[early_stop]
        )
        pbar.update(20)

    # Predictions
    y_pred_train = y_scaler.inverse_transform(model.predict(X_train))
    y_pred_test = y_scaler.inverse_transform(model.predict(X_test))
    y_true_train = y_scaler.inverse_transform(y_train)
    y_true_test = y_scaler.inverse_transform(y_test)

    # Metrics
    train_mae = mean_absolute_error(y_true_train, y_pred_train)
    test_mae = mean_absolute_error(y_true_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    train_r2 = r2_score(y_true_train, y_pred_train)
    test_r2 = r2_score(y_true_test, y_pred_test)

    print(f"\n  LSTM Results:")
    print(f"    Training MAE: {train_mae:.3f}")
    print(f"    Testing MAE: {test_mae:.3f}")
    print(f"    Training RMSE: {train_rmse:.3f}")
    print(f"    Testing RMSE: {test_rmse:.3f}")
    print(f"    Training R²: {train_r2:.3f}")
    print(f"    Testing R²: {test_r2:.3f}")

    return model, history, y_true_test, y_pred_test

# Visualisation
def plot_lstm_results(y_true, y_pred, target_name='avg_latency'):
    """Plot LSTM predictions vs actual."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(f'LSTM Forecast - {target_name}', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel(target_name, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_PATH, f"lstm_{target_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Comparison saving
def save_model_comparison(results_dict):
    """Save LSTM results to CSV for comparison."""
    df_results = pd.DataFrame(results_dict)
    output_path = os.path.join(OUTPUT_PATH, 'model_comparison_lstm.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved LSTM comparison: {output_path}")

# Main
def main():
    print("5G Network Performance Forecasting (LSTM)")

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

    for target_col, target_desc in available_targets.items():
        print(f"\n--- {target_desc} ---")

        X_train, X_test, y_train, y_test, y_scaler = prepare_lstm_data(df, target_col)
        model, history, y_true, y_pred = train_lstm(X_train, X_test, y_train, y_test, y_scaler, target_col)
        plot_lstm_results(y_true, y_pred, target_col)

        results_summary.append({
            'Target': target_desc,
            'Model': 'LSTM',
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        })

    save_model_comparison(results_summary)

    print("\nLSTM Forecasting Complete")
    print(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
