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

def prepare_lstm_data(df, target_col='avg_latency', lookback=10):
    """Prepare sequential data for LSTM forecasting."""
    print(f"\nPreparing LSTM data for: {target_col}")

    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean = df_clean.sort_index()

    # Select numeric features
    feature_cols = [c for c in df_clean.columns if c != target_col and df_clean[c].dtype != 'object']
    if not feature_cols:
        raise ValueError("No numeric feature columns available for LSTM.")

    X_all = df_clean[feature_cols].values
    y_all = df_clean[target_col].values.reshape(-1, 1)

    # Scale data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X_all)
    y_scaled = y_scaler.fit_transform(y_all)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_scaled[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Split train/test
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"  Sequences: {X_seq.shape[0]:,} total, lookback={lookback}")
    print(f"  Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
    print(f"  Features per timestep: {X_seq.shape[2]}")

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