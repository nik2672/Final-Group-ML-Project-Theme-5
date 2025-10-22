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

# Path config
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'forecasting')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load data
def load_forecasting_data():
    """Load pre-split forecasting datasets (train/test)."""
    train_path = os.path.join(DATA_PATH, 'features_for_forecasting_train.csv')
    test_path = os.path.join(DATA_PATH, 'features_for_forecasting_test.csv')

    print("\nLoading forecasting datasets:")
    print(f"  Train = {train_path}")
    print(f"  Test = {test_path}")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Training or testing CSV not found in data folder.")

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    print(f"  Train shape: {df_train.shape}")
    print(f"  Test shape : {df_test.shape}")

    return df_train, df_test

# Data preparation
def prepare_lstm_data(df_train, df_test, target_col='avg_latency', lookback=5):
    """Prepare sequential data for LSTM using given train/test splits."""
    print(f"\nPreparing LSTM data for: {target_col}")

    df_train = df_train.dropna(subset=[target_col]).fillna(0)
    df_test = df_test.dropna(subset=[target_col]).fillna(0)

    feature_cols = [c for c in df_train.columns if c != target_col and df_train[c].dtype != 'object']
    if not feature_cols:
        raise ValueError("No numeric feature columns available for LSTM.")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(df_train[feature_cols])
    y_train_scaled = y_scaler.fit_transform(df_train[[target_col]])

    X_test_scaled = x_scaler.transform(df_test[feature_cols])
    y_test_scaled = y_scaler.transform(df_test[[target_col]])

    def make_sequences(X, y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test_scaled, lookback)

    print(f"  Train sequences: {X_train_seq.shape[0]:,}, Test: {X_test_seq.shape[0]:,}")
    print(f"  Features per timestep: {X_train_seq.shape[2]}")

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, y_scaler

# LSTM training
def train_lstm(X_train, X_test, y_train, y_test, y_scaler, target_name='avg_latency'):
    print(f"\nTraining LSTM model for {target_name} ...")

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

    with tqdm(total=20, desc="  LSTM Training") as pbar:
        model.fit(
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

    print(f"\n  Results:")
    print(f"    Train  MAE={train_mae:.3f}, RMSE={train_rmse:.3f}, R²={train_r2:.3f}")
    print(f"    Test   MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R²={test_r2:.3f}")

    return model, y_true_test, y_pred_test

# Visualisation
def plot_lstm_results(y_true, y_pred, target_name='avg_latency'):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(f'LSTM Forecast - {target_name}', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_PATH, f"lstm_{target_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot → {output_path}")

# Save results
def save_model_comparison(results_dict):
    df_results = pd.DataFrame(results_dict)
    output_path = os.path.join(OUTPUT_PATH, 'model_comparison_lstm.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved metrics → {output_path}")

# Main
def main():
    print("5G Network Performance Forecasting (LSTM - Local Python)")

    df_train, df_test = load_forecasting_data()

    if len(df_train) > 500_000:
        print(f"Train dataset has {len(df_train):,} rows. Using first 500,000 sequential rows.")
        df_train = df_train.iloc[:500_000]

    targets = {
        'avg_latency': 'Avg Latency (ms)',
        'upload_bitrate': 'Upload (Mbps)',
        'download_bitrate': 'Download (Mbps)'
    }
    available_targets = {k: v for k, v in targets.items() if k in df_train.columns}

    results_summary = []
    for target_col, target_desc in available_targets.items():
        print(f"\n--- {target_desc} ---")
        X_train, X_test, y_train, y_test, y_scaler = prepare_lstm_data(df_train, df_test, target_col)
        model, y_true, y_pred = train_lstm(X_train, X_test, y_train, y_test, y_scaler, target_col)
        plot_lstm_results(y_true, y_pred, target_col)
        results_summary.append({
            'Target': target_desc,
            'Model': 'LSTM',
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        })

    save_model_comparison(results_summary)
    print("\nForecasting complete. Results in:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
