import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Setup device-independent paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'forecasting'
MODELS_DIR = PROJECT_ROOT / 'src' / 'models' / 'forecasting'


class SRNNForecastingPipeline:
    def __init__(self):
        """Initialize the forecasting pipeline."""
        # Fixed parameters
        self.train_data_path = DATA_DIR / 'features_for_forecasting_train.csv'
        self.test_data_path = DATA_DIR / 'features_for_forecasting_test.csv'
        self.sequence_length = 24
        self.forecast_horizon = 1
        self.target_columns = ['avg_latency', 'upload_bitrate', 'download_bitrate']
        self.feature_columns = ['hour', 'is_peak_hours', 'avg_latency_lag1', 
                               'upload_bitrate_mbits/sec_lag1', 'download_bitrate_rx_mbits/sec_lag1']
        
        # Initialize attributes
        self.scaler = None
        self.model = None
        self.training_history = None
        self.best_params = None
        
        # Create directories
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare the forecasting datasets."""
        print("Loading forecasting datasets...")
        
        # Load training data
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Handle missing values
        train_df = train_df.fillna(method='ffill').fillna(method='bfill')
        test_df = test_df.fillna(method='ffill').fillna(method='bfill')
        
        # Convert categorical variables
        train_df['day'] = pd.Categorical(train_df['day']).codes
        test_df['day'] = pd.Categorical(test_df['day']).codes
        
        print("Data loaded successfully!")
        return train_df, test_df
    
    def create_sequences(self, data):
        """Create sequences for SRNN training."""
        X, y = [], []
        
        # Group by square_id to create sequences for each location
        for square_id, group in data.groupby('square_id'):
            group = group.sort_values('hour').reset_index(drop=True)
            
            # Prepare features and targets
            features = group[self.feature_columns].values
            targets = group[self.target_columns].values
            
            # Create sequences
            for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                X.append(features[i:i + self.sequence_length])
                y.append(targets[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, train_df, test_df):
        """Preprocess data for SRNN training."""
        print("Preprocessing data...")
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_df)
        X_test, y_test = self.create_sequences(test_df)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Test sequences: {X_test.shape}")
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale validation and test data
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        print("Data preprocessing completed!")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def build_model(self, n_features, n_targets, rnn_units, dropout_rate, learning_rate):
        """Build SRNN model for forecasting."""
        model = Sequential()
        
        # First RNN layer
        model.add(SimpleRNN(rnn_units[0], return_sequences=True, 
                           input_shape=(self.sequence_length, n_features)))
        model.add(Dropout(dropout_rate))
        
        # Second RNN layer
        model.add(SimpleRNN(rnn_units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Single Dense layer
        model.add(Dense(n_targets))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val, n_trials=20):
        """Perform hyperparameter tuning using random search."""
        print("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'rnn_units': [[32, 16], [64, 32], [128, 64], [96, 48]],
            'dropout_rate': [0.2, 0.3, 0.4],
            'batch_size': [16, 32, 64],
            'learning_rate': [0.001, 0.0001]
        }
        
        best_score = float('inf')
        best_params = None
        
        # Random search
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            # Sample parameters
            params = {}
            for param, values in param_grid.items():
                if param == 'batch_size':
                    params[param] = int(np.random.choice(values))
                elif param == 'rnn_units':
                    idx = np.random.randint(0, len(values))
                    params[param] = values[idx]
                else:
                    params[param] = np.random.choice(values)
            
            # Build and train model with sampled parameters
            n_features = X_train.shape[-1]
            n_targets = y_train.shape[-1]
            
            model = self.build_model(
                n_features, n_targets,
                rnn_units=params['rnn_units'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate']
            )
            
            # Train with early stopping
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,  # Reduced for tuning
                batch_size=params['batch_size'],
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=1
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = params.copy()
                print(f"New best score: {best_score:.4f}")
        
        print(f"Best parameters found: {best_params}")
        print(f"Best validation loss: {best_score:.4f}")
        
        self.best_params = best_params
        return best_params
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the SRNN model with best parameters."""
        print("Building and training SRNN model...")
        
        n_features = X_train.shape[-1]
        n_targets = y_train.shape[-1]
        
        # Use best parameters or defaults
        if self.best_params:
            rnn_units = self.best_params['rnn_units']
            dropout_rate = self.best_params['dropout_rate']
            batch_size = self.best_params['batch_size']
            learning_rate = self.best_params['learning_rate']
        else:
            rnn_units = [64, 32]
            dropout_rate = 0.2
            batch_size = 32
            learning_rate = 0.0001
        
        self.model = self.build_model(n_features, n_targets, rnn_units, dropout_rate, learning_rate)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / 'srnn_forecasting_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Model training completed!")
        return self.training_history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model."""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        print(f"Model output shape: {y_pred.shape}")
        print(f"Target shape: {y_test.shape}")
        
        # Reshape if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        if y_test.ndim == 3:
            y_test = y_test.reshape(y_test.shape[0], -1)
        
        print(f"After reshaping - Predictions: {y_pred.shape}, Targets: {y_test.shape}")
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, y_pred
    
    def plot_training_history(self):
        """Plot training history."""
        if self.training_history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.training_history.history['loss'], label='Training Loss')
        ax1.plot(self.training_history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.training_history.history['mae'], label='Training MAE')
        ax2.plot(self.training_history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / "training_history.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, y_test, y_pred):
        """Plot actual vs predicted values."""
        # Ensure y_test and y_pred have the same shape
        if y_test.ndim == 3:
            y_test = y_test.reshape(y_test.shape[0], -1)
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        
        # Get actual number of targets from data shape
        actual_n_targets = y_test.shape[1] if y_test.ndim > 1 else 1
        n_target_columns = len(self.target_columns)
        
        # Use the minimum to avoid index errors
        n_targets = min(actual_n_targets, n_target_columns)
        
        print(f"Plotting {n_targets} targets (actual: {actual_n_targets}, columns: {n_target_columns})")
        
        fig, axes = plt.subplots(n_targets, 1, figsize=(15, 5 * n_targets))
        
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            target_name = self.target_columns[i] if i < len(self.target_columns) else f'Target_{i+1}'
            
            # Plot first 1000 points for visibility
            n_points = min(1000, len(y_test))
            ax.plot(y_test[:n_points, i], label='Actual', alpha=0.7)
            ax.plot(y_pred[:n_points, i], label='Predicted', alpha=0.7)
            ax.set_title(f'{target_name} - Actual vs Predicted')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / "predictions.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_pipeline(self):
        """Run the complete forecasting pipeline."""
        print("Starting SRNN Forecasting Pipeline...")
        print("=" * 50)
        
        # 1. Load data
        train_df, test_df = self.load_data()
        
        # 2. Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(train_df, test_df)
        
        # 3. Hyperparameter tuning
        self.hyperparameter_tuning(X_train, X_val, y_train, y_val)
        
        # 4. Train model
        self.train_model(X_train, X_val, y_train, y_val)
        
        # 5. Evaluate model
        metrics, y_pred = self.evaluate_model(X_test, y_test)
        
        # 6. Visualizations
        self.plot_training_history()
        self.plot_predictions(y_test, y_pred)
        
        # 7. Save results
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(str(RESULTS_DIR / "evaluation_metrics.csv"), index=False)
        
        print("Pipeline completed successfully!")
        return metrics, y_pred


def main():
    """Main function to run the forecasting pipeline."""
    # Initialize and run pipeline
    pipeline = SRNNForecastingPipeline()
    metrics, predictions = pipeline.run_full_pipeline()
    
    return pipeline, metrics, predictions


if __name__ == "__main__":
    pipeline, metrics, predictions = main()