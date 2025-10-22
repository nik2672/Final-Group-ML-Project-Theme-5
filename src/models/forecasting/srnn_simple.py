import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

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
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'forecasting'
MODELS_DIR = PROJECT_ROOT / 'src' / 'models' / 'forecasting'


class SRNNForecastingPipeline:
    def __init__(self):
        """Initialize the forecasting pipeline."""
        # Fixed parameters
        self.train_data_path = DATA_DIR / 'features_for_forecasting_train_improved.csv'
        self.test_data_path = DATA_DIR / 'features_for_forecasting_test_improved.csv'
        self.sequence_length = 24  # 24 hours = 1 day of history
        self.forecast_horizon = 1  # Predict next hour
        
        # Column mappings for improved CSV
        self.target_columns = ['avg_latency', 'upload_bitrate', 'download_bitrate']
        self.feature_columns = ['hour', 'is_peak_hours', 'avg_latency_lag1', 
                               'upload_bitrate_mbits/sec_lag1', 'download_bitrate_rx_mbits/sec_lag1']
        self.day_column = 'day'
        self.square_column = 'square_id'
        
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
        print(f"Available columns: {list(train_df.columns)}")
        
        # Handle missing values
        train_df = train_df.fillna(method='ffill').fillna(method='bfill')
        test_df = test_df.fillna(method='ffill').fillna(method='bfill')
        
        # Convert categorical day to ordinal (Monday=0, ..., Sunday=6)
        day_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        train_df['day_encoded'] = train_df[self.day_column].map(day_mapping)
        test_df['day_encoded'] = test_df[self.day_column].map(day_mapping)
        
        # Fill any unmapped days with mode
        train_df['day_encoded'] = train_df['day_encoded'].fillna(train_df['day_encoded'].mode()[0])
        test_df['day_encoded'] = test_df['day_encoded'].fillna(test_df['day_encoded'].mode()[0])
        
        print("Data loaded successfully!")
        print(f"Unique squares in train: {train_df[self.square_column].nunique()}")
        print(f"Unique squares in test: {test_df[self.square_column].nunique()}")
        
        return train_df, test_df
    
    def create_sequences(self, data):
        """Create sequences for SRNN training with proper temporal ordering."""
        X, y = [], []
        
        # Group by square_id to create sequences for each location
        for square_id, group in data.groupby(self.square_column):
            # Sort by day_encoded and hour to ensure proper temporal order
            group = group.sort_values(['day_encoded', 'hour']).reset_index(drop=True)
            
            # Only process if group has enough data
            if len(group) < self.sequence_length + self.forecast_horizon:
                continue
            
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
        
        # Intermediate Dense layer with ReLU for non-linearity
        model.add(Dense(max(16, rnn_units[1] // 2), activation='relu'))
        model.add(Dropout(dropout_rate * 0.5))  # Lighter dropout for Dense
        
        # Output layer
        model.add(Dense(n_targets))
        
        # Compile model with Huber loss (robust to outliers) and multiple metrics
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='huber',  # More robust than MSE, less sensitive to outliers
            metrics=[
                'mae',      # Mean Absolute Error (interpretable)
                'mse',      # Mean Squared Error (for comparison)
                tf.keras.metrics.RootMeanSquaredError(name='rmse')  # RMSE (same units as target)
            ]
        )
        
        return model
    
    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val, n_trials=3):
        """Perform hyperparameter tuning with 3 carefully selected configurations."""
        print("Starting hyperparameter tuning with 3 optimized configurations...")
        
        # Three most promising configurations based on RNN forecasting best practices:
        # 1. Balanced: Medium capacity, moderate regularization
        # 2. Lightweight: Fast training, good for limited compute
        # 3. Deep: Higher capacity for complex patterns
        configurations = [
            {
                'name': 'Balanced',
                'rnn_units': [64, 32],
                'dropout_rate': 0.3,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            {
                'name': 'Lightweight',
                'rnn_units': [32, 16],
                'dropout_rate': 0.2,
                'batch_size': 64,
                'learning_rate': 0.001
            },
            {
                'name': 'Deep',
                'rnn_units': [128, 64],
                'dropout_rate': 0.4,
                'batch_size': 32,
                'learning_rate': 0.0001
            }
        ]
        
        best_score = float('inf')
        best_params = None
        
        n_features = X_train.shape[-1]
        n_targets = y_train.shape[-1]
        
        # Test each configuration
        for trial, config in enumerate(configurations, 1):
            print(f"\n{'='*60}")
            print(f"Trial {trial}/{n_trials}: {config['name']} Configuration")
            print(f"  RNN Units: {config['rnn_units']}")
            print(f"  Dropout: {config['dropout_rate']}")
            print(f"  Batch Size: {config['batch_size']}")
            print(f"  Learning Rate: {config['learning_rate']}")
            print(f"{'='*60}\n")
            
            # Build model
            model = self.build_model(
                n_features, n_targets,
                rnn_units=config['rnn_units'],
                dropout_rate=config['dropout_rate'],
                learning_rate=config['learning_rate']
            )
            
            # Train with early stopping (reduced epochs for tuning)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,  # Reduced for faster tuning
                batch_size=config['batch_size'],
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True, verbose=0)],
                verbose=1
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            val_mae = min(history.history['val_mae'])
            
            print(f"\n{config['name']} Results:")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation MAE: {val_mae:.4f}")
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = {k: v for k, v in config.items() if k != 'name'}
                best_config_name = config['name']
                print(f"  ✓ New best configuration!")
        
        print(f"\n{'='*60}")
        print(f"Best Configuration: {best_config_name}")
        print(f"Best Validation Loss: {best_score:.4f}")
        print(f"Parameters: {best_params}")
        print(f"{'='*60}\n")
        
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
                monitor='val_mae',  # Monitor MAE (more interpretable than Huber loss)
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(MODELS_DIR / 'srnn_forecasting_model.h5'),
                monitor='val_mae',  # Save best model based on MAE
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
        """Evaluate the trained model with comprehensive metrics."""
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
        
        # Calculate overall metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        print("\n" + "="*60)
        print("Overall Test Set Evaluation:")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric:10s}: {value:.4f}")
        
        # Per-target metrics (if multi-target)
        if y_test.shape[1] > 1:
            print("\n" + "="*60)
            print("Per-Target Metrics:")
            print("="*60)
            for i, target_name in enumerate(self.target_columns):
                target_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                target_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                target_r2 = r2_score(y_test[:, i], y_pred[:, i])
                print(f"\n{target_name}:")
                print(f"  MAE:  {target_mae:.4f}")
                print(f"  RMSE: {target_rmse:.4f}")
                print(f"  R²:   {target_r2:.4f}")
        
        return metrics, y_pred
    
    def plot_training_history(self):
        """Plot training history with multiple metrics."""
        if self.training_history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot Huber Loss
        ax1 = axes[0, 0]
        ax1.plot(self.training_history.history['loss'], label='Training Loss (Huber)', alpha=0.8)
        ax1.plot(self.training_history.history['val_loss'], label='Validation Loss (Huber)', alpha=0.8)
        ax1.set_title('Model Loss (Huber)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Huber Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2 = axes[0, 1]
        ax2.plot(self.training_history.history['mae'], label='Training MAE', alpha=0.8)
        ax2.plot(self.training_history.history['val_mae'], label='Validation MAE', alpha=0.8)
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot RMSE
        ax3 = axes[1, 0]
        ax3.plot(self.training_history.history['rmse'], label='Training RMSE', alpha=0.8)
        ax3.plot(self.training_history.history['val_rmse'], label='Validation RMSE', alpha=0.8)
        ax3.set_title('Root Mean Squared Error')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot MSE
        ax4 = axes[1, 1]
        ax4.plot(self.training_history.history['mse'], label='Training MSE', alpha=0.8)
        ax4.plot(self.training_history.history['val_mse'], label='Validation MSE', alpha=0.8)
        ax4.set_title('Mean Squared Error')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MSE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
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
    
    def run_full_pipeline(self, skip_tuning=False):
        """Run the complete forecasting pipeline.
        
        Args:
            skip_tuning: If True, skip hyperparameter tuning and use Balanced config
        """
        print("Starting SRNN Forecasting Pipeline...")
        print("=" * 50)
        
        # 1. Load data
        train_df, test_df = self.load_data()
        
        # 2. Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(train_df, test_df)
        
        # 3. Hyperparameter tuning (optional)
        if skip_tuning:
            print("\n" + "="*60)
            print("SKIPPING HYPERPARAMETER TUNING")
            print("="*60)
            self.best_params = {
                'rnn_units': [128, 64],
                'dropout_rate': 0.4,
                'batch_size': 32,
                'learning_rate': 0.0001
            }
            print(f"\nConfiguration:")
            print(f"  RNN Units: {self.best_params['rnn_units']}")
            print(f"  Dropout: {self.best_params['dropout_rate']}")
            print(f"  Batch Size: {self.best_params['batch_size']}")
            print(f"  Learning Rate: {self.best_params['learning_rate']}")
            print("="*60 + "\n")
        else:
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
    
    # Skip hyperparameter tuning and use Balanced configuration
    # Set skip_tuning=False to run 3-trial tuning (recommended if you have time/resources)
    metrics, predictions = pipeline.run_full_pipeline(skip_tuning=True)
    
    return pipeline, metrics, predictions


if __name__ == "__main__":
    pipeline, metrics, predictions = main()