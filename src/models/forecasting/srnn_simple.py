import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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


class SRNNEngineeredPipeline:
    def __init__(self):
        """Initialize the forecasting pipeline with engineered features."""
        # Fixed parameters
        self.train_data_path = DATA_DIR / 'features_engineered_train_improved.csv'
        self.test_data_path = DATA_DIR / 'features_engineered_test_improved.csv'
        self.sequence_length = 32
        self.forecast_horizon = 1  # Predict next hour
        
        # Column mappings
        self.target_columns = ['avg_latency', 'upload_bitrate', 'download_bitrate']
        self.feature_columns = None  # Will be determined dynamically
        self.selected_features = None  # Will store selected features
        
        # Feature selection parameters
        self.max_features = 15  # Maximum number of features to select
        self.feature_selection_method = 'mutual_info'  # 'mutual_info', 'f_regression', or 'random_forest'
        
        # Initialize attributes
        self.scaler = None
        self.model = None
        self.training_history = None
        self.best_params = None
        
        # Create directories
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare the engineered forecasting datasets with automatic categorical encoding."""
        print("Loading engineered forecasting datasets...")
        
        # Load training data
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Available columns: {list(train_df.columns)}")
        
        # Handle missing values - use median imputation
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median for numeric columns
        for col in train_df.select_dtypes(include=[np.number]).columns:
            if col in test_df.columns:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
        
        # Automatically detect and encode categorical columns
        categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Detected categorical columns: {categorical_cols}")
        
        # Encode categorical columns using LabelEncoder
        for col in categorical_cols:
            if col in test_df.columns:
                # Fit encoder on combined data to ensure consistent encoding
                combined_data = pd.concat([train_df[col], test_df[col]], axis=0)
                le = LabelEncoder()
                le.fit(combined_data.astype(str))
                
                # Transform both train and test
                train_df[f'{col}_encoded'] = le.transform(train_df[col].astype(str))
                test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
                
                print(f"Encoded {col} -> {col}_encoded")
        
        # Check if target columns exist and filter to available ones
        available_targets = [col for col in self.target_columns if col in train_df.columns]
        if len(available_targets) != len(self.target_columns):
            missing_targets = [col for col in self.target_columns if col not in train_df.columns]
            print(f"Warning: Missing target columns: {missing_targets}")
            self.target_columns = available_targets
        
        # Determine feature columns dynamically (all numeric features except targets)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_cols if col not in self.target_columns]
        
        print("Data loaded successfully!")
        print(f"Total features available: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns[:10]}...")  # Show first 10
        
        return train_df, test_df
    
    def select_features(self, X_train, y_train, target_idx=0):
        """Select the most important features using various methods."""
        print(f"\nPerforming feature selection for target: {self.target_columns[target_idx]}")
        print(f"Original features: {X_train.shape[1]}")
        
        # Get target for feature selection
        y_target = y_train[:, target_idx] if y_train.ndim > 1 else y_train
        
        if self.feature_selection_method == 'mutual_info':
            # Mutual information based selection
            selector = SelectKBest(score_func=mutual_info_regression, k=min(self.max_features, X_train.shape[1]))
            X_selected = selector.fit_transform(X_train, y_target)
            selected_indices = selector.get_support(indices=True)
            
        elif self.feature_selection_method == 'f_regression':
            # F-statistic based selection
            selector = SelectKBest(score_func=f_regression, k=min(self.max_features, X_train.shape[1]))
            X_selected = selector.fit_transform(X_train, y_target)
            selected_indices = selector.get_support(indices=True)
            
        elif self.feature_selection_method == 'random_forest':
            # Random Forest based feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_target)
            
            # Get feature importances and select top features
            importances = rf.feature_importances_
            selected_indices = np.argsort(importances)[-self.max_features:]
            X_selected = X_train[:, selected_indices]
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        
        # Store selected feature names
        self.selected_features = [self.feature_columns[i] for i in selected_indices]
        
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Selected feature names: {self.selected_features[:10]}...")  # Show first 10
        
        return X_selected, selected_indices
    
    def create_sequences(self, data, selected_indices):
        """Create sequences for SRNN training using simple rolling window approach."""
        X, y = [], []
        
        # Use data as-is for simple rolling window approach
        data = data.reset_index(drop=True)
        
        # Prepare features and targets using selected features only
        features = data[self.feature_columns].values[:, selected_indices].astype(np.float32)  # Use float32 to save memory
        targets = data[self.target_columns].values.astype(np.float32)  # Use float32 to save memory
        
        # Create sequences using simple sliding window
        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, train_df, test_df):
        """Preprocess data for SRNN training with feature selection."""
        print("Preprocessing data with feature selection...")
        
        # Prepare data for feature selection (use all features initially)
        X_train_full = train_df[self.feature_columns].values
        y_train_full = train_df[self.target_columns].values
        
        # Select features for each target (use first target for feature selection)
        X_train_selected, selected_indices = self.select_features(X_train_full, y_train_full, target_idx=0)
        
        # Create sequences using selected features
        X_train, y_train = self.create_sequences(train_df, selected_indices)
        X_test, y_test = self.create_sequences(test_df, selected_indices)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Test sequences: {X_test.shape}")
        
        # Save sequences as CSV before splitting
        self.save_sequences_as_csv(X_train, X_test, y_train, y_test, selected_indices)
        
        # Split training data into train/validation
        n = len(X_train)
        n_val = int(0.15 * n)  # Use 15% for validation
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        X_train, y_train = X_train[:-n_val], y_train[:-n_val]
        
        # Use MinMaxScaler for better RNN performance
        self.scaler = MinMaxScaler()
        
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale validation and test data
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        print("Data preprocessing completed!")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def save_sequences_as_csv(self, X_train, X_test, y_train, y_test, selected_indices):
        """Save concatenated X and y sequences as CSV files."""
        print("Saving sequences as CSV files...")
        
        # Get selected feature names
        selected_feature_names = [self.feature_columns[i] for i in selected_indices]
        
        # Flatten sequences for CSV format
        # X_train shape: (samples, sequence_length, features)
        # We'll flatten to (samples, sequence_length * features)
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_test = X_test.shape[0]
        
        # Create column names for flattened sequences
        sequence_columns = []
        for t in range(n_timesteps):
            for f in range(n_features):
                sequence_columns.append(f"t{t}_{selected_feature_names[f]}")
        
        # Flatten X data
        X_train_flat = X_train.reshape(n_samples_train, -1)
        X_test_flat = X_test.reshape(n_samples_test, -1)
        
        # Create DataFrames
        train_df = pd.DataFrame(X_train_flat, columns=sequence_columns)
        test_df = pd.DataFrame(X_test_flat, columns=sequence_columns)
        
        # Add target columns
        for i, target_col in enumerate(self.target_columns):
            train_df[target_col] = y_train[:, i]
            test_df[target_col] = y_test[:, i]
        
        # Save to CSV
        train_path = str(DATA_DIR / "srnn_train_sequences.csv")
        test_path = str(DATA_DIR / "srnn_test_sequences.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Training sequences saved to: {train_path}")
        print(f"Test sequences saved to: {test_path}")
        print(f"Training shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        return train_path, test_path
    
    def build_model(self, n_features, n_targets, rnn_units, dropout_rate, learning_rate):
        """Build SRNN model for forecasting."""
        model = Sequential()
        
        # Single RNN layer
        model.add(SimpleRNN(rnn_units, return_sequences=False, 
                           input_shape=(self.sequence_length, n_features)))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate * 0.5))
        
        # Output layer
        model.add(Dense(n_targets))
        
        # Compile model with MSE loss
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def hyperparameter_tuning(self, X_train, X_val, y_train, y_val, n_trials=3):
        """Perform hyperparameter tuning with simplified configurations."""
        print("Starting hyperparameter tuning with simplified configurations...")
        
        # Simplified configurations
        configurations = [
            {
                'name': 'Simple',
                'rnn_units': 48,
                'dropout_rate': 0.1,
                'batch_size': 128,
                'learning_rate': 0.001
            },
            {
                'name': 'Medium',
                'rnn_units': 64,
                'dropout_rate': 0.2,
                'batch_size': 64,
                'learning_rate': 0.001
            },
            {
                'name': 'Large',
                'rnn_units': 96,
                'dropout_rate': 0.3,
                'batch_size': 32,
                'learning_rate': 0.0005
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
                epochs=15,  # Reduced for faster tuning
                batch_size=config['batch_size'],
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True, verbose=0)],
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
            rnn_units = 48
            dropout_rate = 0.1
            batch_size = 128
            learning_rate = 0.001
        
        self.model = self.build_model(n_features, n_targets, rnn_units, dropout_rate, learning_rate)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_mae',
                patience=7,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,  # Reduced epochs
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
        
        # Plot Loss
        ax1 = axes[0, 0]
        ax1.plot(self.training_history.history['loss'], label='Training Loss', alpha=0.8)
        ax1.plot(self.training_history.history['val_loss'], label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
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
        
        # Plot RMSE (calculate from MSE)
        ax3 = axes[1, 0]
        train_rmse = np.sqrt(self.training_history.history['mse'])
        val_rmse = np.sqrt(self.training_history.history['val_mse'])
        ax3.plot(train_rmse, label='Training RMSE', alpha=0.8)
        ax3.plot(val_rmse, label='Validation RMSE', alpha=0.8)
        ax3.set_title('Root Mean Squared Error')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RMSE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature selection info
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.9, f"Feature Selection Method: {self.feature_selection_method}", 
                transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.7, f"Selected Features: {len(self.selected_features)}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f"Max Features: {self.max_features}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.3, f"Sequence Length: {self.sequence_length}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.1, f"Total Features: {len(self.feature_columns)}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Model Configuration')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / "srnn_engineered_training_history.png"), dpi=300, bbox_inches='tight')
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
            ax.set_title(f'{target_name} - Actual vs Predicted (Engineered Features)')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / "srnn_engineered_predictions.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_pipeline(self, skip_tuning=False):
        """Run the complete forecasting pipeline with engineered features.
        
        Args:
            skip_tuning: If True, skip hyperparameter tuning and use Simple config
        """
        print("Starting SRNN Forecasting Pipeline with Engineered Features...")
        print("=" * 60)
        
        # 1. Load data
        train_df, test_df = self.load_data()
        
        # 2. Preprocess data with feature selection
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(train_df, test_df)
        
        # 3. Hyperparameter tuning (optional)
        if skip_tuning:
            print("\n" + "="*60)
            print("SKIPPING HYPERPARAMETER TUNING")
            print("="*60)
            self.best_params = {
                'rnn_units': 48,
                'dropout_rate': 0.1,
                'batch_size': 128,
                'learning_rate': 0.001
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
        results_df.to_csv(str(RESULTS_DIR / "srnn_engineered_evaluation_metrics.csv"), index=False)
        
        # 8. Save per-target results for comparison script
        per_target_results = []
        if y_test.shape[1] > 1:
            for i, target_name in enumerate(self.target_columns):
                target_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                target_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
                target_r2 = r2_score(y_test[:, i], y_pred[:, i])
                per_target_results.append({
                    'target': target_name,
                    'model': 'SRNN-Engineered',
                    'mae': target_mae,
                    'rmse': target_rmse,
                    'r2': target_r2
                })
        
        per_target_df = pd.DataFrame(per_target_results)
        per_target_df.to_csv(str(MODELS_DIR / "srnnEngineeredResults.csv"), index=False)
        print(f"Per-target results saved to: {MODELS_DIR / 'srnnEngineeredResults.csv'}")
        
        # 9. Save feature selection info
        feature_info = {
            'method': self.feature_selection_method,
            'max_features': self.max_features,
            'selected_features': self.selected_features,
            'n_selected': len(self.selected_features)
        }
        
        with open(str(MODELS_DIR / "srnn_engineered_feature_info.json"), 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print("Pipeline completed successfully!")
        return metrics, y_pred


if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = SRNNEngineeredPipeline()
    
    # Skip hyperparameter tuning and use Simple configuration
    # Set skip_tuning=False to run 3-trial tuning (recommended if you have time/resources)
    metrics, predictions = pipeline.run_full_pipeline(skip_tuning=True)
