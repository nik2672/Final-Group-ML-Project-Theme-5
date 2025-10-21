import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, List

# Config paths
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load separate train and test datasets."""
    train_path = os.path.join(DATA_PATH, 'clean_data_with_imputation_train.csv')
    test_path = os.path.join(DATA_PATH, 'clean_data_with_imputation_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Train/test files not found. Please run leakage_safe_impute.py first.\n"
            f"Expected: {train_path}\n"
            f"Expected: {test_path}"
        )
    
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    
    print(f"Loaded train data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")
    
    return train_df, test_df

def extract_temporal_features(df):
    """Extract temporal features from time columns."""
    print("Extracting temporal features...")

    # Handle both uppercase and lowercase column names
    hour_col = 'HOUR' if 'HOUR' in df.columns else 'hour' if 'hour' in df.columns else None
    min_col = 'MIN' if 'MIN' in df.columns else 'min' if 'min' in df.columns else None

    # Create datetime features
    if hour_col and min_col:
        df['hour_of_day'] = df[hour_col]
        df['minute'] = df[min_col]

    # Time of day categories - FIXED: Handle edge cases
    if hour_col:
        # Ensure all values are within the bin range
        hour_values = df[hour_col].clip(lower=0, upper=23.99)
        df['time_of_day'] = pd.cut(hour_values,
                                    bins=[0, 6, 12, 18, 24],
                                    labels=['night', 'morning', 'afternoon', 'evening'],
                                    include_lowest=True)
        df['is_peak_hours'] = df[hour_col].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)

    # Day of week if available
    if 'Day' in df.columns:
        df['day_of_week'] = df['Day']
    elif 'DAY' in df.columns:
        df['day_of_week'] = df['DAY']
    elif 'day_id' in df.columns:
        df['day_of_week'] = df['day_id']

    return df

def extract_network_performance_features(df):
    """Extract aggregated network performance features."""
    print("Extracting network performance features...")

    # Latency features (svr1-svr4)
    latency_cols = ['svr1', 'svr2', 'svr3', 'svr4']
    available_latency = [col for col in latency_cols if col in df.columns]

    if available_latency:
        df['avg_latency'] = df[available_latency].mean(axis=1)
        df['min_latency'] = df[available_latency].min(axis=1)
        df['max_latency'] = df[available_latency].max(axis=1)
        df['std_latency'] = df[available_latency].std(axis=1)
        df['latency_range'] = df['max_latency'] - df['min_latency']

        # Latency quality categories - FIXED: Handle edge cases
        latency_values = df['avg_latency'].fillna(0)  # Fill NaN with 0 for binning
        df['latency_quality'] = pd.cut(latency_values,
                                       bins=[0, 50, 100, 200, float('inf')],
                                       labels=['excellent', 'good', 'fair', 'poor'],
                                       include_lowest=True)

    # Throughput features
    upload_col = 'upload_bitrate_mbits/sec'
    download_col = 'download_bitrate_rx_mbits/sec'

    if upload_col in df.columns and download_col in df.columns:
        df['total_throughput'] = df[upload_col] + df[download_col]
        df['upload_download_ratio'] = df[upload_col] / (df[download_col] + 1e-6)  # Avoid division by zero

        # Throughput quality categories - FIXED: Handle edge cases
        throughput_values = df['total_throughput'].fillna(0)  # Fill NaN with 0 for binning
        df['throughput_quality'] = pd.cut(throughput_values,
                                         bins=[0, 10, 50, 100, float('inf')],
                                         labels=['poor', 'fair', 'good', 'excellent'],
                                         include_lowest=True)

    return df

def extract_spatial_features(df):
    """Extract spatial/location features."""
    print("Extracting spatial features...")

    # GPS coordinate features
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Round coordinates to create location clusters
        df['lat_rounded'] = df['latitude'].round(3)
        df['lon_rounded'] = df['longitude'].round(3)
        df['location_hash'] = df['lat_rounded'].astype(str) + '_' + df['lon_rounded'].astype(str)

    return df

def create_zone_aggregations_improved(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-zone aggregated features using TRAIN-only statistics with missing value handling."""
    print("Creating zone-level aggregations (improved)...")

    if 'square_id' not in train_df.columns:
        print("Warning: square_id column not found. Skipping zone aggregation.")
        return train_df, test_df

    latency_cols = ['svr1', 'svr2', 'svr3', 'svr4']
    available_latency = [col for col in latency_cols if col in train_df.columns]

    # Compute statistics from TRAIN data only
    grouped_train = train_df.groupby('square_id', dropna=True)
    zone_df = grouped_train.size().rename('zone_record_count').to_frame()

    if available_latency:
        latency_mean = grouped_train[available_latency].mean()
        latency_std = grouped_train[available_latency].std()
        zone_df['zone_avg_latency'] = latency_mean.mean(axis=1)
        zone_df['zone_std_latency'] = latency_std.mean(axis=1)

    if 'upload_bitrate_mbits/sec' in train_df.columns:
        zone_df['zone_avg_upload'] = grouped_train['upload_bitrate_mbits/sec'].mean()
    if 'download_bitrate_rx_mbits/sec' in train_df.columns:
        zone_df['zone_avg_download'] = grouped_train['download_bitrate_rx_mbits/sec'].mean()

    if 'latitude' in train_df.columns and 'longitude' in train_df.columns:
        zone_df['zone_center_lat'] = grouped_train['latitude'].mean()
        zone_df['zone_center_lon'] = grouped_train['longitude'].mean()

    zone_df = zone_df.reset_index()

    # IMPROVED: Handle missing values in aggregations
    # Fill missing values with global statistics
    for col in zone_df.columns:
        if col != 'square_id' and zone_df[col].isnull().any():
            global_mean = zone_df[col].mean()
            zone_df[col] = zone_df[col].fillna(global_mean)
            print(f"Filled missing values in {col} with global mean: {global_mean:.4f}")

    # Apply same statistics to both train and test
    train_df = train_df.merge(zone_df, on='square_id', how='left')
    test_df = test_df.merge(zone_df, on='square_id', how='left')
    
    # IMPROVED: Fill missing values for test data with global statistics
    zone_cols = [col for col in zone_df.columns if col != 'square_id']
    for col in zone_cols:
        if col in test_df.columns and test_df[col].isnull().any():
            # Use train data statistics to fill test missing values
            train_mean = train_df[col].mean()
            test_df[col] = test_df[col].fillna(train_mean)
            print(f"Filled missing values in test {col} with train mean: {train_mean:.4f}")
    
    return train_df, test_df

def create_temporal_aggregations_improved(train_df: pd.DataFrame, test_df: pd.DataFrame, day_column='DAY') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-day aggregated features using TRAIN-only statistics with missing value handling."""
    print("Creating day-level aggregations (improved)...")

    if day_column not in train_df.columns:
        print(f"Warning: {day_column} column not found. Skipping day aggregation.")
        return train_df, test_df

    latency_cols = ['svr1', 'svr2', 'svr3', 'svr4']
    available_latency = [col for col in latency_cols if col in train_df.columns]

    # Compute statistics from TRAIN data only
    grouped_train = train_df.groupby(day_column, dropna=True)
    daily_df = grouped_train.size().rename('day_record_count').to_frame()

    if available_latency:
        latency_mean = grouped_train[available_latency].mean()
        latency_std = grouped_train[available_latency].std()
        daily_df['day_avg_latency'] = latency_mean.mean(axis=1)
        daily_df['day_std_latency'] = latency_std.mean(axis=1)

    if 'upload_bitrate_mbits/sec' in train_df.columns:
        daily_df['day_avg_upload'] = grouped_train['upload_bitrate_mbits/sec'].mean()
    if 'download_bitrate_rx_mbits/sec' in train_df.columns:
        daily_df['day_avg_download'] = grouped_train['download_bitrate_rx_mbits/sec'].mean()

    daily_df = daily_df.reset_index()

    # IMPROVED: Handle missing values in aggregations
    for col in daily_df.columns:
        if col != day_column and daily_df[col].isnull().any():
            global_mean = daily_df[col].mean()
            daily_df[col] = daily_df[col].fillna(global_mean)
            print(f"Filled missing values in {col} with global mean: {global_mean:.4f}")

    # Apply same statistics to both train and test
    train_df = train_df.merge(daily_df, on=day_column, how='left')
    test_df = test_df.merge(daily_df, on=day_column, how='left')
    
    # IMPROVED: Fill missing values for test data
    daily_cols = [col for col in daily_df.columns if col != day_column]
    for col in daily_cols:
        if col in test_df.columns and test_df[col].isnull().any():
            train_mean = train_df[col].mean()
            test_df[col] = test_df[col].fillna(train_mean)
            print(f"Filled missing values in test {col} with train mean: {train_mean:.4f}")
    
    return train_df, test_df

def create_lag_features_improved(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create lag features without data leakage, with missing value handling."""
    print("Creating lag features (improved)...")
    
    # Sort both datasets by time
    hour_col = 'HOUR' if 'HOUR' in train_df.columns else 'hour' if 'hour' in train_df.columns else None
    day_col = 'day_id' if 'day_id' in train_df.columns else 'DAY' if 'DAY' in train_df.columns else 'Day' if 'Day' in train_df.columns else None
    
    if not hour_col or 'square_id' not in train_df.columns:
        print("Warning: Required columns for lag features not found. Skipping lag features.")
        return train_df, test_df
    
    def add_lag_features(df):
        df_copy = df.copy()
        if day_col:
            df_sorted = df_copy.sort_values(['square_id', day_col, hour_col])
        else:
            df_sorted = df_copy.sort_values(['square_id', hour_col])
        
        # Create lag features (previous hour values) within each square_id
        for col in ['avg_latency', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']:
            if col in df_sorted.columns:
                df_sorted[f'{col}_lag1'] = df_sorted.groupby('square_id')[col].shift(1)
        
        return df_sorted
    
    # Apply lag features separately to train and test
    train_df = add_lag_features(train_df)
    test_df = add_lag_features(test_df)
    
    # IMPROVED: Handle missing values in lag features
    lag_cols = [col for col in train_df.columns if col.endswith('_lag1')]
    for col in lag_cols:
        if col in train_df.columns:
            # Fill missing values with forward fill within each group
            train_df[col] = train_df.groupby('square_id')[col].fillna(method='ffill')
            # For the first record in each group, use the mean of that group
            train_df[col] = train_df.groupby('square_id')[col].transform(
                lambda x: x.fillna(x.mean()) if not x.isnull().all() else x
            )
        
        if col in test_df.columns:
            # For test data, use train data statistics to fill missing values
            train_mean = train_df[col].mean()
            test_df[col] = test_df[col].fillna(train_mean)
    
    return train_df, test_df

def create_clustering_features(df):
    """Create specific features optimized for clustering."""
    print("Creating clustering-optimized features...")

    clustering_features = pd.DataFrame()

    # Zone identifier
    if 'square_id' in df.columns:
        clustering_features['square_id'] = df['square_id']

    # Location features
    if 'latitude' in df.columns:
        clustering_features['latitude'] = df['latitude']
    if 'longitude' in df.columns:
        clustering_features['longitude'] = df['longitude']

    # Performance metrics
    if 'avg_latency' in df.columns:
        clustering_features['avg_latency'] = df['avg_latency']
    if 'std_latency' in df.columns:
        clustering_features['std_latency'] = df['std_latency']
    if 'total_throughput' in df.columns:
        clustering_features['total_throughput'] = df['total_throughput']

    # Zone-level aggregates
    if 'zone_avg_latency' in df.columns:
        clustering_features['zone_avg_latency'] = df['zone_avg_latency']
    if 'zone_avg_upload' in df.columns:
        clustering_features['zone_avg_upload'] = df['zone_avg_upload']
    if 'zone_avg_download' in df.columns:
        clustering_features['zone_avg_download'] = df['zone_avg_download']

    return clustering_features

def create_forecasting_features(df):
    """Create specific features optimized for time-series forecasting."""
    print("Creating forecasting-optimized features...")

    forecasting_features = pd.DataFrame()

    # Temporal features (handle both cases)
    hour_col = 'HOUR' if 'HOUR' in df.columns else 'hour' if 'hour' in df.columns else None
    if hour_col:
        forecasting_features['hour'] = df[hour_col]

    if 'day_id' in df.columns:
        forecasting_features['day_id'] = df['day_id']
    elif 'DAY' in df.columns:
        forecasting_features['day'] = df['DAY']
    elif 'Day' in df.columns:
        forecasting_features['day'] = df['Day']
    if 'is_peak_hours' in df.columns:
        forecasting_features['is_peak_hours'] = df['is_peak_hours']

    # Location
    if 'square_id' in df.columns:
        forecasting_features['square_id'] = df['square_id']

    # Target variables for forecasting
    if 'avg_latency' in df.columns:
        forecasting_features['avg_latency'] = df['avg_latency']
    if 'upload_bitrate_mbits/sec' in df.columns:
        forecasting_features['upload_bitrate'] = df['upload_bitrate_mbits/sec']
    if 'download_bitrate_rx_mbits/sec' in df.columns:
        forecasting_features['download_bitrate'] = df['download_bitrate_rx_mbits/sec']

    # Lag features (already computed leakage-safe)
    for col in ['avg_latency', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']:
        lag_col = f'{col}_lag1'
        if lag_col in df.columns:
            forecasting_features[lag_col] = df[lag_col]

    return forecasting_features

def main():
    print("Starting IMPROVED leakage-safe feature engineering pipeline...")

    # Load separate train and test datasets
    train_df, test_df = load_train_test_data()

    # Extract basic features (no leakage risk)
    print("\n=== Extracting basic features ===")
    train_df = extract_temporal_features(train_df)
    test_df = extract_temporal_features(test_df)
    
    train_df = extract_network_performance_features(train_df)
    test_df = extract_network_performance_features(test_df)
    
    train_df = extract_spatial_features(train_df)
    test_df = extract_spatial_features(test_df)

    # Find day column
    day_column = 'day_id' if 'day_id' in train_df.columns else None
    if not day_column:
        for col in ['DAY', 'Day', 'DATES', 'DATE', 'Date', 'day', 'date']:
            if col in train_df.columns:
                day_column = col
                break

    # Create aggregations (improved)
    print("\n=== Creating aggregations (improved) ===")
    if day_column:
        train_df, test_df = create_temporal_aggregations_improved(train_df, test_df, day_column=day_column)
    
    train_df, test_df = create_zone_aggregations_improved(train_df, test_df)
    
    # Create lag features (improved)
    print("\n=== Creating lag features (improved) ===")
    train_df, test_df = create_lag_features_improved(train_df, test_df)

    # IMPROVED: Final missing value check and handling
    print("\n=== Final missing value handling ===")
    for df_name, df in [("train", train_df), ("test", test_df)]:
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {df_name} data has {missing_count} missing values")
            # Fill remaining missing values with median
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['object', 'category']:
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
                    else:
                        df[col] = df[col].fillna(df[col].median())
            print(f"Filled remaining missing values in {df_name} data")
        else:
            print(f"{df_name} data has no missing values")

    # Save full feature sets
    print("\n=== Saving feature sets ===")
    train_output = os.path.join(DATA_PATH, 'features_engineered_train_improved.csv')
    test_output = os.path.join(DATA_PATH, 'features_engineered_test_improved.csv')
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"Saved train features: {train_df.shape} -> {train_output}")
    print(f"Saved test features: {test_df.shape} -> {test_output}")

    # Create and save clustering-specific features
    print("\n=== Creating clustering features ===")
    train_clustering = create_clustering_features(train_df)
    test_clustering = create_clustering_features(test_df)
    
    train_clustering_output = os.path.join(DATA_PATH, 'features_for_clustering_train_improved.csv')
    test_clustering_output = os.path.join(DATA_PATH, 'features_for_clustering_test_improved.csv')
    
    train_clustering.to_csv(train_clustering_output, index=False)
    test_clustering.to_csv(test_clustering_output, index=False)
    print(f"Saved train clustering features: {train_clustering.shape} -> {train_clustering_output}")
    print(f"Saved test clustering features: {test_clustering.shape} -> {test_clustering_output}")

    # Create and save forecasting-specific features
    print("\n=== Creating forecasting features ===")
    train_forecasting = create_forecasting_features(train_df)
    test_forecasting = create_forecasting_features(test_df)
    
    train_forecasting_output = os.path.join(DATA_PATH, 'features_for_forecasting_train_improved.csv')
    test_forecasting_output = os.path.join(DATA_PATH, 'features_for_forecasting_test_improved.csv')
    
    train_forecasting.to_csv(train_forecasting_output, index=False)
    test_forecasting.to_csv(test_forecasting_output, index=False)
    print(f"Saved train forecasting features: {train_forecasting.shape} -> {train_forecasting_output}")
    print(f"Saved test forecasting features: {test_forecasting.shape} -> {test_forecasting_output}")

    print("\n=== IMPROVED Feature engineering completed successfully! ===")
    print("\nGenerated files:")
    print(f"  1. {train_output} - Train features (improved)")
    print(f"  2. {test_output} - Test features (improved)")
    print(f"  3. {train_clustering_output} - Train clustering features (improved)")
    print(f"  4. {test_clustering_output} - Test clustering features (improved)")
    print(f"  5. {train_forecasting_output} - Train forecasting features (improved)")
    print(f"  6. {test_forecasting_output} - Test forecasting features (improved)")

    print("\nFeature summary:")
    print(f"  Train features: {len(train_df.columns)}")
    print(f"  Test features: {len(test_df.columns)}")
    print(f"  Train clustering features: {len(train_clustering.columns)}")
    print(f"  Test clustering features: {len(test_clustering.columns)}")
    print(f"  Train forecasting features: {len(train_forecasting.columns)}")
    print(f"  Test forecasting features: {len(test_forecasting.columns)}")

if __name__ == "__main__":
    main()