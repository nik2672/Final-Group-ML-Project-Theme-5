import os
import pandas as pd
import numpy as np
from pathlib import Path

# Config paths
_HERE = os.path.dirname(os.path.abspath(__file__))
# Navigate to project root, then to data folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
input_dataset_path = os.path.join(DATA_PATH, 'my_clean_data_after_assurance.csv')
output_dataset_path = os.path.join(DATA_PATH, 'features_engineered.csv')

def extract_temporal_features(df):
    """Extract temporal features from time columns."""
    print("Extracting temporal features...")

    # Create datetime column if not exists
    if 'HOUR' in df.columns and 'MIN' in df.columns:
        df['hour_of_day'] = df['HOUR']
        df['minute'] = df['MIN']

    # Time of day categories
    if 'HOUR' in df.columns:
        df['time_of_day'] = pd.cut(df['HOUR'],
                                    bins=[0, 6, 12, 18, 24],
                                    labels=['night', 'morning', 'afternoon', 'evening'],
                                    include_lowest=True)
        df['is_peak_hours'] = df['HOUR'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)

    # Day of week if available (Day column contains Mon/Tue/etc)
    if 'Day' in df.columns:
        df['day_of_week'] = df['Day']
    elif 'DAY' in df.columns:
        df['day_of_week'] = df['DAY']

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

        # Latency quality categories
        df['latency_quality'] = pd.cut(df['avg_latency'],
                                       bins=[0, 50, 100, 200, float('inf')],
                                       labels=['excellent', 'good', 'fair', 'poor'])

    # Throughput features
    upload_col = 'upload_bitrate_mbits/sec'
    download_col = 'download_bitrate_rx_mbits/sec'

    if upload_col in df.columns and download_col in df.columns:
        df['total_throughput'] = df[upload_col] + df[download_col]
        df['upload_download_ratio'] = df[upload_col] / (df[download_col] + 1e-6)  # Avoid division by zero

        # Throughput quality categories
        df['throughput_quality'] = pd.cut(df['total_throughput'],
                                         bins=[0, 10, 50, 100, float('inf')],
                                         labels=['poor', 'fair', 'good', 'excellent'])

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

    # Square ID is already a zone identifier
    # We can add zone density features later

    return df

def create_zone_aggregations(df):
    """Create per-zone aggregated features."""
    print("Creating zone-level aggregations...")

    if 'square_id' not in df.columns:
        print("Warning: square_id column not found. Skipping zone aggregation.")
        return df

    # Group by zone (square_id) and calculate statistics
    zone_features = []

    latency_cols = ['svr1', 'svr2', 'svr3', 'svr4']
    available_latency = [col for col in latency_cols if col in df.columns]

    for zone in df['square_id'].unique():
        if pd.isna(zone):
            continue

        zone_data = df[df['square_id'] == zone]

        zone_stats = {
            'square_id': zone,
            'zone_record_count': len(zone_data)
        }

        # Average latency per zone
        if available_latency:
            zone_stats['zone_avg_latency'] = zone_data[available_latency].mean().mean()
            zone_stats['zone_std_latency'] = zone_data[available_latency].std().mean()

        # Average throughput per zone
        if 'upload_bitrate_mbits/sec' in df.columns:
            zone_stats['zone_avg_upload'] = zone_data['upload_bitrate_mbits/sec'].mean()
        if 'download_bitrate_rx_mbits/sec' in df.columns:
            zone_stats['zone_avg_download'] = zone_data['download_bitrate_rx_mbits/sec'].mean()

        # GPS center of zone
        if 'latitude' in df.columns and 'longitude' in df.columns:
            zone_stats['zone_center_lat'] = zone_data['latitude'].mean()
            zone_stats['zone_center_lon'] = zone_data['longitude'].mean()

        zone_features.append(zone_stats)

    # Create zone features dataframe
    zone_df = pd.DataFrame(zone_features)

    # Merge zone features back to main dataframe
    df = df.merge(zone_df, on='square_id', how='left')

    return df

def create_temporal_aggregations(df, day_column='DAY'):
    """Create per-day aggregated features."""
    print("Creating day-level aggregations...")

    if day_column not in df.columns:
        print(f"Warning: {day_column} column not found. Skipping day aggregation.")
        return df

    daily_features = []

    latency_cols = ['svr1', 'svr2', 'svr3', 'svr4']
    available_latency = [col for col in latency_cols if col in df.columns]

    for day in df[day_column].unique():
        if pd.isna(day):
            continue

        day_data = df[df[day_column] == day]

        day_stats = {
            day_column: day,
            'day_record_count': len(day_data)
        }

        # Average latency per day
        if available_latency:
            day_stats['day_avg_latency'] = day_data[available_latency].mean().mean()
            day_stats['day_std_latency'] = day_data[available_latency].std().mean()

        # Average throughput per day
        if 'upload_bitrate_mbits/sec' in df.columns:
            day_stats['day_avg_upload'] = day_data['upload_bitrate_mbits/sec'].mean()
        if 'download_bitrate_rx_mbits/sec' in df.columns:
            day_stats['day_avg_download'] = day_data['download_bitrate_rx_mbits/sec'].mean()

        daily_features.append(day_stats)

    # Create daily features dataframe
    daily_df = pd.DataFrame(daily_features)

    # Merge daily features back to main dataframe
    df = df.merge(daily_df, on=day_column, how='left')

    return df

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

    # Temporal features
    if 'hour' in df.columns:
        forecasting_features['hour'] = df['hour']
    elif 'HOUR' in df.columns:
        forecasting_features['hour'] = df['HOUR']

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

    # Historical features (lag features)
    # Sort by time first if possible
    hour_col = 'hour' if 'hour' in df.columns else 'HOUR' if 'HOUR' in df.columns else None
    day_col = 'day_id' if 'day_id' in df.columns else 'DAY' if 'DAY' in df.columns else 'Day' if 'Day' in df.columns else None

    if hour_col and 'square_id' in df.columns:
        if day_col:
            df_sorted = df.sort_values(['square_id', day_col, hour_col])
        else:
            df_sorted = df.sort_values(['square_id', hour_col])

        # Create lag features (previous hour values)
        for col in ['avg_latency', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']:
            if col in df_sorted.columns:
                forecasting_features[f'{col}_lag1'] = df_sorted.groupby('square_id')[col].shift(1)

    return forecasting_features

if __name__ == "__main__":
    print("Starting feature engineering pipeline...")

    # Check if input file exists
    if not os.path.exists(input_dataset_path):
        print(f"Error: Input file not found at {input_dataset_path}")
        print("Please ensure data preprocessing has been completed.")
        exit(1)

    # Load cleaned dataset
    print(f"Loading dataset from {input_dataset_path}...")
    df = pd.read_csv(input_dataset_path, low_memory=False)
    print(f"Loaded dataset shape: {df.shape}")

    # Extract features
    df = extract_temporal_features(df)
    df = extract_network_performance_features(df)
    df = extract_spatial_features(df)

    # Use day_id if available, otherwise try to find day column
    day_column = 'day_id' if 'day_id' in df.columns else None

    if not day_column:
        for col in ['DAY', 'Day', 'DATES', 'DATE', 'Date', 'day', 'date']:
            if col in df.columns:
                day_column = col
                break

    if day_column:
        df = create_temporal_aggregations(df, day_column=day_column)

    df = create_zone_aggregations(df)

    # Save full feature set
    print(f"Saving engineered features to {output_dataset_path}...")
    df.to_csv(output_dataset_path, index=False)
    print(f"Saved full feature set: {df.shape}")

    # Create and save clustering-specific features
    clustering_features = create_clustering_features(df)
    clustering_output = os.path.join(DATA_PATH, 'features_for_clustering.csv')
    clustering_features.to_csv(clustering_output, index=False)
    print(f"Saved clustering features: {clustering_features.shape} -> {clustering_output}")

    # Create and save forecasting-specific features
    forecasting_features = create_forecasting_features(df)
    forecasting_output = os.path.join(DATA_PATH, 'features_for_forecasting.csv')
    forecasting_features.to_csv(forecasting_output, index=False)
    print(f"Saved forecasting features: {forecasting_features.shape} -> {forecasting_output}")

    print("\nFeature engineering completed successfully!")
    print("\nGenerated files:")
    print(f"  1. {output_dataset_path} - Full feature set")
    print(f"  2. {clustering_output} - Optimized for clustering")
    print(f"  3. {forecasting_output} - Optimized for time-series forecasting")

    print("\nFeature summary:")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Clustering features: {len(clustering_features.columns)}")
    print(f"  Forecasting features: {len(forecasting_features.columns)}")
