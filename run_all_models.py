#!/usr/bin/env python3
"""
Master script to run all ML models for 5G network performance analysis.
Executes clustering and forecasting models in sequence.
"""

import os
import sys
import subprocess
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CLUSTERING_SCRIPT = PROJECT_ROOT / 'src' / 'models' / 'clustering' / 'main.py'
FORECASTING_SCRIPT = PROJECT_ROOT / 'src' / 'models' / 'forecasting' / 'main.py'
FEATURE_ENGINEERING_SCRIPT = PROJECT_ROOT / 'src' / 'features' / 'feature_engineering.py'

def check_dependencies():
    """Check if required data files exist."""
    data_path = PROJECT_ROOT / 'data'
    clustering_features = data_path / 'features_for_clustering.csv'
    forecasting_features = data_path / 'features_for_forecasting.csv'

    if not clustering_features.exists() or not forecasting_features.exists():
        print("=" * 70)
        print("WARNING: Feature files not found!")
        print("=" * 70)
        print("\nRequired files:")
        print(f"  - {clustering_features}")
        print(f"  - {forecasting_features}")
        print("\nRunning feature engineering first...")
        print("-" * 70)

        if not FEATURE_ENGINEERING_SCRIPT.exists():
            print(f"ERROR: Feature engineering script not found at {FEATURE_ENGINEERING_SCRIPT}")
            sys.exit(1)

        # Run feature engineering
        result = subprocess.run([sys.executable, str(FEATURE_ENGINEERING_SCRIPT)])
        if result.returncode != 0:
            print("ERROR: Feature engineering failed!")
            sys.exit(1)
        print("-" * 70)
        print("Feature engineering completed successfully!\n")

def run_script(script_path, script_name):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"Running {script_name}...")
    print("=" * 70)

    if not script_path.exists():
        print(f"ERROR: Script not found at {script_path}")
        return False

    result = subprocess.run([sys.executable, str(script_path)])

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with exit code {result.returncode}")
        return False

    print(f"\n{script_name} completed successfully!")
    return True

def main():
    """Main execution flow."""
    print("=" * 70)
    print("5G Network Performance ML Analysis - Full Pipeline")
    print("=" * 70)

    # Check dependencies
    check_dependencies()

    # Run clustering models
    success = run_script(CLUSTERING_SCRIPT, "Clustering Models (K-Means & DBSCAN)")
    if not success:
        print("\nClustering failed. Continuing with forecasting...")

    # Run forecasting models
    success = run_script(FORECASTING_SCRIPT, "Forecasting Models (XGBoost & ARIMA)")
    if not success:
        print("\nForecasting failed.")

    # Summary
    print("\n" + "=" * 70)
    print("ML Analysis Complete!")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - results/clustering/")
    print("  - results/forecasting/")
    print("\nNext steps:")
    print("  1. Review visualizations in results directories")
    print("  2. Analyze cluster assignments and forecasts")
    print("  3. Tune model parameters if needed (see src/models/README.md)")
    print("=" * 70)

if __name__ == "__main__":
    main()
