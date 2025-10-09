# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project for 5G network performance analysis using GPS-tagged latency and throughput measurements. The codebase implements a data processing pipeline that transforms raw CSV files into ML-ready datasets through concatenation, cleaning, imputation, and feature engineering.

## Project Structure

```
├── src/
│   ├── preprocessing/     # Data preprocessing pipeline
│   └── features/         # Feature engineering
├── notebooks/            # Jupyter notebooks for analysis/EDA
├── docs/                # Project documentation
└── data/               # Data files (gitignored)
```

## Data Processing Pipeline

The pipeline must be executed in this specific order:

1. **Raw data concatenation** (`src/preprocessing/concatenate.py`)
   - Combines multiple CSV files from `data/` directory
   - Creates `day_id` from Year-Month-Date columns
   - Processes data by day (as required by assignment)
   - Outputs: `data/combined_raw.csv`

2. **Data cleaning** (`src/preprocessing/cleaned_dataset_no_imputation.py`)
   - Standardizes column schema to match target format
   - Converts units (sizes to MB, bitrates to Mbps)
   - Filters invalid GPS coordinates (removes values like 0.0, 99.999, 999.0)
   - Processes by day using `day_id`
   - Input: `data/combined_raw.csv`
   - Output: `data/processed_data_no_imputation.csv`

3. **Imputation** (`src/preprocessing/cleaned_dataset_with_imputation.py`)
   - Drops completely empty columns
   - Fills numeric columns with median
   - Fills categorical columns with "Unknown"
   - Removes constant columns
   - Processes by day using `day_id`
   - Input: `data/processed_data_no_imputation.csv`
   - Output: `data/processed_data_with_imputation.csv`

4. **Feature Engineering** (`src/features/feature_engineering.py`)
   - Extracts temporal, spatial, and network performance features
   - Creates zone-level and day-level aggregations
   - Outputs 3 files optimized for different ML tasks
   - Input: `data/my_clean_data_after_assurance.csv`

## Running the Pipeline

```bash
# Step 1: Concatenate raw data files
python src/preprocessing/concatenate.py

# Step 2: Clean and standardize data
python src/preprocessing/cleaned_dataset_no_imputation.py

# Step 3: Impute missing values
python src/preprocessing/cleaned_dataset_with_imputation.py

# Step 4: Feature engineering
python src/features/feature_engineering.py

# Step 5: Quality assurance (optional - Jupyter notebook)
jupyter notebook notebooks/data_assurance.ipynb
```

## Dataset Schema

Final cleaned dataset contains these columns (in order):
- Time fields: `time`, `Convert_time`, `DATES`, `TIME`, `DAY`, `YEAR`, `MONTH`, `DATE`, `HOUR`, `MIN`, `SEC`
- GPS: `latitude`, `longitude`
- Server latency: `svr1`, `svr2`, `svr3`, `svr4`
- Upload metrics: `upload_transfer_size_mbytes`, `upload_bitrate_mbits/sec`
- Download metrics: `download_transfer_size_rx_mbytes`, `download_bitrate_rx_mbits/sec`
- Other: `application_data`, `square_id`

All numeric columns use consistent units (MB for sizes, Mbps for bitrates, milliseconds for latency).

## Key Data Quality Rules

- **Invalid GPS values**: `{0.0, 99.999, 99.9999, 999.0, 999.999}` are placeholder errors and must be filtered
- **GPS range validation**: Latitude must be in [-90, 90], longitude in [-180, 180]
- **No negative values** allowed in latency or throughput metrics
- **Duplicate handling**: Exact duplicates are data collection errors (found 109,379 in dataset) and should be removed
- **Outliers**: Rows with z-score > 3 in latency/throughput columns are flagged but retained

## Path Configuration

All scripts use absolute paths derived from `os.path.dirname(os.path.abspath(__file__))` to ensure portability. The `data/` folder is always resolved relative to the script location.

## Dependencies

- pandas
- scikit-learn (SimpleImputer)
- scipy (for data_assurance.ipynb)
- numpy
- jupyter (for notebooks)
