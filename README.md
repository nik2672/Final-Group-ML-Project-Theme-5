# 5G Network Performance Analysis - Theme 5

ML project for analyzing 5G network performance using clustering and time-series forecasting.

## Project Structure

```
├── data/                          # Data files (gitignored)
├── src/
│   ├── preprocessing/            # Data preprocessing scripts
│   │   ├── concatenate.py       # Combine raw CSV files
│   │   ├── cleaned_dataset_no_imputation.py
│   │   └── cleaned_dataset_with_imputation.py
│   └── features/                # Feature engineering
│       └── feature_engineering.py
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── data_assurance.ipynb
│   └── partial_eda.ipynb
├── docs/                        # Documentation
│   └── COS40007 Design Project.pdf
├── CLAUDE.md                    # AI assistant guidance
└── README.md                    # This file
```

## Pipeline

Run scripts in this order:

1. **Data Concatenation**
   ```bash
   python src/preprocessing/concatenate.py
   ```
   Combines raw CSV files and processes by day

2. **Data Cleaning**
   ```bash
   python src/preprocessing/cleaned_dataset_no_imputation.py
   ```
   Standardizes schema, converts units, filters invalid GPS

3. **Data Imputation**
   ```bash
   python src/preprocessing/cleaned_dataset_with_imputation.py
   ```
   Fills missing values (median for numeric, "Unknown" for categorical)

4. **Feature Engineering**
   ```bash
   python src/features/feature_engineering.py
   ```
   Generates ML-ready features:
   - `features_engineered.csv` - Full feature set
   - `features_for_clustering.csv` - Zone clustering features
   - `features_for_forecasting.csv` - Time-series prediction features

## Key Questions

1. **How many groups can be categorized using 5G network performance and location?**
   - Use `features_for_clustering.csv` with K-means/DBSCAN

2. **What will network performance be in the next period?**
   - Use `features_for_forecasting.csv` with LSTM/ARIMA

## Features

### Clustering Features (9 columns)
- Location: `square_id`, `latitude`, `longitude`
- Performance: `avg_latency`, `std_latency`, `total_throughput`
- Zone aggregates: `zone_avg_latency`, `zone_avg_upload`, `zone_avg_download`

### Forecasting Features (8 columns)
- Temporal: `hour`, `day_id`, `is_peak_hours`
- Location: `square_id`
- Metrics: `avg_latency`, `upload_bitrate`, `download_bitrate`
- Lag features: Previous hour values for prediction

## Requirements

```bash
pip install pandas numpy scikit-learn scipy jupyter
```
