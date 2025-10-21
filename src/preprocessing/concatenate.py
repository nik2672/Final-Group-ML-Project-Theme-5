import os
import glob
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Navigate to project root, then to data folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

# Retrieve all files in the data directory (only the raw garbo files)
all_files = glob.glob(os.path.join(DATA_PATH, "*-garbo*.csv"))
print("Number of raw files:", len(all_files))

# Combine the retrieved data files with memory optimization
df_list = [pd.read_csv(file, low_memory=False) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)
del df_list  # Free memory

print(f"Initial shape: {combined_df.shape}")

# Process data by day - create unique day identifier from Year-Month-Date
if 'Year' in combined_df.columns and 'Month' in combined_df.columns and 'Date' in combined_df.columns:
    # Create unique day identifier (properly formatted as YYYY-MM-DD)
    # Handle NaN values more efficiently
    year_vals = pd.to_numeric(combined_df['Year'], errors='coerce')
    month_vals = pd.to_numeric(combined_df['Month'], errors='coerce')
    date_vals = pd.to_numeric(combined_df['Date'], errors='coerce')
    
    # Only keep rows with valid date components
    valid_mask = ~(year_vals.isna() | month_vals.isna() | date_vals.isna())
    combined_df = combined_df[valid_mask].copy()
    
    # Now create properly formatted day_id
    combined_df['day_id'] = (
        year_vals[valid_mask].astype(int).astype(str) + '-' + 
        month_vals[valid_mask].astype(int).astype(str).str.zfill(2) + '-' + 
        date_vals[valid_mask].astype(int).astype(str).str.zfill(2)
    )

    print(f"Processing data by day using Year-Month-Date")
    print(f"Shape after filtering valid dates: {combined_df.shape}")
    unique_days = sorted(combined_df['day_id'].unique())
    print(f"Number of unique days: {len(unique_days)}")
    print(f"Date range: {min(unique_days)} to {max(unique_days)}")
else:
    print("Warning: No day columns found. Processing all data as single batch.")

output_file = os.path.join(DATA_PATH, "combined_raw_data.csv")
combined_df.to_csv(output_file, index=False)
print("Final shape of the combined dataset:", combined_df.shape)
print(f"Saved to: {output_file}")