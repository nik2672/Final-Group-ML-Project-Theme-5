import os
import glob
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
# Navigate to project root, then to data folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

# Retrieve all files in the data directory
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
all_files = [f for f in all_files if "clean_data.csv" not in f]  # Ignore the clean_data CSV file
print("Number of raw files:", len(all_files))

# Combine the retrieved data files
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Process data by day - create unique day identifier from Year-Month-Date
if 'Year' in combined_df.columns and 'Month' in combined_df.columns and 'Date' in combined_df.columns:
    # Create unique day identifier
    combined_df['day_id'] = combined_df['Year'].astype(str) + '-' + combined_df['Month'].astype(str).str.split('.').str[0] + '-' + combined_df['Date'].astype(str).str.split('.').str[0]

    # Clean up day_id (remove entries with nan)
    combined_df = combined_df[combined_df['day_id'].str.contains('nan') == False].copy()

    print(f"Processing data by day using Year-Month-Date")
    unique_days = sorted(combined_df['day_id'].unique())
    print(f"Number of unique days: {len(unique_days)}")

    # Process each day and store results
    daily_results = []
    for day in unique_days:
        day_data = combined_df[combined_df['day_id'] == day].copy()
        daily_results.append(day_data)
        print(f"Day {day}: {len(day_data)} records")

    # Combine all daily processed data
    combined_df = pd.concat(daily_results, ignore_index=True)
else:
    print("Warning: No day columns found. Processing all data as single batch.")

output_file = os.path.join(DATA_PATH, "combined_raw_data.csv")
combined_df.to_csv(output_file, index=False)
print("Shape of the combined dataset:", combined_df.shape)