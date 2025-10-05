import os
import glob
import pandas as pd

_HERE = _HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, 'data')

# Retrieve all files in the data directory
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
all_files = [f for f in all_files if "clean_data.csv" not in f]  # Ignore the clean_data CSV file
print("Number of raw files:", len(all_files))

# Combine the retrieved data files
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Process data by day - identify day column
day_column = None
for col in ['DAY', 'DATES', 'DATE', 'day', 'date']:
    if col in combined_df.columns:
        day_column = col
        break

if day_column:
    # Group by day and process each day separately
    print(f"Processing data by day using column: {day_column}")
    unique_days = combined_df[day_column].unique()
    print(f"Number of unique days: {len(unique_days)}")

    # Process each day and store results
    daily_results = []
    for day in unique_days:
        day_data = combined_df[combined_df[day_column] == day].copy()
        daily_results.append(day_data)
        print(f"Day {day}: {len(day_data)} records")

    # Combine all daily processed data
    combined_df = pd.concat(daily_results, ignore_index=True)
else:
    print("Warning: No day column found. Processing all data as single batch.")

output_file = os.path.join(DATA_PATH, "combined_raw.csv")
combined_df.to_csv(output_file, index=False)
print("Shape of the combined dataset:", combined_df.shape)