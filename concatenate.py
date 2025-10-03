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
output_file = os.path.join(DATA_PATH, "combined_raw.csv")
combined_df.to_csv(output_file, index=False)
print("Shape of the combined dataset:", combined_df.shape)