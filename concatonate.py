import pandas as pd
import glob
import os

#path = r"C:\swinburne\FINAL YEAR!!!\AI FOR ENIGNEERING\FinalgroupProject\data"
all_files = glob.glob(os.path.join(path, "*.csv"))
#ignoring the clean_data fiel 
all_files = [f for f in all_files if "clean_data.csv" not in f]

print("number of raw files ", len(all_files))

df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)
output_file = os.path.join(path, "combined_raw.csv")
combined_df.to_csv(output_file, index=False)

print("hape of combined dataset", combined_df.shape)
