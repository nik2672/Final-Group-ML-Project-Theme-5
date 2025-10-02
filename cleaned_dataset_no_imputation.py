import os
import math
import pandas as pd

# config: update paths for your dataset
input_dataset_path  = r"C:\swinburne\FINAL YEAR!!!\AI FOR ENIGNEERING\new\Final-Group-ML-Project-Theme-5\data\combined_raw.csv"
output_dataset_path = r"C:\swinburne\FINAL YEAR!!!\AI FOR ENIGNEERING\new\Final-Group-ML-Project-Theme-5\data\my_clean_data_no_imputation.csv"

# invalid gps values that should be removed
bad_lat_lon_values = {0.0, 99.999, 99.9999, 999.0, 999.999}

def convert_to_float(value):
    # convert to float if possible, else nan
    if pd.isna(value):
        return math.nan
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return math.nan

def convert_size_to_mb(amount, unit_label):
    # convert transfer size to mb
    number = convert_to_float(amount)
    unit = str(unit_label).lower() if isinstance(unit_label, str) else ""
    if pd.isna(number):
        return math.nan
    if unit in ("mb", "mbyte", "mbytes"):
        return number
    if unit in ("kb", "kbyte", "kbytes"):
        return number / 1024.0
    if unit in ("gb", "gbyte", "gbytes"):
        return number * 1024.0
    return number

def convert_bitrate_to_mbps(amount, unit_label):
    # convert bitrate to mbps
    number = convert_to_float(amount)
    unit = str(unit_label).lower() if isinstance(unit_label, str) else ""
    if pd.isna(number):
        return math.nan
    if unit in ("mb/s", "mbps", "mbyte/s", "mbytes/s"):
        return number
    if unit in ("kb/s", "kbps"):
        return number / 1024.0
    if unit in ("gb/s", "gbps"):
        return number * 1024.0
    if "mbit" in unit:
        return number
    return number

def filter_invalid_gps(df):
    # drop rows with invalid gps coordinates
    mask_invalid = df["latitude"].apply(lambda x: convert_to_float(x) in bad_lat_lon_values) | \
                   df["longitude"].apply(lambda x: convert_to_float(x) in bad_lat_lon_values)
    return df.loc[~mask_invalid].copy()

def transform_to_clean_structure(raw_df):
    # create new dataframe in clean_data.csv format
    cleaned = pd.DataFrame()

    # copy over time related fields if present
    for column in ["time","Convert_time","DATES","TIME","DAY","YEAR","MONTH","DATE","HOUR","MIN","SEC"]:
        cleaned[column] = raw_df[column] if column in raw_df.columns else pd.NA

    # gps
    cleaned["latitude"]  = raw_df.get("latitude")
    cleaned["longitude"] = raw_df.get("longitude")

    # server response times
    for server in ["svr1","svr2","svr3","svr4"]:
        cleaned[server] = raw_df.get(server)

    # upload stats
    cleaned["upload_transfer_size_mbytes"] = [
        convert_size_to_mb(val, unit) for val, unit in zip(raw_df.get("Transfer size"), raw_df.get("Transfer unit"))
    ]
    cleaned["upload_bitrate_mbits/sec"] = [
        convert_bitrate_to_mbps(val, unit) for val, unit in zip(raw_df.get("Bitrate"), raw_df.get("bitrate_unit"))
    ]

    # download stats
    cleaned["download_transfer_size_rx_mbytes"] = [
        convert_size_to_mb(val, unit) for val, unit in zip(raw_df.get("Transfer size-RX"), raw_df.get("Transfer unit-RX"))
    ]
    cleaned["download_bitrate_rx_mbits/sec"] = [
        convert_bitrate_to_mbps(val, unit) for val, unit in zip(raw_df.get("Bitrate-RX"), raw_df.get("bitrate_unit-RX"))
    ]

    # app data
    if "application_data" in raw_df.columns:
        cleaned["application_data"] = raw_df["application_data"]
    else:
        cleaned["application_data"] = raw_df.get("send_data")

    # square id
    cleaned["square_id"] = raw_df.get("square_id")

    # filter gps
    cleaned = filter_invalid_gps(cleaned)

    # reorder columns to match clean_data.csv
    final_order = [
        "time","Convert_time","DATES","TIME","DAY","YEAR","MONTH","DATE","HOUR","MIN","SEC",
        "latitude","longitude","svr1","svr2","svr3","svr4",
        "upload_transfer_size_mbytes","upload_bitrate_mbits/sec",
        "download_transfer_size_rx_mbytes","download_bitrate_rx_mbits/sec",
        "application_data","square_id"
    ]
    return cleaned[final_order]

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_dataset_path), exist_ok=True)
    raw_dataset = pd.read_csv(input_dataset_path, low_memory=False)
    final_clean_dataset = transform_to_clean_structure(raw_dataset)
    final_clean_dataset.to_csv(output_dataset_path, index=False)
    print("raw input shape:", raw_dataset.shape)
    print("output clean shape:", final_clean_dataset.shape)
    print("saved file at:", output_dataset_path)
