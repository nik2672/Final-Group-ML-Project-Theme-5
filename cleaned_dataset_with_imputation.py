import os
import pandas as pd
from sklearn.impute import SimpleImputer

# Config paths
data_folder = os.path.join(os.path.dirname(__file__), "data")
input_filename = "my_clean_data_no_imputation.csv"
output_filename = "my_clean_data_with_imputation.csv"

input_dataset_path = os.path.join(data_folder, input_filename)
output_dataset_path = os.path.join(data_folder, output_filename)

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns that are entirely empty (all NaN)

    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after_cols = df.shape[1]

    print(f"Dropped {before_cols - after_cols} completely empty columns")

    return df

def impute_dataset(df: pd.DataFrame) -> (pd.DataFrame, int):
    # Perform imputation:
        # Median for numeric columns
        # 'Unknown' for categorical columns
    # Returns the imputed df and total number of filled values."""
    total_imputed = 0

    # Numeric columns (median imputation)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")

        before_nulls = df[numeric_cols].isna().sum().sum()
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        after_nulls = df[numeric_cols].isna().sum().sum()

        total_imputed += (before_nulls - after_nulls)

    # All categorical columns fill with 'Unknown' (only square_id for this dataset)
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    for col in categorical_cols:
        before_nulls = df[col].isna().sum()
        df[col] = df[col].fillna("Unknown")
        after_nulls = df[col].isna().sum()

        total_imputed += (before_nulls - after_nulls)

    return df, total_imputed

def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Drop columns where all values are the same (constant)
    before_cols = df.shape[1]
    df = df.loc[:, (df != df.iloc[0]).any()]
    after_cols = df.shape[1]

    print(f"Dropped {before_cols - after_cols} constant columns")

    return df

if __name__ == "__main__":
    os.makedirs(data_folder, exist_ok=True)

    # Load the previously cleaned dataset
    dataset = pd.read_csv(input_dataset_path, low_memory=False)

    # Drop completely empty columns first
    dataset = drop_empty_columns(dataset)

    # Impute missing values
    imputed_dataset, total_imputed = impute_dataset(dataset)

    # Drop constant columns
    imputed_dataset = remove_constant_columns(imputed_dataset)

    # Save result
    imputed_dataset.to_csv(output_dataset_path, index=False)

    print("Input shape:", dataset.shape)
    print("Output (with imputation) shape:", imputed_dataset.shape)
    print("Total values imputed:", total_imputed)
    print("Saved file at:", output_dataset_path)