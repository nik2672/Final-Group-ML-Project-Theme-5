import os
import pandas as pd

from hierarchical_imputer import (
    impute_leakage_safe,
    save_imputed_dataset,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')


def load_input_dataframe() -> pd.DataFrame:
    """Load the cleaned dataset to be imputed.

    Priority order:
    1) processed_data_no_imputation.csv (from cleaned_dataset_no_imputation.py)
    2) clean_data.csv (sample or pre-cleaned)
    """
    candidates = [
        os.path.join(DATA_PATH, 'processed_data_no_imputation.csv'),
        os.path.join(DATA_PATH, 'clean_data.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError("No input dataset found in data/. Expected one of: processed_data_no_imputation.csv, clean_data.csv")


def main() -> None:
    df = load_input_dataframe()

    # Impute with hierarchical, leakage-safe approach.
    train_imp, test_imp, full_imputed, used_numeric = impute_leakage_safe(
        df,
        date_key_col='date_key',  # will be derived if missing
        numeric_cols=None,         # auto-select numeric telemetry
        extra_exclude_numeric=None,
        test_fraction=0.2,
        enable_knn_refine=True,
        knn_neighbors=5,
    )

    # Save outputs
    out_full = os.path.join(DATA_PATH, 'clean_data_with_imputation.csv')
    out_train = os.path.join(DATA_PATH, 'clean_data_with_imputation_train.csv')
    out_test = os.path.join(DATA_PATH, 'clean_data_with_imputation_test.csv')

    save_imputed_dataset(full_imputed, out_full)
    save_imputed_dataset(train_imp, out_train)
    save_imputed_dataset(test_imp, out_test)

    print(f"Imputation complete. Numeric columns used ({len(used_numeric)}): {used_numeric}")
    print(f"Saved: {out_full}")
    print(f"Saved: {out_train}")
    print(f"Saved: {out_test}")


if __name__ == '__main__':
    main()


