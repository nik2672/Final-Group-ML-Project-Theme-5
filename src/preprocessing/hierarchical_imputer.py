import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    # Optional dependency; only used if KNN refinement is enabled
    from sklearn.impute import KNNImputer  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    KNNImputer = None  # type: ignore


def derive_date_key(df: pd.DataFrame) -> pd.Series:
    """Return a deterministic per-row date key string (YYYY-MM-DD).

    Priority:
    1) 'date_key' if already present
    2) 'DATES' (already looks like YYYY-MM-DD)
    3) 'day_id' (assumed YYYY-M-D or similar)
    4) Compose from 'YEAR/Year', 'MONTH/Month', 'DATE/Date'
    5) Parse from 'Convert_time' datetime and take date
    """
    if 'date_key' in df.columns:
        return df['date_key'].astype(str)
    if 'DATES' in df.columns and df['DATES'].notna().any():
        return pd.to_datetime(df['DATES'], errors='coerce').dt.date.astype(str)
    if 'day_id' in df.columns:
        # Normalize to proper YYYY-MM-DD
        return pd.to_datetime(df['day_id'], errors='coerce').dt.date.astype(str)

    # Compose from separate parts if available
    year_col = 'YEAR' if 'YEAR' in df.columns else 'Year' if 'Year' in df.columns else None
    month_col = 'MONTH' if 'MONTH' in df.columns else 'Month' if 'Month' in df.columns else None
    date_col = 'DATE' if 'DATE' in df.columns else 'Date' if 'Date' in df.columns else None
    if year_col and month_col and date_col:
        y = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
        m = pd.to_numeric(df[month_col], errors='coerce').astype('Int64')
        d = pd.to_numeric(df[date_col], errors='coerce').astype('Int64')
        return pd.to_datetime(
            pd.DataFrame({'y': y, 'm': m, 'd': d}).astype('Int64').astype(str).agg('-'.join, axis=1),
            errors='coerce'
        ).dt.date.astype(str)

    if 'Convert_time' in df.columns:
        return pd.to_datetime(df['Convert_time'], errors='coerce').dt.date.astype(str)

    raise ValueError("Unable to derive date_key; please provide 'DATES' or 'day_id' or YEAR/MONTH/DATE columns.")


DEFAULT_EXCLUDE_NUMERIC_COLS = {
    # Identifiers / calendar / coordinates that should not be imputed
    'time', 'day_id', 'date_key', 'square_id',
    'YEAR', 'Year', 'MONTH', 'Month', 'DATE', 'Date',
    'HOUR', 'hour', 'MIN', 'min', 'SEC', 'sec',
    'latitude', 'longitude'
}


def select_numeric_telemetry_columns(df, extra_exclude):
    """Select numeric telemetry columns for imputation, excluding non-telemetry fields.

    Rules:
    - Start from all numeric dtype columns
    - Exclude known calendar/ID/coordinate columns and any user-provided exclusions
    - Keep only columns with at least one NaN (otherwise no need to impute)
    """
    exclude = set(DEFAULT_EXCLUDE_NUMERIC_COLS)
    if extra_exclude:
        exclude.update(extra_exclude)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    telemetry_cols = [c for c in numeric_cols if c not in exclude]
    needs_impute = [c for c in telemetry_cols if df[c].isna().any()]
    return needs_impute


def compute_group_medians(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
    date_key_col: str = 'date_key',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute hierarchical medians on TRAIN only.

    Returns:
    - med_l1: ['square_id', date_key] medians per numeric col
    - med_l2: ['square_id'] medians
    - med_l3: [date_key] medians
    - med_global: global medians (Series)
    """
    # Level 1: by square_id + date
    med_l1 = (
        train_df[['square_id', date_key_col] + numeric_cols]
        .groupby(['square_id', date_key_col], dropna=False)
        .median(numeric_only=True)
        .reset_index()
    )

    # Level 2: by square_id
    med_l2 = (
        train_df[['square_id'] + numeric_cols]
        .groupby(['square_id'], dropna=False)
        .median(numeric_only=True)
        .reset_index()
    )

    # Level 3: by date
    med_l3 = (
        train_df[[date_key_col] + numeric_cols]
        .groupby([date_key_col], dropna=False)
        .median(numeric_only=True)
        .reset_index()
    )

    # Global
    med_global = train_df[numeric_cols].median(numeric_only=True)

    return med_l1, med_l2, med_l3, med_global


def _fill_with_level(
    df: pd.DataFrame,
    key_cols: List[str],
    medians_df: pd.DataFrame,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """Fill NaNs using medians from a specific hierarchy level.

    Only fills rows where the target cell is NaN and a group median exists.
    """
    if not numeric_cols:
        return df

    # Merge medians in; suffix to avoid collision
    suffix = '__med'
    merged = df.merge(medians_df, on=key_cols, how='left', suffixes=('', suffix))

    for col in numeric_cols:
        med_col = f"{col}{suffix}"
        if med_col not in merged.columns:
            continue
        mask = merged[col].isna() & merged[med_col].notna()
        merged.loc[mask, col] = merged.loc[mask, med_col]
        # Drop helper column
        merged.drop(columns=[med_col], inplace=True)

    return merged


def apply_hierarchical_imputation(train_df, test_df, numeric_cols, date_key_col='date_key'):
    """Impute TRAIN/TEST deterministically using TRAIN-only medians with hierarchy.

    Order: [square_id, date] -> [square_id] -> [date] -> global median fallback.
    """
    med_l1, med_l2, med_l3, med_global = compute_group_medians(train_df, numeric_cols, date_key_col)

    def _apply(df):
        out = df.copy()
        # L1
        out = _fill_with_level(out, ['square_id', date_key_col], med_l1, numeric_cols)
        # L2
        out = _fill_with_level(out, ['square_id'], med_l2, numeric_cols)
        # L3
        out = _fill_with_level(out, [date_key_col], med_l3, numeric_cols)
        # Global
        for col in numeric_cols:
            mask = out[col].isna()
            if mask.any():
                out.loc[mask, col] = med_global[col]
        return out

    train_imputed = _apply(train_df)
    test_imputed = _apply(test_df)
    return train_imputed, test_imputed


def knn_refine_svr_block(train_df, test_df, neighbors=5):
    """Optionally refine svr1–svr4 via KNNImputer fitted on TRAIN only.

    If KNN is unavailable or preconditions are not met, returns inputs unchanged.
    """
    svr_cols = [c for c in ['svr1', 'svr2', 'svr3', 'svr4'] if c in train_df.columns]
    if not svr_cols:
        return train_df, test_df
    if KNNImputer is None:
        return train_df, test_df

    # Must have at least one row to fit
    if len(train_df) == 0:
        return train_df, test_df

    imputer = KNNImputer(n_neighbors=neighbors)

    # Fit on TRAIN svr block
    train_block = train_df[svr_cols]
    try:
        train_filled = imputer.fit_transform(train_block)
    except Exception:
        # If fit fails (e.g., all-NaN), skip deterministically
        return train_df, test_df

    train_out = train_df.copy()
    train_out[svr_cols] = train_filled

    # Transform TEST with same imputer
    test_block = test_df[svr_cols]
    try:
        test_filled = imputer.transform(test_block)
    except Exception:
        return train_out, test_df

    test_out = test_df.copy()
    test_out[svr_cols] = test_filled
    return train_out, test_out


def split_train_test_by_date(df, date_key_col='date_key', test_fraction=0.2):
    """Deterministically split TRAIN/TEST by unique date_key without leakage.

    - Sort unique dates ascending; last X% of days become TEST.
    - Ensures temporal ordering (no look-ahead).
    """
    unique_days = sorted(d for d in df[date_key_col].dropna().astype(str).unique())
    if not unique_days:
        # If no date_key, treat everything as TRAIN
        return df.copy(), df.iloc[0:0].copy()

    n_days = len(unique_days)
    n_test_days = max(1, int(round(n_days * test_fraction)))
    test_days = set(unique_days[-n_test_days:])

    train_df = df[~df[date_key_col].isin(test_days)].copy()
    test_df = df[df[date_key_col].isin(test_days)].copy()
    return train_df, test_df


def impute_leakage_safe(df, date_key_col='date_key', numeric_cols=None, extra_exclude_numeric=None, test_fraction=0.2, enable_knn_refine=True, knn_neighbors=5):
    """Top-level leakage-safe imputation pipeline.

    Returns (train_imputed, test_imputed, full_imputed_concatenated, used_numeric_cols)
    """
    df = df.copy()
    if date_key_col not in df.columns:
        df[date_key_col] = derive_date_key(df)

    # Determine numeric telemetry columns
    used_numeric_cols = numeric_cols if numeric_cols is not None else select_numeric_telemetry_columns(
        df, extra_exclude=extra_exclude_numeric
    )

    # Split first (no leakage)
    train_df, test_df = split_train_test_by_date(df, date_key_col=date_key_col, test_fraction=test_fraction)

    # Apply hierarchical fills
    train_imp, test_imp = apply_hierarchical_imputation(
        train_df, test_df, numeric_cols=used_numeric_cols, date_key_col=date_key_col
    )

    # Optional KNN refinement on svr block
    if enable_knn_refine:
        train_imp, test_imp = knn_refine_svr_block(train_imp, test_imp, neighbors=knn_neighbors)

    # Concatenate back in original date order
    full_imputed = pd.concat([train_imp, test_imp], axis=0, ignore_index=True)

    return train_imp, test_imp, full_imputed, used_numeric_cols


def save_imputed_dataset(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)