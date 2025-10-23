# compare_models_from_csvs.py
# this script compares gru, lstm, and xgboost metrics from three csv files (arima excluded)
# it unifies headers and target labels, computes mse, makes per-target 2x2 plots,
# and writes a summary csv showing which model wins each metric.

import os                    # filesystem paths and checks
import math                  # small math helpers (floor/ceil)
import argparse              # command-line flags
import numpy as np           # arrays / nans
import pandas as pd          # csv i/o and data wrangling
import matplotlib.pyplot as plt  # plotting
from pandas.api.types import CategoricalDtype  # ordered categories for legend order

def normalize_column_names(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    # make headers consistent across files (lowercase + strip spaces)
    df = input_dataframe.copy()  # keep original untouched
    df.columns = [str(c).strip().lower() for c in df.columns]  # standardize header style

    # reconcile r2 aliases (e.g. "R^2", "r_2") to plain "r2"
    for c in list(df.columns):                         # iterate over column names
        flattened = c.replace('^', '').replace('_', '')  # drop special chars for easy compare
        if flattened == 'r2' and c != 'r2':            # matches r2 but not exactly named r2
            df.rename(columns={c: 'r2'}, inplace=True) # rename to r2

    # enforce final expected names (no-ops if already matching)
    df.rename(
        columns={'target': 'target', 'model': 'model', 'mae': 'mae', 'rmse': 'rmse', 'r2': 'r2'},
        inplace=True,
        errors='ignore',  # ignore missing keys
    )
    return df  # return normalized frame

def canonicalize_target_label(raw_label: str) -> str:
    # map messy target labels to a single clean form (e.g., "avg latency (ms)" -> "avg_latency")
    s = str(raw_label).strip().lower()       # simple normalize
    s = s.replace('-', ' ').replace('_', ' ')  # unify separators
    s = ' '.join(s.split())                  # collapse extra spaces

    # known variants mapped to canonical tokens used across files/plots
    mapping = {
        'avg latency': 'avg_latency',
        'avg latency (ms)': 'avg_latency',
        'average latency': 'avg_latency',
        'avg latency ms': 'avg_latency',
        'avg_latency': 'avg_latency',
        'upload bitrate': 'upload_bitrate',
        'upload (mbps)': 'upload_bitrate',
        'upload mbps': 'upload_bitrate',
        'upload_bitrate': 'upload_bitrate',
        'download bitrate': 'download_bitrate',
        'download (mbps)': 'download_bitrate',
        'download mbps': 'download_bitrate',
        'download_bitrate': 'download_bitrate',
    }
    return mapping.get(s, s.replace(' ', '_'))  # default: underscore words

def standardize_model_label(raw_label: str) -> str:
    # normalize model names so legends and summaries look clean
    s = str(raw_label).strip()  # keep case for mapping
    replacements = {
        'GRU-lean-v2': 'GRU',
        'gru-lean-v2': 'GRU',
        'LSTM-lean-v2': 'LSTM',
        'lstm-lean-v2': 'LSTM',
        'xgboost': 'XGBoost',
        'arima': 'ARIMA',
    }
    return replacements.get(s, s)  # fallback to original if not in map

def load_results_file(file_path: str, forced_model_label: str | None = None) -> pd.DataFrame:
    # read one results csv, normalize columns, coerce numbers, compute mse, and clean labels
    df = pd.read_csv(file_path)                   # load csv
    df = normalize_column_names(df)               # fix header styles + r2 name

    # ensure required columns exist (target, mae, rmse are mandatory)
    for required_name in ['target', 'mae', 'rmse']:
        if required_name not in df.columns:
            raise ValueError(f"{os.path.basename(file_path)} is missing required column '{required_name}'")

    # set/derive the model label column
    if forced_model_label:
        df['model'] = forced_model_label          # override when we know the file is a single model
    else:
        if 'model' not in df.columns:
            # if no model column in file, infer it from filename (e.g., "gruResults.csv" -> "gruResults")
            df['model'] = os.path.splitext(os.path.basename(file_path))[0]

    # coerce numeric metric columns; anything non-numeric becomes nan
    for metric_name in ['mae', 'rmse', 'r2']:
        if metric_name in df.columns:
            df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')

    # add mse (since some files only have rmse)
    df['mse'] = df['rmse'] ** 2  # mse = rmse squared

    # clean target and model labels for consistent grouping/plotting
    df['target'] = df['target'].apply(canonicalize_target_label)
    df['model'] = df['model'].apply(standardize_model_label)

    # keep a tidy set of expected columns; add if missing so downstream code is stable
    keep_columns = ['target', 'model', 'mae', 'rmse', 'mse', 'r2']
    for name in keep_columns:
        if name not in df.columns:
            df[name] = np.nan  # fill with nan if absent

    return df[keep_columns]  # return only the columns we need

def build_categorical_order(dataframe: pd.DataFrame, preferred_order: list[str]) -> pd.DataFrame:
    # build an ordered categorical dtype for 'model' to control legend/bar order
    existing_labels = list(dataframe['model'].dropna().unique())  # what models are present
    categorical_order = list(dict.fromkeys(preferred_order + sorted(existing_labels)))  # combine + dedupe
    cat_dtype = CategoricalDtype(categories=categorical_order, ordered=True)  # ordered categories
    dataframe['model'] = dataframe['model'].astype(cat_dtype)  # apply dtype
    return dataframe  # return with ordered model column

def render_grouped_bar_comparisons(axis, subset: pd.DataFrame, metric_name: str, y_label: str | None = None):
    # draw grouped bars: each target is a group; within each group, one bar per model
    target_labels = list(subset['target'].unique())  # group axis
    model_labels = list(subset['model'].unique())    # bars per group

    x_positions = np.arange(len(target_labels))      # base x positions for groups
    bar_width = 0.8 / max(len(model_labels), 1)      # split group width among models

    # build bars for each model across all targets
    for idx, model_label in enumerate(model_labels):
        y_values = []
        for t in target_labels:
            # find the value for (target t, model model_label); nan if missing
            selection = subset[(subset['target'] == t) & (subset['model'] == model_label)][metric_name]
            y_values.append(selection.iloc[0] if not selection.empty else np.nan)
        # offset each model’s bars within the group
        axis.bar(
            x_positions + idx * bar_width - (len(model_labels) - 1) * bar_width / 2,
            y_values,
            width=bar_width,
            label=model_label,
        )

    # prettify axes
    axis.set_xticks(x_positions)                                # put ticks at group centers
    axis.set_xticklabels(target_labels, rotation=0)             # label targets
    axis.set_ylabel(y_label or metric_name.upper())             # y label default = metric name
    axis.set_title(y_label or metric_name.upper())              # title mirrors y label
    axis.legend()                                               # show model legend
    axis.grid(axis='y', linestyle='--', alpha=0.4)              # light horizontal grid

def assemble_summary_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    # compute best model per target for mae/rmse/mse (lower=better) and r2 (higher=better)

    def pick_best(group_df: pd.DataFrame, metric: str, larger_is_better: bool):
        # choose row with best metric within one target group
        filtered = group_df.dropna(subset=[metric])                   # ignore rows with missing metric
        if filtered.empty:
            return None                                               # nothing to pick
        best_index = filtered[metric].idxmax() if larger_is_better else filtered[metric].idxmin()
        return filtered.loc[best_index, ['target', 'model', metric]]  # return small summary row

    summary_rows = []  # collect per-target winners across metrics
    for target_name, target_group in dataframe.groupby('target'):                 # loop each target
        for metric_name, larger_flag in [('mae', False), ('rmse', False), ('mse', False), ('r2', True)]:
            best_row = pick_best(target_group, metric_name, larger_flag)          # pick winner
            if best_row is not None:
                row = best_row.copy()
                row['criterion'] = f"best_{metric_name}"                          # label the rule used
                summary_rows.append(row)

    if not summary_rows:
        # return empty frame with expected columns if nothing found
        return pd.DataFrame(columns=['target', 'model', 'metric', 'criterion'])
    # sort by target then criterion so the csv is easy to read
    return pd.DataFrame(summary_rows).sort_values(['target', 'criterion'])

def generate_all_figures(dataframe: pd.DataFrame, output_directory: str):
    # create a 2x2 panel per target showing mae, rmse, mse, and r2 comparisons
    os.makedirs(output_directory, exist_ok=True)                    # ensure out dir exists
    metric_specs = [('mae', 'mae'), ('rmse', 'rmse'), ('mse', 'mse'), ('r2', 'r²')]  # keys + pretty labels

    for target_name, target_group in dataframe.groupby('target'):   # one figure per target
        figure, axes_array = plt.subplots(2, 2, figsize=(12, 8))    # 2x2 panel
        axes_list = axes_array.ravel()                               # flatten for easy looping

        for ax, (metric_key, pretty_label) in zip(axes_list, metric_specs):
            render_grouped_bar_comparisons(ax, target_group, metric_key, pretty_label)  # draw subplot
            if metric_key == 'r2':
                # set r2 y-limits a bit beyond observed range to avoid cramped bars
                r2_min = target_group['r2'].min() if target_group['r2'].notnull().any() else 0.0
                r2_max = target_group['r2'].max() if target_group['r2'].notnull().any() else 1.0
                lower_bound = math.floor(min(-0.1, float(r2_min)) * 10) / 10  # pad lower
                upper_bound = math.ceil(max(1.0, float(r2_max)) * 10) / 10    # pad upper
                ax.set_ylim(lower_bound, upper_bound)

        figure.suptitle(f"model comparison — {target_name}", fontsize=14, fontweight='bold')  # title per target
        figure.tight_layout(rect=[0, 0.02, 1, 0.96])                                         # leave space for title
        output_path = os.path.join(output_directory, f"comparison_{target_name}.png")        # path for png
        figure.savefig(output_path, dpi=160)                                                 # write image
        plt.close(figure)                                                                     # free memory

def parse_command_line_arguments():
    # define cli flags and defaults
    parser = argparse.ArgumentParser(description='compare gru, lstm, and xgboost csv results (arima excluded)')
    parser.add_argument('--gru', default='gruResults.csv', help='path to gru results csv')               # gru csv path
    parser.add_argument('--lstm', default='lstmResults.csv', help='path to lstm results csv')            # lstm csv path
    parser.add_argument('--xgb_arima', default='xgbArimaResults.csv', help='path to xgboost+arima results csv')  # xgb+arima csv
    parser.add_argument('--outdir', default='plots', help='output directory for figures and summary csv')        # out dir
    parser.add_argument(
        '--include',
        nargs='*',
        default=['GRU', 'LSTM', 'XGBoost'],     # by default we ignore arima
        help='models to include (default: gru lstm xgboost)',
    )
    return parser.parse_args()  # parse and return args

if __name__ == '__main__':
    args = parse_command_line_arguments()  # read cli flags

    # load up to three sources; force model labels for per-model files; keep labels from xgb/arima file
    dataframe_list = []  # will hold frames to concat
    if os.path.exists(args.gru):   # only read if file is present
        dataframe_list.append(load_results_file(args.gru, forced_model_label='GRU'))   # tag as gru
    if os.path.exists(args.lstm):
        dataframe_list.append(load_results_file(args.lstm, forced_model_label='LSTM')) # tag as lstm
    if os.path.exists(args.xgb_arima):
        dataframe_list.append(load_results_file(args.xgb_arima))                        # keep embedded labels

    if not dataframe_list:
        # nothing to compare → fail fast with a helpful message
        raise FileNotFoundError('no csv files were found next to this script')

    combined_dataframe = pd.concat(dataframe_list, ignore_index=True)  # stack all rows

    # keep only requested models (arima excluded by default via --include)
    combined_dataframe = combined_dataframe[combined_dataframe['model'].isin(args.include)].reset_index(drop=True)

    # lock a stable legend/order so plots look consistent across runs
    preferred_model_order = ['GRU', 'LSTM', 'XGBoost']  # desired order
    combined_dataframe = build_categorical_order(combined_dataframe, preferred_model_order)

    # write summary csv and render per-target figures
    summary_dataframe = assemble_summary_table(combined_dataframe)    # compute winners
    os.makedirs(args.outdir, exist_ok=True)                           # ensure output folder
    summary_path = os.path.join(args.outdir, 'metrics_summary.csv')   # summary path
    summary_dataframe.to_csv(summary_path, index=False)               # save summary

    generate_all_figures(combined_dataframe, args.outdir)             # generate all comparison plots
