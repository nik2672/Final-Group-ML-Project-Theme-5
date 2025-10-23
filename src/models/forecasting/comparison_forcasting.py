# compare_models_from_csvs.py
# this script compares gru, lstm, and xgboost metrics from three csv files (arima excluded)
# it unifies headers and target labels, computes mse, produces per-target 2x2 plots,
# and writes a small summary table of best models per metric.

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype


def normalize_column_names(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    # make headers consistent across files
    df = input_dataframe.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # reconcile r2 aliases
    for c in list(df.columns):
        flattened = c.replace('^', '').replace('_', '')
        if flattened == 'r2' and c != 'r2':
            df.rename(columns={c: 'r2'}, inplace=True)

    # final expected names
    df.rename(
        columns={'target': 'target', 'model': 'model', 'mae': 'mae', 'rmse': 'rmse', 'r2': 'r2'},
        inplace=True,
        errors='ignore',
    )
    return df


def canonicalize_target_label(raw_label: str) -> str:
    # unify target labels from various files into a single canonical form
    s = str(raw_label).strip().lower()
    s = s.replace('-', ' ').replace('_', ' ')
    s = ' '.join(s.split())

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
    return mapping.get(s, s.replace(' ', '_'))


def standardize_model_label(raw_label: str) -> str:
    # standardize common model strings
    s = str(raw_label).strip()
    replacements = {
        'GRU-lean-v2': 'GRU',
        'gru-lean-v2': 'GRU',
        'LSTM-lean-v2': 'LSTM',
        'lstm-lean-v2': 'LSTM',
        'xgboost': 'XGBoost',
        'arima': 'ARIMA',
        'srnn': 'SRNN',
    }
    return replacements.get(s, s)


def load_results_file(file_path: str, forced_model_label: str | None = None) -> pd.DataFrame:
    # read one csv, coerce numeric fields, compute mse, and clean labels
    df = pd.read_csv(file_path)
    df = normalize_column_names(df)

    for required_name in ['target', 'mae', 'rmse']:
        if required_name not in df.columns:
            raise ValueError(f"{os.path.basename(file_path)} is missing required column '{required_name}'")

    if forced_model_label:
        df['model'] = forced_model_label
    else:
        if 'model' not in df.columns:
            df['model'] = os.path.splitext(os.path.basename(file_path))[0]

    for metric_name in ['mae', 'rmse', 'r2']:
        if metric_name in df.columns:
            df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')

    df['mse'] = df['rmse'] ** 2
    df['target'] = df['target'].apply(canonicalize_target_label)
    df['model'] = df['model'].apply(standardize_model_label)

    keep_columns = ['target', 'model', 'mae', 'rmse', 'mse', 'r2']
    for name in keep_columns:
        if name not in df.columns:
            df[name] = np.nan

    return df[keep_columns]


def build_categorical_order(dataframe: pd.DataFrame, preferred_order: list[str]) -> pd.DataFrame:
    # construct a deduplicated categorical order for plotting legends and grouping
    existing_labels = list(dataframe['model'].dropna().unique())
    categorical_order = list(dict.fromkeys(preferred_order + sorted(existing_labels)))
    cat_dtype = CategoricalDtype(categories=categorical_order, ordered=True)
    dataframe['model'] = dataframe['model'].astype(cat_dtype)
    return dataframe


def render_grouped_bar_comparisons(axis, subset: pd.DataFrame, metric_name: str, y_label: str | None = None):
    # draw grouped bars: one group per target, one bar per model within the group
    target_labels = list(subset['target'].unique())
    model_labels = list(subset['model'].unique())

    x_positions = np.arange(len(target_labels))
    bar_width = 0.8 / max(len(model_labels), 1)

    for idx, model_label in enumerate(model_labels):
        y_values = []
        for t in target_labels:
            selection = subset[(subset['target'] == t) & (subset['model'] == model_label)][metric_name]
            y_values.append(selection.iloc[0] if not selection.empty else np.nan)
        axis.bar(
            x_positions + idx * bar_width - (len(model_labels) - 1) * bar_width / 2,
            y_values,
            width=bar_width,
            label=model_label,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(target_labels, rotation=0)
    axis.set_ylabel(y_label or metric_name.upper())
    axis.set_title(y_label or metric_name.upper())
    axis.legend()
    axis.grid(axis='y', linestyle='--', alpha=0.4)


def assemble_summary_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    # compute winners per target for mae, rmse, mse (lower is better) and r2 (higher is better)
    def pick_best(group_df: pd.DataFrame, metric: str, larger_is_better: bool):
        filtered = group_df.dropna(subset=[metric])
        if filtered.empty:
            return None
        best_index = filtered[metric].idxmax() if larger_is_better else filtered[metric].idxmin()
        return filtered.loc[best_index, ['target', 'model', metric]]

    summary_rows = []
    for target_name, target_group in dataframe.groupby('target'):
        for metric_name, larger_flag in [('mae', False), ('rmse', False), ('mse', False), ('r2', True)]:
            best_row = pick_best(target_group, metric_name, larger_flag)
            if best_row is not None:
                row = best_row.copy()
                row['criterion'] = f"best_{metric_name}"
                summary_rows.append(row)

    if not summary_rows:
        return pd.DataFrame(columns=['target', 'model', 'metric', 'criterion'])
    return pd.DataFrame(summary_rows).sort_values(['target', 'criterion'])


def generate_all_figures(dataframe: pd.DataFrame, output_directory: str):
    # create a 2x2 panel for each target with mae, rmse, mse, and r2
    os.makedirs(output_directory, exist_ok=True)
    metric_specs = [('mae', 'mae'), ('rmse', 'rmse'), ('mse', 'mse'), ('r2', 'r²')]

    for target_name, target_group in dataframe.groupby('target'):
        figure, axes_array = plt.subplots(2, 2, figsize=(12, 8))
        axes_list = axes_array.ravel()

        for ax, (metric_key, pretty_label) in zip(axes_list, metric_specs):
            render_grouped_bar_comparisons(ax, target_group, metric_key, pretty_label)
            if metric_key == 'r2':
                r2_min = target_group['r2'].min() if target_group['r2'].notnull().any() else 0.0
                r2_max = target_group['r2'].max() if target_group['r2'].notnull().any() else 1.0
                lower_bound = math.floor(min(-0.1, float(r2_min)) * 10) / 10
                upper_bound = math.ceil(max(1.0, float(r2_max)) * 10) / 10
                ax.set_ylim(lower_bound, upper_bound)

        figure.suptitle(f"model comparison — {target_name}", fontsize=14, fontweight='bold')
        figure.tight_layout(rect=[0, 0.02, 1, 0.96])
        output_path = os.path.join(output_directory, f"comparison_{target_name}.png")
        figure.savefig(output_path, dpi=160)
        plt.close(figure)



def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='compare gru, lstm, xgboost, and srnn csv results (arima excluded)')
    parser.add_argument('--gru', default='gruResults.csv', help='path to gru results csv')
    parser.add_argument('--lstm', default='lstmResults.csv', help='path to lstm results csv')
    parser.add_argument('--xgb_arima', default='xgbArimaResults.csv', help='path to xgboost+arima results csv')
    parser.add_argument('--srnn', default='srnnResults.csv', help='path to srnn results csv')
    parser.add_argument('--outdir', default='plots', help='output directory for figures and summary csv')
    parser.add_argument(
        '--include',
        nargs='*',
        default=['GRU', 'LSTM', 'XGBoost', 'SRNN'],
        help='models to include (default: gru lstm xgboost srnn)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_arguments()

    # load four sources; force labels for the per-model files; xgb/arima file keeps embedded labels
    dataframe_list = []
    if os.path.exists(args.gru):
        dataframe_list.append(load_results_file(args.gru, forced_model_label='GRU'))
    if os.path.exists(args.lstm):
        dataframe_list.append(load_results_file(args.lstm, forced_model_label='LSTM'))
    if os.path.exists(args.xgb_arima):
        dataframe_list.append(load_results_file(args.xgb_arima))
    if os.path.exists(args.srnn):
        dataframe_list.append(load_results_file(args.srnn, forced_model_label='SRNN'))

    if not dataframe_list:
        raise FileNotFoundError('no csv files were found next to this script')

    combined_dataframe = pd.concat(dataframe_list, ignore_index=True)

    # filter to requested models (arima excluded by default)
    combined_dataframe = combined_dataframe[combined_dataframe['model'].isin(args.include)].reset_index(drop=True)

    # enforce stable legend order
    preferred_model_order = ['GRU', 'LSTM', 'XGBoost', 'SRNN']
    combined_dataframe = build_categorical_order(combined_dataframe, preferred_model_order)

    # write summary and render figures
    summary_dataframe = assemble_summary_table(combined_dataframe)
    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, 'metrics_summary.csv')
    summary_dataframe.to_csv(summary_path, index=False)

    generate_all_figures(combined_dataframe, args.outdir)
