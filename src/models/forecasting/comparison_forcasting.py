# compares gru, lstm, srnn, xgboost and arima metrics from five csv files

#
import os                   
import math                  
import argparse              #flooor ceil psacing 
import numpy as np           
import pandas as pd           
import matplotlib.pyplot as plt  
from pandas.api.types import CategoricalDtype  
#ensuringn headings are all lowercsasing 
def normalize_column_names(input_dataframe: pd.DataFrame) -> pd.DataFrame:#takes the padas DAtaFrame and returns a cleaned up Dataframe. 
    df = input_dataframe.copy()  #make a copy such that the original dataframe isnt changed 
    df.columns = [str(c).strip().lower() for c in df.columns]  #CONVERT EVERY COLLUMN TO LOWERCASE STRING AND TRIM SPACES GIVES - helps easy to match column names

    for c in list(df.columns):                         #iterate over each column name (list() so it can safely rename while iterating
        flattened = c.replace('^', '').replace('_', '')   #remove (^) and _ to help detect varaints like r^, r_2 etc 
        if flattened == 'r2' and c != 'r2':             #renames standard name r2 
            df.rename(columns={c: 'r2'}, inplace=True) #rename to r2as ^needs to be repalced 

    #renaming all teh columsn - amke sure the colun names are exactly as shown as some of the them are slightly off 
    df.rename(
        columns={'target': 'target', 'model': 'model', 'mae': 'mae', 'rmse': 'rmse', 'r2': 'r2'},
        inplace=True,#apply the rename directly to teh df - dataframe 
        errors='ignore',   #if any of those columns dont exist just 'skip'  
    )
    return df   #reutrn the dataframe 
#Avg Latency (ms) -> Avg_latency | uploead mbs -> upload_bitrate | pint time -> ping_time

def canonicalize_target_label(raw_label: str) -> str:
    s = str(raw_label).strip().lower()       #makes it lowercase strign with no extra spaces at the end | as there are AVG Latency vs avg latency
    s = s.replace('-', ' ').replace('_', ' ')  #turn daskes/underscore sinto space | differences = avg - latency -> avg latency) lowercasign + trimming makes them match 
    s = ' '.join(s.split())                  # square the multiple spaces down to one. | avg latency -> shouldnt be different from "avg latency" 

    #dictionary for knwon messy versions | eg (avegerage latenc, avg latency (ms), avg_latency) -> mappign forces them to the same stadnard key liek avg_latency
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
    return mapping.get(s, s.replace(' ', '_'))  #if we recongise the cleaned text s, return the mapped standard (e.g avg_latency) otherwise just replace SPACES WITH UNDERSCORES
# 
def standardize_model_label(raw_label: str) -> str:#turn messsy mdoel names into clean, consstant ones for charts/tables | REASON: so legends dont show up as DUPLICATES e.g gru-lean-v2 vs GRU 
    # normalizemodel names so legends andsummaries look clean
    s = str(raw_label).strip()  # keep case for mapping
    replacements = {
        'GRU-lean-v2': 'GRU',
        'gru-lean-v2': 'GRU',
        'LSTM-lean-v2': 'LSTM',
        'lstm-lean-v2': 'LSTM',
        'xgboost': 'XGBoost',
        'arima': 'ARIMA',
        'srnn': 'SRNN',
    }
    return replacements.get(s, s)  #if s iis in the map reutrn the clean name othewsie return 's' uncahnged to ensure variants are consisntat; unknown labels still pass throguh safely (no crashs) 

def load_results_file(file_path: str, forced_model_label: str | None = None) -> pd.DataFrame:#rreasd one reuslts CSV and cleans it to standard format -> so all downstream codecan rely on the same columns adn labels
    df = pd.read_csv(file_path)                   # we need the raw results in memory to lean n ause tehm 
    df = normalize_column_names(df)               # 
#lowercase stadardise column names (fix r^2 -> r2 e) help prevent bugs related to namign differences 
    for required_name in ['target', 'mae', 'rmse']:#go throguh list [target, mae, rmse] heck if eahc o those column names are in the csv | if one of them missing thn show errro 
        if required_name not in df.columns:
            raise ValueError(f"{os.path.basename(file_path)} is missing required column '{required_name}'")

    #if functioan is told what model file it belongs to like ('GRU') it will se teh model column to that name for every row  -> esnrue the data is correctly labelled, even if the CSV doesnt already hav ea model columns etderive the model label column
    if forced_model_label:#if function is told what model this file beongs to like (GRU) it sets teh mdoel column to that name for every row 
        df['model'] = forced_model_label          #
    else:#if no labe is iven the file doesnt have a model column, it uses teh filename like (gruResults) as the model name -> this way the resutls aways ahas a model name to idneitfy it 
        if 'model' not in df.columns:
            df['model'] = os.path.splitext(os.path.basename(file_path))[0]

    # coerce numeric metric columns anything nonnumeric becomes nan | makes sure the metric values are numbers if it doesnt thant it turns to NAN
    for metric_name in ['mae', 'rmse', 'r2']:
        if metric_name in df.columns:
            df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')

    # add mse since some files only have rmse| cacalte MSE by suareign teh RMSE 
    df['mse'] = df['rmse'] ** 2  #  

    #clean target and model labels for consistent grouping/plotting|cleans up the targe names like (turning AVG Latency (ms) into avg_latency') | makes sure all target use teh same naming style for acruate grouping alter 
    df['target'] = df['target'].apply(canonicalize_target_label)#cleans up the target names (like trunign avg latency(ms) to avg_latency) makes sure all targets are using teh same anming style for accruage groupng later
    df['model'] = df['model'].apply(standardize_model_label)#cleans up model anmes (like "gru lean -v2 -> GRU" ) keeps theanme tindy adn consistna tin carts and tables 




#lists the columsn we awant ot keep in teh final version -> make sure only useufl data is reutured in the same order every time 
    #keep a tidy set of expected columns add if missing so downstream code is stable
    keep_columns = ['target', 'model', 'mae', 'rmse', 'mse', 'r2']
    for name in keep_columns:#if any expect column insmissing it will be filled wiht empty NAN values 
        if name not in df.columns:
            df[name] = np.nan  # fill th nan if absent

    return df[keep_columns]  #return only the columns we need| returns teh cleaned and organized DatFrame with just those columsn _> the rest of the program can now safel usue it nowing the strucutre is aways always teh same. 
#help fix the display order fo the model names - solegends and abr groups appear in aconsisntnt oder ex GRU, LSTM, XGBOOOST etc 
def build_categorical_order(dataframe: pd.DataFrame, preferred_order: list[str]) -> pd.DataFrame:
    # build an oordered categorical dtype for model to control legendbar order
    #collect distinct model name that are actually present in the data (ignorign blanks) - only want to order mdoels that exist in the dataaset 
    existing_labels = list(dataframe['model'].dropna().unique())  
    categorical_order = list(dict.fromkeys(preferred_order + sorted(existing_labels)))  #make a final order throguh puttign preffered order first, then remove duplicates 
    cat_dtype = CategoricalDtype(categories=categorical_order, ordered=True)  #pandas Categorical type with the exact order 
    dataframe['model'] = dataframe['model'].astype(cat_dtype)  #convert the 'model' column to the ordered categorical type - applies the orderign so plots/legends will follow it automatically 
    return dataframe  #reeturn with ordered mmodel column
#help plottign that will draw grouped bar charts for one metric (e.g MAE)- help visually cmapre models side by side for each target 
def render_grouped_bar_comparisons(axis, subset: pd.DataFrame, metric_name: str, y_label: str | None = None):
    target_labels = list(subset['target'].unique())  #collect target [avg_latency, upload_bitrate] that goes on the x axis (becomes one "group" of bars)
    model_labels = list(subset['model'].unique())    #collect the distinct model names (e.g GRU, lstm xgbost)
    #DRAWS ONE BAR PERMODEL 

    x_positions = np.arange(len(target_labels))      #help create x position (0,1,2) for each target group 
    bar_width = 0.8 / max(len(model_labels), 1)    #needs numbering to help palce the groups along the x axsi 

    #loop over each model and its index drawing one set of bars per modle 
    for idx, model_label in enumerate(model_labels):
        y_values = []#start the list to help hold metric values for the model across all targets - matplotlib needs a Y value per target to help draw the bars 
        for t in target_labels:
            selection = subset[(subset['target'] == t) & (subset['model'] == model_label)][metric_name]
            y_values.append(selection.iloc[0] if not selection.empty else np.nan)#loop over each tart group on the x axis such that metrics for the model is retireved at each target 
        axis.bar(
            x_positions + idx * bar_width - (len(model_labels) - 1) * bar_width / 2,
            y_values,
            width=bar_width,
            label=model_label,
        )#draw bars for the model across all the targets this will help redner the grouped bars 

    #tick markers at eahc target group center - helps align the labels with bar groups 
    axis.set_xticks(x_positions)                             
    axis.set_xticklabels(target_labels, rotation=0)             #use the target names under each group (no rotation) helps make teh x axis human readable 
    axis.set_ylabel(y_label or metric_name.upper())             #lael the y axis name if not then the metric name helps clarify what the bar ehight will represent 
    axis.set_title(y_label or metric_name.upper())              #title the subplots 
    axis.legend()                                               #show which color will corespond to which model help viewers batch the bars to the models 
    axis.grid(axis='y', linestyle='--', alpha=0.4)              # add a horzontal grid for the chart 

#builds a small table that will show which model si best for each metric qucikly see which model wins  
def assemble_summary_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    # compute best model per target for mae/rmse/mse =higherbetter)

    def pick_best(group_df: pd.DataFrame, metric: str, larger_is_better: bool):
        # choose row with best metric within one target group
        filtered = group_df.dropna(subset=[metric])                   #ignore rows with missing metric
        if filtered.empty:
            return None                                               #nothing to pick give up for metric - helpa void error 
        best_index = filtered[metric].idxmax() if larger_is_better else filtered[metric].idxmin()
        return filtered.loc[best_index, ['target', 'model', metric]]  # pck the row with the alrgest vlaue for r^2 or smallest value for MAE/RMSE/MSE return small summary row

    summary_rows = []  #list for collectign all the winner rows
    for target_name, target_group in dataframe.groupby('target'):                 #loop each target work on one target at a time, winners are chosen per target 
        for metric_name, larger_flag in [('mae', False), ('rmse', False), ('mse', False), ('r2', True)]:
            best_row = pick_best(target_group, metric_name, larger_flag)          #loop ove each metric and whather bigger is better encodes the 'direction' once so the logic is simple 
            if best_row is not None:
                row = best_row.copy()
                row['criterion'] = f"best_{metric_name}"                          #get teh winnig row onnly porcssed once winner si winning row is ofund if not then ski p 
                summary_rows.append(row)

    if not summary_rows:
        # return empty frame with expected columns if nothing foundd, helps avoid returnign NONE and keeps the return type conssintat 
        return pd.DataFrame(columns=['target', 'model', 'metric', 'criterion'])
    # sort tarkget by csv k
    return pd.DataFrame(summary_rows).sort_values(['target', 'criterion'])

def generate_all_figures(dataframe: pd.DataFrame, output_directory: str):#makes charst that compare models for each target (mae, rmse, mse, r^2 )
    # create a 2x2 panel per target showing mae, rmse, mse, and r2 cooomparisons
    os.makedirs(output_directory, exist_ok=True)                    # ensure out dir exists prevnet errors when saving imaegs 
    metric_specs = [('mae', 'mae'), ('rmse', 'rmse'), ('mse', 'mse'), ('r2', 'r²')]  #list of which metrics to plot / how to label them 4 subplots 

    for target_name, target_group in dataframe.groupby('target'):   # handle one taget at a time e.g (avg_latency)
        figure, axes_array = plt.subplots(2, 2, figsize=(12, 8))    #make a 2x2 grid of subplots one panel per emtic (MAE, RMSE, MSE r^2 )
        axes_list = axes_array.ravel()                               #pair each suplot wiht a metric and its label - helps keep the code short and matches apnesl to metrics  loopingdnjsdakhfiajs

        for ax, (metric_key, pretty_label) in zip(axes_list, metric_specs):#pair each subplot with a a metric and its label - keeps the code shrot and matches panels to metrics 
            render_grouped_bar_comparisons(ax, target_group, metric_key, pretty_label)  #Draw the grouped bars mdoels vs metric fro teh target shwos sid eby side model perofrmance 
            if metric_key == 'r2':
                #special hadlign fro the r^2 plot  set r2 ylimits a bit beyrond obrserved ratnge to avoid cramped bars
                r2_min = target_group['r2'].min() if target_group['r2'].notnull().any() else 0.0
                r2_max = target_group['r2'].max() if target_group['r2'].notnull().any() else 1.0
                lower_bound = math.floor(min(-0.1, float(r2_min)) * 10) / 10  # pad lower
                upper_bound = math.ceil(max(1.0, float(r2_max)) * 10) / 10    # pad upper
                ax.set_ylim(lower_bound, upper_bound)

        figure.suptitle(f"model comparison — {target_name}", fontsize=14, fontweight='bold')  # title per target
        figure.tight_layout(rect=[0, 0.02, 1, 0.96])                                         # leave space for title
        output_path = os.path.join(output_directory, f"comparison_{target_name}.png")        # path for odpng
        figure.savefig(output_path, dpi=160)                                                 # write imageee
        plt.close(figure)                                                                     # free memoryy

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='compare gru, lstm, xgboost, and srnn csv results (arima excluded)')
    parser.add_argument('gru', default='gruResults.csv', help='path to gru results csv')
    parser.add_argument('lstm', default='lstmResults.csv', help='path to lstm results csv')
    parser.add_argument('xgb_arima', default='xgbArimaResults.csv', help='path to xgboost+arima results csv')
    parser.add_argument('srnn', default='srnnResults.csv', help='path to srnn results csv')
    parser.add_argument('outdir', default='plots', help='output directory for figures and summary csv')
    parser.add_argument(
        '--include',
        nargs='*',
        default=['GRU', 'LSTM', 'XGBoost', 'SRNN'],
        help='models to include (default: gru lstm xgboost srnn)',
    )
    return parser.parse_args()  

if __name__ == '__main__':
    args = parse_command_line_arguments()  # read cli flagssss

    # for fels  xgbarima file keeps embedded labels
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

    combined_dataframe = pd.concat(dataframe_list, ignore_index=True)  # 

    # exclduign armia for space reasons 
    combined_dataframe = combined_dataframe[combined_dataframe['model'].isin(args.include)].reset_index(drop=True)
#oooridngtn 
    
    preferred_model_order = ['GRU', 'LSTM', 'XGBoost', 'SRNN']
    combined_dataframe = build_categorical_order(combined_dataframe, preferred_model_order)
#summery 
    summary_dataframe = assemble_summary_table(combined_dataframe)     
    os.makedirs(args.outdir, exist_ok=True)                            
    summary_path = os.path.join(args.outdir, 'metrics_summary.csv')    
    summary_dataframe.to_csv(summary_path, index=False)                

    generate_all_figures(combined_dataframe, args.outdir)            
