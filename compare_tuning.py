import pandas as pd

print('='*80)
print('COMPARISON: tuning.py vs main2_optimized.py')
print('='*80)

# Load results
df_tuning = pd.read_csv('results/clustering/tuning/best_per_model_combined.csv')
df_birch_opt = pd.read_csv('results/clustering_optimized/birch_trials.csv')
df_optics_opt = pd.read_csv('results/clustering_optimized/optics_trials.csv')

print('\n### BIRCH ###')
print('\ntuning.py:')
birch1 = df_tuning[df_tuning['Model']=='Birch'].iloc[0]
print(f'  Params: {birch1["Param"]}')
print(f'  Silhouette: {birch1["Silhouette"]:.4f}')
print(f'  DBI: {birch1["DBI"]:.4f}')
print(f'  Calinski-Harabasz: {birch1["Calinski_Harabasz"]:.2f}')
print(f'  Combined Score: {birch1["CombinedScore"]:.4f}')

print('\nmain2_optimized.py:')
birch2 = df_birch_opt.iloc[0]
print(f'  Params: n_clusters={int(birch2["n_clusters"])}, threshold={birch2["threshold"]}, branching_factor={int(birch2["branching_factor"])}')
print(f'  Silhouette: {birch2["silhouette"]:.4f}')
print(f'  DBI: {birch2["davies_bouldin"]:.4f}')
print(f'  Calinski-Harabasz: {birch2["calinski_harabasz"]:.2f}')
print(f'  Combined Score: {birch2["combined_score"]:.4f}')

print('\n' + '-'*80)
print('\n### OPTICS ###')
print('\ntuning.py:')
optics1 = df_tuning[df_tuning['Model']=='OPTICS'].iloc[0]
print(f'  Params: {optics1["Param"]}')
print(f'  Silhouette: {optics1["Silhouette"]:.4f}')
print(f'  DBI: {optics1["DBI"]:.4f}')
print(f'  Calinski-Harabasz: {optics1["Calinski_Harabasz"]:.2f}')
print(f'  Combined Score: {optics1["CombinedScore"]:.4f}')

print('\nmain2_optimized.py:')
optics2 = df_optics_opt.iloc[0]
print(f'  Params: min_samples={int(optics2["min_samples"])}, max_eps={optics2["max_eps"]}, xi={optics2["xi"]}, cluster_method={optics2["cluster_method"]}')
print(f'  Silhouette: {optics2["silhouette"]:.4f}')
print(f'  Calinski-Harabasz: {optics2["calinski_harabasz"]:.2f}')
print(f'  Combined Score: {optics2["combined_score"]:.4f}')

print('\n' + '='*80)
print('\n### KEY DIFFERENCES ###')
print('\nBIRCH:')
print(f'  • tuning.py found: k=2, threshold=0.5 (Combined: {birch1["CombinedScore"]:.4f})')
print(f'  • optimized.py found: k=2, threshold=0.2 (Combined: {birch2["combined_score"]:.4f})')
print(f'  ➜ Optimized.py is {"BETTER" if birch2["combined_score"] > birch1["CombinedScore"] else "WORSE"} ({abs(birch2["combined_score"]-birch1["CombinedScore"]):.4f} difference)')

print('\nOPTICS:')
print(f'  • tuning.py found: min_samples=7, max_eps=2.0, xi=0.05, method=xi')
print(f'  • optimized.py found: min_samples=5, max_eps=2.0, xi=0.05, method=dbscan')
print(f'  ➜ Different cluster_method! (xi vs dbscan)')
print(f'  ➜ optimized.py tested cluster_method=dbscan (tuning.py only tested xi)')
print(f'  ➜ Optimized silhouette: {optics2["silhouette"]:.4f} vs tuning: {optics1["Silhouette"]:.4f}')
print(f'  ➜ Optimized.py is {"BETTER" if optics2["silhouette"] < optics1["Silhouette"] else "WORSE"} (lower silhouette but found better method)')

print('\n' + '='*80)
