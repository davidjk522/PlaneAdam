import os

import pandas as pd
from scipy.stats import ttest_ind

current_dir = os.getcwd()
# Load your data (assuming the CSV files are in the current directory)
csv_files = ['/data/data1/zft/code/FMIR/FMIR/logs/EMCV_logs/R=1 encoderOnlyACDCComplexcostvolumn.csv', '/data/data1/zft/code/FMIR/FMIR/logs/EMCV_logs/R=0 encoderOnlyACDCComplexcostvolumn.csv']

dataframes = [pd.read_csv(file_name) for file_name in csv_files]
# Extract labels
lkunet_df = dataframes[1]
labels = lkunet_df.columns.tolist()[1:10]
labels_sorted = sorted(labels, key=lambda x: lkunet_df[x].mean(), reverse=True)

# Perform t-tests
results = {}
textscf_df = dataframes[0]
for label in labels_sorted:
    results[label] = {}
    for i, df in enumerate(dataframes[1:], 1):
        stat, p = ttest_ind(textscf_df[label], df[label], equal_var=False)  # Assume unequal variance
        results[label][csv_files[i].replace('.csv', '')] = p

# Display the results
for label in results:
    print(f"Label: {label}")
    for method, p_value in results[label].items():
        print(f"  {method} vs R=1, p-value: {p_value}")
    print()
