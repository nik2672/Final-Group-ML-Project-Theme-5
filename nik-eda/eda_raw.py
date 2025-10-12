import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#path to dataset
data_path = r"C:\swinburne\FINAL YEAR!!!\AI FOR ENIGNEERING\new\Final-Group-ML-Project-Theme-5\data\combined_raw.csv"

df = pd.read_csv(data_path, low_memory=False)
print("raw dataset shape:", df.shape)
print("\ncolumn info:\n")
print(df.info())

#check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print("\nmissing values (top 10):\n", missing.head(10))

#stats for numerical columns
print("\nsummary statistics:\n", df.describe())

# chekc rows that are duplicated 
print("\nnumber of duplicate rows:", df.duplicated().sum())

# visualisation 
num_cols = df.select_dtypes(include="number").columns[:5]
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
