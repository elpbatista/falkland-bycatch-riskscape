import pandas as pd

df = pd.read_parquet("data/processed/sst_gradient_2014.parquet")

print(df.head())
print(df["sst_grad"].describe())