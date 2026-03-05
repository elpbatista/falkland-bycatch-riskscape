import pandas as pd

df = pd.read_parquet("data/layer1/year=2014.parquet")

print(len(df))
print(df.groupby("date").size().head())