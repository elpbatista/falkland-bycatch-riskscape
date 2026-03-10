import pandas as pd

df = pd.read_parquet("data/layer2/year=2014.parquet")

print(df.columns)
print(df[["sst_grad", "chl_grad", "ssh_grad"]].describe())