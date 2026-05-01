import pandas as pd

from riskscape.config import paths

# path = paths["data"] / "processed" / "h3_neighbor_index.parquet"
df = pd.read_parquet(paths["data"] / "features/environmental/year=2023/part.parquet")

print(df[["sst_grad", "chl_log_grad", "ssh_grad"]].describe())

# df = pd.read_parquet(path)

print(df.columns.tolist())
# print(df.head())