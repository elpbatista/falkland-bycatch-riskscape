import pandas as pd

from riskscape.config import paths

# path = paths["data"] / "processed" / "h3_neighbor_index.parquet"
path = paths["data"] / "features" / "environmental" / "year=2023/part.parquet"
path = paths["data"] / "features" / "static" / "static.parquet"
path = paths["data"] / "modeling" / "fishing_training" / "year=2022/part.parquet"
path = paths["data"] / "modeling" / "species_training" / "year=2022/part.parquet"

# df = pd.read_parquet(paths["data"] / "features/environmental/year=2023/part.parquet")

# print(df[["sst_grad", "chl_log_grad", "ssh_grad"]].describe())

df = pd.read_parquet(path)

print(df.columns.tolist())
print(df.dtypes)
print(df.head())