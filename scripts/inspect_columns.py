import pandas as pd
import numpy as np
from riskscape.config import paths
from riskscape.model.dataset import SPECIES_TARGET

# path = paths["data"] / "processed" / "h3_neighbor_index.parquet"
path = paths["data"] / "features" / "environmental" / "year=2023/part.parquet"
path = paths["data"] / "features" / "static" / "static.parquet"
path = paths["data"] / "modeling" / "fishing_training" / "year=2022/part.parquet"
path = paths["data"] / "modeling" / "species_training" / "year=2022/part.parquet"
# path = paths["data"] / "modeling" / "predictions" / "year=2022/part.parquet"
path = paths["data"] / "processed" / "h3_neighbors.parquet"
# print(df[["sst_grad", "chl_log_grad", "ssh_grad"]].describe())

df = pd.read_parquet(path)

# print(df["risk_log_pred"].describe())

# print("\nBy species: mean")
# print(df.groupby(["h3", "species"])["risk_log_pred"].mean())

# print("\nBy species: max")
# print(df.groupby(["h3", "species"])["risk_log_pred"].max())

# print("\nBy month and species: mean")
# df["month"] = df["date"].dt.month
# print(df.groupby(["month", "species"])["risk_log_pred"].mean())

# print("\nnonzero only")
# print(df[df["risk_log_pred"] > 0]["risk_log_pred"].describe())

# print("\nTop percentiles:")
# nonzero = df[df["risk_log_pred"] > 0]
# print(nonzero["risk_log_pred"].quantile([0.90, 0.95, 0.99, 0.999]))

# print("\nConsistency with fishing activity:")
# print(df[df["risk_log_pred"] > 0]["fishing_activity"].describe())

# print("\nConsistency with fishing activity log:")
# print(df[df["risk_log_pred"] > 0]["fishing_activity_log"].describe())

# print("\nHotspots:")
# print(df[df["risk_log_pred"] > df["risk_log_pred"].quantile(0.95)])

# print("\nPersistent hotspots:")
# print(df.groupby(["h3", "species"])["risk_log_pred"].mean().nlargest(50))

# print("\nTemporal hotspots:")
# print(df.groupby(["month", "species"])["risk_log_pred"].mean().nlargest(50))

# print(df.columns.tolist())
# print(df.dtypes)
# print(df.head())

# df.plot.scatter("chl_log", SPECIES_TARGET)

# print(df["presence_count"].head(50))

# for f in ["sst", "chl_log", "ssh"]:

#     print(f, np.corrcoef(df[f], df[SPECIES_TARGET])[0, 1])

# fg = pd.read_parquet("data/modeling/feature_grid/year=2022/part.parquet")

# env = pd.read_parquet("data/features/environmental/year=2022/part.parquet")

# print("env rows:", len(env))

# print("feature_grid rows:", len(fg))

# print("dropped:", len(env[["h3", "date"]].drop_duplicates()) - len(fg))

df = pd.read_parquet("data/modeling/species_training/year=2022/part.parquet")

print(df["residence_index"].quantile([0.5, 0.9, 0.95, 0.99]))

print(df.groupby("species")["residence_index"].describe())

print((df["residence_index"] > 0).mean())