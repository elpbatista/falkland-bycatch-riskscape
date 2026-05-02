import pandas as pd

from riskscape.config import paths

# path = paths["data"] / "processed" / "h3_neighbor_index.parquet"
path = paths["data"] / "features" / "environmental" / "year=2023/part.parquet"
path = paths["data"] / "features" / "static" / "static.parquet"
path = paths["data"] / "modeling" / "fishing_training" / "year=2022/part.parquet"
path = paths["data"] / "modeling" / "species_training" / "year=2022/part.parquet"
path = paths["data"] / "modeling" / "predictions" / "year=2023/part-00000.parquet"

# print(df[["sst_grad", "chl_log_grad", "ssh_grad"]].describe())

df = pd.read_parquet(path)

# print(df["risk_pred"].describe())

# print("\nBy species: mean")
# print(df.groupby(["h3", "species"])["risk_pred"].mean())

# print("\nBy species: max")
# print(df.groupby(["h3", "species"])["risk_pred"].max())

# print("\nBy month and species: mean")
df["month"] = df["date"].dt.month
# print(df.groupby(["month", "species"])["risk_pred"].mean())

# print("\nnonzero only")
# print(df[df["risk_pred"] > 0]["risk_pred"].describe())

# print("\nTop percentiles:")
# print(df["risk_pred"].quantile([0.90, 0.95, 0.99, 0.999]))

# print("\nConsistency with fishing activity:")
# print(df[df["risk_pred"] > 0]["fishing_activity_pred"].describe())

print("\nHotspots:")
print(df[df["risk_pred"] > df["risk_pred"].quantile(0.95)])

print("\nPersistent hotspots:")
print(df.groupby(["h3", "species"])["risk_pred"].mean().nlargest(50))

print("\nTemporal hotspots:")
print(df.groupby(["month", "species"])["risk_pred"].mean().nlargest(50))

# print(df.columns.tolist())
# print(df.dtypes)
# print(df.head())