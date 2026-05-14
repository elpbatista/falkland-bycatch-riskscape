
from zipfile import Path

import pandas as pd
from riskscape.model.dataset import SPECIES_TARGET

import numpy as np
import pyarrow.parquet as pq

from riskscape.config import paths


# df = pd.read_parquet("data/modeling/predictions/year=2022/part.parquet")

# wide = df.pivot_table(
#     index=["h3", "date"],
#     columns="species",
#     values="species_use_log_pred",
# )

# print("Correlation:")
# print(wide.corr())

# print("\nDifference stats:")
# print((wide["BBAL"] - wide["SAFS"]).describe())

# df = pd.read_parquet(
#     "data/modeling/species_training/year=2022/part.parquet"
# )
path = paths["data"] / "modeling" / "fishing_training" / "year=2022" / "part.parquet"
# path = paths["data"] / "features" / "environmental" / "year=2022" / "part.parquet"

path = paths["data"] / "modeling" / "predictions" / "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30" / "joint" / "year=2022" / "part.parquet"

df = pd.read_parquet(path)


# print(df["residence_index"].describe())
# print(df["residence_index"].nunique())
# print(df["residence_index"].value_counts().head(30))
# print(sorted(df["residence_index"].unique())[:50])

# print((df.groupby("species")["residence_index"].describe()))


# print(df.groupby("species")[SPECIES_TARGET].std())

# print(df.groupby("species")[SPECIES_TARGET].mean())

# print(df.groupby("species")["presence_count"].mean())

# print(df.groupby("species")["presence_count"].std())

# print(df.groupby("species")[SPECIES_TARGET].describe())

# print(df["residence_index"].value_counts().head())
# print(df.groupby("species")["residence_index"].describe())
# print(df.groupby("species")["date"].nunique())

# print(np.percentile(df["risk_log_pred"], [90, 95, 99, 99.9]))

# print("\nTop percentiles:")
# nonzero = df[df["risk_log_pred"] > 0]
# print(nonzero["risk_log_pred"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

# print(np.percentile(df["risk_log_pred"], [50, 75, 90, 95, 99, 99.5, 99.9]))

# print(
#     df[df["species_use_log_pred"] > 0]
#     .groupby("species")["species_use_log_pred"]
#     .describe()
# )

print(df.columns.tolist())

# print(df.head())

# print(pq.read_schema(path).field("date").type)
# print(df["date"].dtype)
# print(type(df["date"].iloc[0]))
# print(df["date"].head())

print(df[["plausibility", "plausibility_gate", "hybrid_alpha"]].describe())

