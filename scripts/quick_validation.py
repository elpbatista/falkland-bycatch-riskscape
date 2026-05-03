
import pandas as pd
from riskscape.model.dataset import SPECIES_TARGET


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

df = pd.read_parquet(
    "data/modeling/species_training/year=2022/part.parquet"
)

# print(df["residence_index"].describe())
# print(df["residence_index"].nunique())
# print(df["residence_index"].value_counts().head(30))
# print(sorted(df["residence_index"].unique())[:50])

# print((df.groupby("species")["residence_index"].describe()))


# print(df.groupby("species")[SPECIES_TARGET].std())

# print(df.groupby("species")[SPECIES_TARGET].mean())

# print(df.groupby("species")["presence_count"].mean())

# print(df.groupby("species")["presence_count"].std())

print(df.groupby("species")[SPECIES_TARGET].describe())