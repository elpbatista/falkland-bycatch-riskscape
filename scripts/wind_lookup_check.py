import pandas as pd

h3_grid = pd.read_parquet("data/grids/h3_res6_falkland_islands.parquet")
wind_lookup = pd.read_parquet("data/lookups/wind_lookup.parquet")

all_cells = set(h3_grid["id"].astype(str))
covered = set(wind_lookup["h3"].astype(str))

missing = all_cells - covered

print("Total H3 cells:", len(all_cells))
print("Covered by wind lookup:", len(covered))
print("Missing H3 cells:", len(missing))