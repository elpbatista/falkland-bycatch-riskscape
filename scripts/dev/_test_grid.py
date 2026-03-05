import geopandas as gpd

gdf = gpd.read_parquet(
    "data/grids/h3_res6_falkland_islands.parquet"
)

print(gdf.head())