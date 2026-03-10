import numpy as np
import xarray as xr
from pathlib import Path

from riskscape.config import paths

chl_dir = Path(paths["raw"]) / "chl"
files = sorted(chl_dir.glob("*.nc"))

if not files:
    raise RuntimeError(f"No NetCDF files found in {chl_dir}")

ds = xr.open_dataset(files[0])

chl = ds["CHL"].values
chl = chl[np.isfinite(chl)]

zero_count = int((chl == 0).sum())
positive_vals = chl[chl > 0]

min_positive = float(positive_vals.min()) if positive_vals.size else float("inf")

print("Total non-NaN values:", int(chl.size))
print("Total zero values:", zero_count)
print("Minimum positive chlorophyll:", min_positive)

ds.close()