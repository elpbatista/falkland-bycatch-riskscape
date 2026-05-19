# Reference Data

Reference layers are required for maps, spatial overlays, and Falkland Islands
study-area setup, but the downloaded geospatial files are not tracked in Git.

To restore the reference layers, run:

```bash
python scripts/download_reference_data.py
```

The downloader currently handles:

- Natural Earth 10m land
- Natural Earth 10m coastline
- Falkland Islands fisheries grid squares from the SAERI data portal
- Falkland Islands FICZ/FOCZ conservation zones from the SAERI data portal

If the SAERI portal does not expose a direct downloadable resource, the script
prints the source page and expected local destination for manual placement.
