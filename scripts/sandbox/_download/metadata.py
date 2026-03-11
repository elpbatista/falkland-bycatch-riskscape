"""Dataset metadata utilities."""

import datetime
import json

from riskscape.config import cfg


def write_dataset_metadata(dataset_name, dataset_cfg, output_dir):
    """Write metadata.json describing the downloaded dataset."""

    metadata = {
        "dataset": dataset_name,
        "provider": dataset_cfg.get("provider"),
        "product": dataset_cfg.get("product"),
        "variable": dataset_cfg.get("variable"),
        "download_date": datetime.date.today().isoformat(),
        "time_range": {
            "start": cfg["time"]["start"],
            "end": cfg["time"]["end"],
        },
        "base_url": dataset_cfg.get("base_url"),
    }

    metadata_file = output_dir / "metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)