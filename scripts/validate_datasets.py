"""Validate dataset identifiers defined in config.yaml."""

import copernicusmarine

from riskscape.config import cfg


def validate_copernicus(dataset_name, dataset_cfg):
    """Check that a Copernicus dataset exists."""

    dataset_id = dataset_cfg["product"]

    try:
        copernicusmarine.describe(dataset_id)
        print(f"OK: {dataset_name} → {dataset_id}")

    except Exception:
        print(f"ERROR: {dataset_name} → dataset not found")
        print(f"       {dataset_id}")
        raise


def main():

    datasets = cfg["datasets"]

    for name, ds in datasets.items():

        provider = ds["provider"]

        if provider == "copernicus":
            validate_copernicus(name, ds)


if __name__ == "__main__":
    main()