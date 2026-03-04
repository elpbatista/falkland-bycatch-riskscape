from riskscape.datasets.catalog import get_datasets

datasets = get_datasets()

for name, ds in datasets.items():
    print(name, ds["provider"])
