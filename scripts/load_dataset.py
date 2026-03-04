from riskscape.datasets.catalog import get_dataset

sst = get_dataset("sst")

print(sst["product"])