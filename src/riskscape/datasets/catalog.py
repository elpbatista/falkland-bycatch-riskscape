"""Dataset catalog."""

from riskscape.config import cfg


def get_datasets():
    """Return dataset definitions from config."""
    return cfg["datasets"]


def get_dataset(name):
    """Return a single dataset definition."""
    return cfg["datasets"][name]