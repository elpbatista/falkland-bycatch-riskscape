"""Provider loader."""

import importlib


def get_provider(name):
    """Return provider module by name."""
    module_name = f"riskscape.providers.{name}"

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Unknown provider: {name}") from exc
        raise