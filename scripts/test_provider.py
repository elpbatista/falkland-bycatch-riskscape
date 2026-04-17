"""Test provider auto-discovery and loading."""

from riskscape.providers import get_provider
import pkgutil
import riskscape.providers


def discover_providers():
    """Discover provider modules."""
    return [
        name
        for _, name, _ in pkgutil.iter_modules(riskscape.providers.__path__)
        if name != "__init__"
    ]


def main():
    """Test all providers."""
    names = discover_providers()

    for name in names:
        try:
            provider = get_provider(name)
            print(f"[OK] {name} -> {provider.__name__}")
        except Exception as exc:
            print(f"[FAIL] {name} -> {exc}")


if __name__ == "__main__":
    main()