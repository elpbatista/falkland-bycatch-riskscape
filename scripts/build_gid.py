"""Build the H3 grid defined in config.yaml."""

from riskscape.grid import build_h3_grid


def main():
    """Run the grid generation pipeline."""
    build_h3_grid()


if __name__ == "__main__":
    main()