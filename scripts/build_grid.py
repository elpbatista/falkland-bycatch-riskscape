"""Build the H3 grid defined in the active configuration."""

from riskcape.grid import build_h3_grid


def main():
    """Run grid generation."""
    print("Building H3 grid...")
    build_h3_grid()
    print("Grid generation complete.")


if __name__ == "__main__":
    main()