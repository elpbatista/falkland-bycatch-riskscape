"""Plot H3 bathymetry map with land on top."""

import matplotlib.pyplot as plt

from riskscape.visualization.base_map import (
    draw_bathymetry_base_layer,
    draw_reference_layers,
    load_reference_layers,
    setup_map,
)


def main() -> int:
    """Plot bathymetry from static H3 features."""
    land, coast = load_reference_layers()
    fig, ax, bbox_gdf = setup_map()

    draw_bathymetry_base_layer(ax)
    draw_reference_layers(ax, bbox_gdf, land, coast)

    ax.set_title("Bathymetry Depth Map (H3 mean depth, m)")
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
