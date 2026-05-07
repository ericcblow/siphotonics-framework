# src/pdk/cross_sections.py

"""gdsfactory cross sections for the tutorial SOI platform."""

import gdsfactory as gf

from src.pdk.layers import WG


def strip(width: float = 0.5):
    """Simple silicon strip waveguide cross section.

    Parameters
    ----------
    width:
        Waveguide width in microns.
    """
    return gf.cross_section.strip(
        width=width,
        layer=WG,
    )
