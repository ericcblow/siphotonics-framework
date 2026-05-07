# src/devices/straight.py

"""Straight waveguide layout cell."""

import gdsfactory as gf
from gdsfactory.gpdk import get_generic_pdk

from src.pdk.cross_sections import strip


get_generic_pdk().activate()


@gf.cell
def straight_waveguide(length: float = 10.0, width: float = 0.5):
    """Straight silicon strip waveguide.

    Parameters
    ----------
    length:
        Waveguide length in microns.
    width:
        Waveguide width in microns.
    """
    return gf.components.straight(
        length=length,
        cross_section=strip(width=width),
    )


if __name__ == "__main__":
    c = straight_waveguide(length=10.0, width=0.5)
    c.write_gds("straight_waveguide.gds")
    c.show()