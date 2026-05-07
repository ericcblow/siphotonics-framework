# src/pdk/specs.py

"""Shared device specifications for layout and simulation.

These specs are intended to keep layout parameters and simulation parameters
consistent.
"""

from dataclasses import dataclass

from src.pdk.materials import THICKNESS_SI_UM, WAVELENGTH_UM


@dataclass(frozen=True)
class StripWaveguideSpec:
    """Shared specification for a silicon strip waveguide.

    Units:
        width_um: microns
        thickness_um: microns
        wavelength_um: microns
    """

    width_um: float = 0.5
    thickness_um: float = THICKNESS_SI_UM
    wavelength_um: float = WAVELENGTH_UM
    