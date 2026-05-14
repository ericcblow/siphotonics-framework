# src/compact_models/ring.py

"""Simple ring-resonator compact-model estimates.

This module uses waveguide-level quantities such as group index to estimate
ring-level quantities such as free spectral range.

Units:
    length: microns
"""

from dataclasses import dataclass

import numpy as np

from src.pdk.materials import WAVELENGTH_UM


@dataclass(frozen=True)
class RingResonatorSpec:
    """Basic ring resonator specification."""

    radius_um: float = 10.0
    wavelength_um: float = WAVELENGTH_UM
    group_index: float = 4.0


def ring_round_trip_length_um(spec: RingResonatorSpec) -> float:
    """Return ring round-trip length."""
    return float(2 * np.pi * spec.radius_um)


def estimate_ring_fsr_um(spec: RingResonatorSpec) -> float:
    """Estimate ring free spectral range in microns.

    Approximation:
        FSR ≈ lambda^2 / (n_g * L_rt)

    This is valid for small wavelength spacing near the target wavelength.
    """
    round_trip_length_um = ring_round_trip_length_um(spec)

    fsr_um = spec.wavelength_um**2 / (
        spec.group_index * round_trip_length_um
    )

    return float(fsr_um)


def estimate_ring_fsr_nm(spec: RingResonatorSpec) -> float:
    """Estimate ring free spectral range in nanometers."""
    return 1000 * estimate_ring_fsr_um(spec)


if __name__ == "__main__":
    spec = RingResonatorSpec(
        radius_um=10.0,
        wavelength_um=1.55,
        group_index=4.0497,
    )

    round_trip_length_um = ring_round_trip_length_um(spec)
    fsr_um = estimate_ring_fsr_um(spec)
    fsr_nm = estimate_ring_fsr_nm(spec)

    print("Ring resonator estimate")
    print("-----------------------")
    print(f"radius:             {spec.radius_um:.3f} um")
    print(f"wavelength:         {spec.wavelength_um:.3f} um")
    print(f"group index:        {spec.group_index:.3f}")
    print(f"round-trip length:  {round_trip_length_um:.3f} um")
    print(f"FSR:                {fsr_um:.6f} um")
    print(f"FSR:                {fsr_nm:.3f} nm")