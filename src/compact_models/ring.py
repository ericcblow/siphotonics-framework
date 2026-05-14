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

    radius_um: float = 8
    wavelength_um: float = WAVELENGTH_UM
    group_index: float = 4.0497


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


def all_pass_ring_through_power(
    wavelengths_um: np.ndarray,
    spec: RingResonatorSpec,
    power_coupling: float = 0.1,
    round_trip_power_loss: float = 0.02,
) -> np.ndarray:
    """Return all-pass ring through-port power transmission.

    Parameters
    ----------
    wavelengths_um:
        Wavelength array in microns.
    spec:
        Ring specification containing radius, wavelength, and group index.
    power_coupling:
        Bus-to-ring power coupling coefficient kappa^2.
        Must satisfy 0 <= power_coupling <= 1.
    round_trip_power_loss:
        Round-trip power loss in the ring.
        Must satisfy 0 <= round_trip_power_loss < 1.

    Returns
    -------
    np.ndarray
        Through-port power transmission versus wavelength.
    """
    if not 0 <= power_coupling <= 1:
        raise ValueError("power_coupling must be between 0 and 1.")

    if not 0 <= round_trip_power_loss < 1:
        raise ValueError("round_trip_power_loss must be between 0 and 1.")

    wavelengths_um = np.asarray(wavelengths_um)

    round_trip_length_um = ring_round_trip_length_um(spec)

    # Convert power quantities to field amplitudes.
    t = np.sqrt(1 - power_coupling)
    a = np.sqrt(1 - round_trip_power_loss)

    # Local phase model around spec.wavelength_um.
    # This makes resonance spacing governed by group index.
    phase = 2 * np.pi * spec.group_index * round_trip_length_um * (
        1 / wavelengths_um - 1 / spec.wavelength_um
    )

    field_transfer = (t - a * np.exp(-1j * phase)) / (
        1 - a * t * np.exp(-1j * phase)
    )

    return np.abs(field_transfer) ** 2


def wavelength_grid_around_center(
    center_wavelength_um: float,
    span_nm: float = 40.0,
    num_points: int = 2001,
) -> np.ndarray:
    """Return wavelength grid centered around a target wavelength."""
    span_um = span_nm / 1000
    return np.linspace(
        center_wavelength_um - span_um / 2,
        center_wavelength_um + span_um / 2,
        num_points,
    )


def save_ring_spectrum_csv(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
    output_path,
) -> None:
    """Save ring spectrum to CSV."""
    import csv
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["wavelength_um", "through_power"],
        )
        writer.writeheader()

        for wavelength_um, through_power in zip(wavelengths_um, transmission):
            writer.writerow(
                {
                    "wavelength_um": float(wavelength_um),
                    "through_power": float(through_power),
                }
            )


def plot_ring_spectrum(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
    output_path,
) -> None:
    """Plot all-pass ring through-port spectrum."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(wavelengths_um * 1000, transmission)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title("All-pass ring through-port spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    spec = RingResonatorSpec(
        radius_um=8.0,
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

    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=spec.wavelength_um,
        span_nm=40.0,
        num_points=2001,
    )

    transmission = all_pass_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.1,
        round_trip_power_loss=0.02,
    )

    spectrum_csv = "data/sweeps/ring_all_pass_spectrum.csv"
    save_ring_spectrum_csv(wavelengths_um, transmission, spectrum_csv)

    spectrum_plot = "results/figures/ring_all_pass_spectrum.png"
    plot_ring_spectrum(wavelengths_um, transmission, spectrum_plot)

    print()
    print(f"Saved ring spectrum to: {spectrum_csv}")
    print(f"Saved ring spectrum plot to: {spectrum_plot}")
