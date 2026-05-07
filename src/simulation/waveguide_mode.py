# src/simulation/waveguide_mode.py

"""Approximate waveguide mode calculation for a silicon strip waveguide.

This uses a simple effective-index method:

1. Solve the vertical slab waveguide.
2. Use the vertical slab effective index as the core index for the lateral slab.

Note: this is an approximation, not a full-vector eigenmode solve.

Assumptions:
1. Symmetric oxide cladding above and below
2. Perfect rectangular waveguide
3. No sidewall angle
4. No material dispersion beyond constant n
5. Scalar-ish TE slab theory
6. Separability of vertical and lateral confinement
7. No full-vector polarization mixing

Units:
    length: microns
"""

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from scipy.optimize import brentq

from src.pdk.materials import (
    SI_N_1550,
    SIO2_N_1550,
    WAVELENGTH_UM,
    THICKNESS_SI_UM,
)
from src.pdk.specs import StripWaveguideSpec

@dataclass(frozen=True)
class WaveguideCrossSection:
    """Physical cross section of a strip waveguide."""

    width_um: float = 0.5
    thickness_um: float = THICKNESS_SI_UM
    wavelength_um: float = WAVELENGTH_UM
    n_core: float = SI_N_1550
    n_clad: float = SIO2_N_1550


def print_cross_section(xs: WaveguideCrossSection) -> None:
    """Print the physical simulation assumptions."""
    print("Waveguide cross section")
    print("-----------------------")
    print(f"width:      {xs.width_um:.3f} um")
    print(f"thickness:  {xs.thickness_um:.3f} um")
    print(f"wavelength: {xs.wavelength_um:.3f} um")
    print(f"n_core:     {xs.n_core:.3f}")
    print(f"n_clad:     {xs.n_clad:.3f}")


def estimate_effective_index_bounds(xs: WaveguideCrossSection) -> tuple[float, float]:
    """Return simple physical bounds for the effective index."""
    return xs.n_clad, xs.n_core


def solve_symmetric_slab_te0(
    thickness_um: float,
    wavelength_um: float,
    n_core: float,
    n_clad: float,
) -> float:
    """Solve approximate TE0 effective index of a symmetric slab waveguide.

    This solves the even TE0 slab mode equation:

        u tan(u) = w

    with:

        u^2 + w^2 = V^2

    where:

        V = k0 * a * sqrt(n_core^2 - n_clad^2)
        a = thickness / 2

    Parameters
    ----------
    thickness_um:
        Full slab thickness in microns.
    wavelength_um:
        Free-space wavelength in microns.
    n_core:
        Core refractive index.
    n_clad:
        Cladding refractive index.

    Returns
    -------
    float
        Approximate TE0 effective index.
    """
    if n_core <= n_clad:
        raise ValueError("n_core must be greater than n_clad for guided modes.")

    k0 = 2 * np.pi / wavelength_um
    a = thickness_um / 2
    v_number = k0 * a * np.sqrt(n_core**2 - n_clad**2)

    if v_number <= 0:
        raise ValueError("V-number must be positive.")

    def dispersion_equation(u: float) -> float:
        w = np.sqrt(v_number**2 - u**2)
        return u * np.tan(u) - w

    # TE0 even mode has u between 0 and pi/2.
    upper = min(v_number * (1 - 1e-9), np.pi / 2 - 1e-9)
    lower = 1e-9

    u = brentq(dispersion_equation, lower, upper)

    beta = np.sqrt((k0 * n_core) ** 2 - (u / a) ** 2)
    n_eff = beta / k0

    return float(n_eff)


def estimate_rectangular_waveguide_neff_eim(
    xs: WaveguideCrossSection,
) -> tuple[float, float]:
    """Estimate rectangular waveguide effective index using EIM.

    First solve the vertical slab:
        Si core thickness = xs.thickness_um
        cladding = oxide

    Then solve the lateral slab:
        effective core index = vertical slab n_eff
        width = xs.width_um

    Returns
    -------
    tuple[float, float]
        vertical_slab_neff, rectangular_waveguide_neff
    """
    vertical_neff = solve_symmetric_slab_te0(
        thickness_um=xs.thickness_um,
        wavelength_um=xs.wavelength_um,
        n_core=xs.n_core,
        n_clad=xs.n_clad,
    )

    rectangular_neff = solve_symmetric_slab_te0(
        thickness_um=xs.width_um,
        wavelength_um=xs.wavelength_um,
        n_core=vertical_neff,
        n_clad=xs.n_clad,
    )

    return vertical_neff, rectangular_neff


def sweep_widths_eim(
    widths_um: np.ndarray,
    base_xs: WaveguideCrossSection,
) -> list[dict[str, float]]:
    """Sweep waveguide width and estimate n_eff using EIM.

    Parameters
    ----------
    widths_um:
        Array of waveguide widths in microns.
    base_xs:
        Base waveguide cross section. Its width is replaced during the sweep.

    Returns
    -------
    list[dict[str, float]]
        One dictionary per width.
    """
    results = []

    for width_um in widths_um:
        xs = WaveguideCrossSection(
            width_um=float(width_um),
            thickness_um=base_xs.thickness_um,
            wavelength_um=base_xs.wavelength_um,
            n_core=base_xs.n_core,
            n_clad=base_xs.n_clad,
        )

        vertical_neff, rectangular_neff = estimate_rectangular_waveguide_neff_eim(xs)

        results.append(
            {
                "width_um": xs.width_um,
                "vertical_neff": vertical_neff,
                "rectangular_neff": rectangular_neff,
            }
        )

    return results

def save_sweep_results_csv(
    results: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Save width-sweep results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["width_um", "vertical_neff", "rectangular_neff"]

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def plot_width_sweep(
    results: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot EIM effective index versus waveguide width."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    widths_um = [row["width_um"] for row in results]
    neffs = [row["rectangular_neff"] for row in results]

    plt.figure()
    plt.plot(widths_um, neffs, marker="o")
    plt.xlabel("Waveguide width (um)")
    plt.ylabel("Estimated TE-like n_eff")
    plt.title("EIM width sweep for SOI strip waveguide")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    spec = StripWaveguideSpec(width_um=0.5)

    xs = WaveguideCrossSection(
        width_um=spec.width_um,
        thickness_um=spec.thickness_um,
        wavelength_um=spec.wavelength_um,
    )
    print_cross_section(xs)

    n_min, n_max = estimate_effective_index_bounds(xs)
    print()
    print("Expected effective-index bounds")
    print("-------------------------------")
    print(f"{n_min:.3f} < n_eff < {n_max:.3f}")

    vertical_neff, rectangular_neff = estimate_rectangular_waveguide_neff_eim(xs)

    print()
    print("Effective-index method estimate")
    print("-------------------------------")
    print(f"vertical slab n_eff:         {vertical_neff:.4f}")
    print(f"rectangular waveguide n_eff: {rectangular_neff:.4f}")

    widths_um = np.linspace(0.35, 0.70, 8)
    sweep_results = sweep_widths_eim(widths_um, xs)

    print()
    print("Width sweep using EIM")
    print("---------------------")
    print("width_um, rectangular_n_eff")

    for row in sweep_results:
        print(f"{row['width_um']:.3f}, {row['rectangular_neff']:.4f}")

    output_path = Path("data/sweeps/waveguide_width_sweep_eim.csv")
    save_sweep_results_csv(sweep_results, output_path)

    print()
    print(f"Saved sweep results to: {output_path}")

    figure_path = Path("results/figures/waveguide_width_sweep_eim.png")
    plot_width_sweep(sweep_results, figure_path)

    print(f"Saved width-sweep plot to: {figure_path}")
    
    print()
    print("Meep imported successfully.")
    print(f"Meep version: {mp.__version__}")