# src/simulation/waveguide_mode_numeric.py

"""Numerical waveguide mode-solver scaffold.

Goal:
    Define the numerical simulation problem for a 500 nm x 220 nm SOI strip
    waveguide in oxide, then compare future numerical n_eff results against
    the existing EIM estimate.

This file is intentionally conservative:
    - It defines the simulation assumptions.
    - It builds Meep material/geometry objects.
    - It prints sanity checks.
    - It does not yet claim a validated full-vector eigenmode result.

Units:
    length: microns
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import contextlib
import os

import matplotlib.pyplot as plt
import meep as mp
from meep import mpb
import numpy as np
from scipy.optimize import brentq


from src.pdk.materials import SI_N_1550, SIO2_N_1550
from src.pdk.specs import StripWaveguideSpec
from src.simulation.waveguide_mode import (
    WaveguideCrossSection,
    estimate_effective_index_bounds,
    estimate_rectangular_waveguide_neff_eim,
)


@dataclass(frozen=True)
class NumericalModeConfig:
    """Numerical simulation-domain settings for a waveguide mode solve.

    Parameters
    ----------
    padding_um:
        Cladding padding around the waveguide core in the transverse directions.
    resolution_px_per_um:
        Spatial resolution in pixels per micron.
    """

    padding_um: float = 2.0
    resolution_px_per_um: int = 70


@dataclass(frozen=True)
class NumericalWaveguideProblem:
    """Complete numerical waveguide problem definition."""

    spec: StripWaveguideSpec
    config: NumericalModeConfig
    n_core: float = SI_N_1550
    n_clad: float = SIO2_N_1550

    @property
    def cell_width_um(self) -> float:
        """Horizontal simulation-domain width."""
        return self.spec.width_um + 2 * self.config.padding_um

    @property
    def cell_height_um(self) -> float:
        """Vertical simulation-domain height."""
        return self.spec.thickness_um + 2 * self.config.padding_um


def build_meep_materials(problem: NumericalWaveguideProblem) -> tuple[mp.Medium, mp.Medium]:
    """Build Meep material objects for silicon core and oxide cladding."""
    core = mp.Medium(index=problem.n_core)
    clad = mp.Medium(index=problem.n_clad)
    return core, clad


def build_meep_geometry(problem: NumericalWaveguideProblem) -> list[mp.GeometricObject]:
    """Build Meep geometry for a rectangular silicon core in oxide.

    Coordinate convention for this cross-section scaffold:
        x: horizontal transverse direction, waveguide width
        z: vertical transverse direction, silicon thickness
        propagation direction will eventually be the missing/third direction
    """
    core, _ = build_meep_materials(problem)

    silicon_core = mp.Block(
        material=core,
        size=mp.Vector3(problem.spec.width_um, 0, problem.spec.thickness_um),
        center=mp.Vector3(0, 0, 0),
    )

    return [silicon_core]


def print_problem_summary(problem: NumericalWaveguideProblem) -> None:
    """Print numerical mode-problem assumptions."""
    print("Numerical waveguide mode problem")
    print("--------------------------------")
    print(f"waveguide width:       {problem.spec.width_um:.3f} um")
    print(f"silicon thickness:     {problem.spec.thickness_um:.3f} um")
    print(f"wavelength:            {problem.spec.wavelength_um:.3f} um")
    print(f"n_core:                {problem.n_core:.3f}")
    print(f"n_clad:                {problem.n_clad:.3f}")
    print(f"padding:               {problem.config.padding_um:.3f} um")
    print(f"resolution:            {problem.config.resolution_px_per_um} px/um")
    print(f"cell width:            {problem.cell_width_um:.3f} um")
    print(f"cell height:           {problem.cell_height_um:.3f} um")


def compare_against_eim(problem: NumericalWaveguideProblem) -> None:
    """Print EIM result to use as a numerical-solver sanity reference."""
    xs = WaveguideCrossSection(
        width_um=problem.spec.width_um,
        thickness_um=problem.spec.thickness_um,
        wavelength_um=problem.spec.wavelength_um,
        n_core=problem.n_core,
        n_clad=problem.n_clad,
    )

    n_min, n_max = estimate_effective_index_bounds(xs)
    vertical_neff, rectangular_neff = estimate_rectangular_waveguide_neff_eim(xs)

    print()
    print("Reference EIM estimate")
    print("----------------------")
    print(f"physical bounds:                {n_min:.4f} < n_eff < {n_max:.4f}")
    print(f"EIM vertical slab n_eff:         {vertical_neff:.4f}")
    print(f"EIM rectangular waveguide n_eff: {rectangular_neff:.4f}")

def run_mpb_quietly(mode_solver: mpb.ModeSolver) -> None:
    """Run MPB while suppressing verbose stdout output."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            mode_solver.run()

def build_mpb_solver(
    problem: NumericalWaveguideProblem,
    kx: float,
    num_bands: int = 4,
) -> mpb.ModeSolver:
    """Build an MPB mode solver for a given propagation wavevector kx.

    Coordinate convention:
        x: propagation direction
        y: waveguide width direction
        z: waveguide thickness direction
    """
    core = mp.Medium(index=problem.n_core)
    clad = mp.Medium(index=problem.n_clad)

    geometry_lattice = mp.Lattice(
        size=mp.Vector3(
            0,
            problem.cell_width_um,
            problem.cell_height_um,
        )
    )

    geometry = [
        mp.Block(
            material=core,
            size=mp.Vector3(
                mp.inf,
                problem.spec.width_um,
                problem.spec.thickness_um,
            ),
            center=mp.Vector3(0, 0, 0),
        )
    ]

    return mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        default_material=clad,
        resolution=problem.config.resolution_px_per_um,
        num_bands=num_bands,
        k_points=[mp.Vector3(kx, 0, 0)],
    )

def solve_mpb_waveguide_k_and_neff(
    problem: NumericalWaveguideProblem,
    band_num: int = 1,
) -> tuple[float, float]:
    """Solve for propagation wavevector kx and n_eff at target wavelength.

    MPB solves frequency for a given k. We root-find kx such that the selected
    band frequency equals the target frequency.
    """
    target_freq = 1 / problem.spec.wavelength_um

    def mode_frequency_for_k(kx: float) -> float:
        """Return MPB frequency of the selected band for a trial kx."""
        mode_solver = build_mpb_solver(
            problem=problem,
            kx=kx,
            num_bands=max(4, band_num),
        )

        run_mpb_quietly(mode_solver)

        return float(mode_solver.all_freqs[0][band_num - 1])

    k_min = problem.n_clad * target_freq * 1.001
    k_max = problem.n_core * target_freq * 0.999

    def residual(kx: float) -> float:
        return mode_frequency_for_k(kx) - target_freq

    k_solution = brentq(residual, k_min, k_max, xtol=1e-5, rtol=1e-5)
    numerical_neff = k_solution / target_freq

    return float(k_solution), float(numerical_neff)

def solve_mpb_waveguide_neff(
    problem: NumericalWaveguideProblem,
    band_num: int = 1,
) -> float:
    """Estimate waveguide n_eff using MPB by solving for beta at target frequency."""
    _, numerical_neff = solve_mpb_waveguide_k_and_neff(
        problem=problem,
        band_num=band_num,
    )
    return numerical_neff

def sweep_bands_mpb(
    problem: NumericalWaveguideProblem,
    band_nums: list[int],
) -> list[dict[str, float | str]]:
    """Estimate n_eff for several MPB bands.

    This is a simple mode-identity diagnostic. Some bands may not have a
    guided solution at the target wavelength within the search interval.
    Those bands are reported as failed instead of crashing the script.
    """
    results = []

    for band_num in band_nums:
        try:
            neff = solve_mpb_waveguide_neff(problem, band_num=band_num)

            results.append(
                {
                    "band_num": float(band_num),
                    "numerical_neff": neff,
                    "status": "ok",
                }
            )

        except ValueError as error:
            results.append(
                {
                    "band_num": float(band_num),
                    "numerical_neff": float("nan"),
                    "status": f"failed: {error}",
                }
            )

    return results

def sweep_resolution_mpb(
    base_problem: NumericalWaveguideProblem,
    resolutions_px_per_um: list[int],
) -> list[dict[str, float]]:
    """Sweep MPB resolution and estimate numerical n_eff.

    This is a convergence check. A reliable numerical mode result should not
    change significantly as resolution increases.
    """
    results = []

    for resolution in resolutions_px_per_um:
        config = NumericalModeConfig(
            padding_um=base_problem.config.padding_um,
            resolution_px_per_um=resolution,
        )

        problem = NumericalWaveguideProblem(
            spec=base_problem.spec,
            config=config,
            n_core=base_problem.n_core,
            n_clad=base_problem.n_clad,
        )

        neff = solve_mpb_waveguide_neff(problem)

        results.append(
            {
                "resolution_px_per_um": float(resolution),
                "numerical_neff": neff,
            }
        )

    return results


def sweep_padding_mpb(
    base_problem: NumericalWaveguideProblem,
    paddings_um: list[float],
) -> list[dict[str, float]]:
    """Sweep simulation-domain padding and estimate numerical n_eff.

    This is a domain-size convergence check. A reliable numerical mode result
    should not change significantly as the cladding padding increases.
    """
    results = []

    for padding_um in paddings_um:
        config = NumericalModeConfig(
            padding_um=padding_um,
            resolution_px_per_um=base_problem.config.resolution_px_per_um,
        )

        problem = NumericalWaveguideProblem(
            spec=base_problem.spec,
            config=config,
            n_core=base_problem.n_core,
            n_clad=base_problem.n_clad,
        )

        neff = solve_mpb_waveguide_neff(problem)

        results.append(
            {
                "padding_um": float(padding_um),
                "numerical_neff": neff,
                "cell_width_um": problem.cell_width_um,
                "cell_height_um": problem.cell_height_um,
            }
        )

    return results


def extract_band_field_intensity(
    problem: NumericalWaveguideProblem,
    band_num: int = 1,
) -> dict[str, np.ndarray | float]:
    """Extract electric-field intensity for a selected MPB band.

    Returns a dictionary containing:
        y_um: horizontal transverse coordinates
        z_um: vertical transverse coordinates
        intensity: |E|^2 field intensity on the y-z cross section
        epsilon: dielectric profile
        kx: solved propagation wavevector
        neff: effective index
    """
    kx, neff = solve_mpb_waveguide_k_and_neff(problem, band_num=band_num)

    mode_solver = build_mpb_solver(
        problem=problem,
        kx=kx,
        num_bands=max(4, band_num),
    )
    run_mpb_quietly(mode_solver)

    efield = mode_solver.get_efield(band_num, bloch_phase=False)
    epsilon = mode_solver.get_epsilon()

    efield = np.asarray(efield)
    epsilon = np.asarray(epsilon)

    # MPB gives a degenerate first dimension for the propagation direction.
    # Squeeze removes dimensions of length 1.
    efield = np.squeeze(efield)
    epsilon = np.squeeze(epsilon)

    # Handle common MPB array shapes.
    # For vector fields, the last axis is usually field component:
    #     efield[..., 0] = Ex
    #     efield[..., 1] = Ey
    #     efield[..., 2] = Ez
    if efield.ndim == 3 and efield.shape[-1] == 3:
        ex = efield[..., 0]
        ey = efield[..., 1]
        ez = efield[..., 2]
        intensity = (
            np.abs(ex) ** 2
            + np.abs(ey) ** 2
            + np.abs(ez) ** 2
        )
    elif efield.ndim == 2:
        # Fallback for scalar-like field data.
        ex = efield
        ey = np.zeros_like(efield)
        ez = np.zeros_like(efield)
        intensity = np.abs(efield) ** 2
    else:
        raise ValueError(f"Unexpected efield shape after squeeze: {efield.shape}")
    
    # After squeezing, expected shape is approximately (Ny, Nz).
    ny, nz = intensity.shape

    y_um = np.linspace(
        -problem.cell_width_um / 2,
        problem.cell_width_um / 2,
        ny,
    )
    z_um = np.linspace(
        -problem.cell_height_um / 2,
        problem.cell_height_um / 2,
        nz,
    )

    return {
        "y_um": y_um,
        "z_um": z_um,
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "intensity": intensity,
        "epsilon": epsilon,
        "kx": kx,
        "neff": neff,
    }

def save_field_data_npz(
    field_data: dict[str, np.ndarray | float],
    output_path: str | Path,
) -> None:
    """Save extracted field data to compressed NumPy format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        y_um=field_data["y_um"],
        z_um=field_data["z_um"],
        ex=field_data["ex"],
        ey=field_data["ey"],
        ez=field_data["ez"],
        intensity=field_data["intensity"],
        epsilon=field_data["epsilon"],
        kx=field_data["kx"],
        neff=field_data["neff"],
    )

def compute_field_component_fractions(
    field_data: dict[str, np.ndarray | float],
) -> dict[str, float | str]:
    """Compute electric-field component energy fractions.

    Coordinate convention:
        Ex: propagation-direction electric field component
        Ey: horizontal transverse electric field component
        Ez: vertical transverse electric field component

    For this waveguide convention:
        TE-like modes are expected to have dominant Ey.
        TM-like modes are expected to have dominant Ez.
    """
    ex = field_data["ex"]
    ey = field_data["ey"]
    ez = field_data["ez"]

    ex_energy = float(np.sum(np.abs(ex) ** 2))
    ey_energy = float(np.sum(np.abs(ey) ** 2))
    ez_energy = float(np.sum(np.abs(ez) ** 2))

    total_energy = ex_energy + ey_energy + ez_energy

    if total_energy <= 0:
        raise ValueError("Total field energy must be positive.")

    fractions = {
        "ex_fraction": ex_energy / total_energy,
        "ey_fraction": ey_energy / total_energy,
        "ez_fraction": ez_energy / total_energy,
    }

    dominant_component = max(
        fractions,
        key=lambda component: float(fractions[component]),
    )

    if dominant_component == "ey_fraction":
        classification = "TE-like"
    elif dominant_component == "ez_fraction":
        classification = "TM-like"
    else:
        classification = "hybrid/longitudinal"

    return {
        **fractions,
        "dominant_component": dominant_component,
        "classification": classification,
    }

def sweep_resolution_with_polarization_mpb(
    base_problem: NumericalWaveguideProblem,
    resolutions_px_per_um: list[int],
    band_num: int = 1,
) -> list[dict[str, float | str]]:
    """Sweep resolution and record n_eff plus polarization fractions.

    This checks whether the selected band remains the same TE-like physical
    mode as resolution changes.
    """
    results = []

    for resolution in resolutions_px_per_um:
        config = NumericalModeConfig(
            padding_um=base_problem.config.padding_um,
            resolution_px_per_um=resolution,
        )

        problem = NumericalWaveguideProblem(
            spec=base_problem.spec,
            config=config,
            n_core=base_problem.n_core,
            n_clad=base_problem.n_clad,
        )

        field_data = extract_band_field_intensity(
            problem=problem,
            band_num=band_num,
        )

        polarization = compute_field_component_fractions(field_data)

        results.append(
            {
                "resolution_px_per_um": float(resolution),
                "numerical_neff": float(field_data["neff"]),
                "ex_fraction": float(polarization["ex_fraction"]),
                "ey_fraction": float(polarization["ey_fraction"]),
                "ez_fraction": float(polarization["ez_fraction"]),
                "dominant_component": str(polarization["dominant_component"]),
                "classification": str(polarization["classification"]),
            }
        )

    return results

def sweep_wavelength_mpb(
    base_problem: NumericalWaveguideProblem,
    wavelengths_um: list[float],
    band_num: int = 1,
) -> list[dict[str, float | str]]:
    """Sweep wavelength and estimate numerical n_eff.

    This uses the same geometry, padding, and resolution while changing the
    target wavelength in the shared StripWaveguideSpec.

    Note:
        This currently keeps material indices fixed at their 1550 nm values.
        So this captures waveguide dispersion but not material dispersion.
    """
    results = []

    for wavelength_um in wavelengths_um:
        spec = StripWaveguideSpec(
            width_um=base_problem.spec.width_um,
            thickness_um=base_problem.spec.thickness_um,
            wavelength_um=wavelength_um,
        )

        problem = NumericalWaveguideProblem(
            spec=spec,
            config=base_problem.config,
            n_core=base_problem.n_core,
            n_clad=base_problem.n_clad,
        )

        try:
            kx, neff = solve_mpb_waveguide_k_and_neff(
                problem=problem,
                band_num=band_num,
            )

            results.append(
                {
                    "wavelength_um": float(wavelength_um),
                    "kx": float(kx),
                    "numerical_neff": float(neff),
                    "status": "ok",
                }
            )

        except ValueError as error:
            results.append(
                {
                    "wavelength_um": float(wavelength_um),
                    "kx": float("nan"),
                    "numerical_neff": float("nan"),
                    "status": f"failed: {error}",
                }
            )

    return results

def estimate_group_index_from_wavelength_sweep(
    wavelength_results: list[dict[str, float | str]],
    target_wavelength_um: float,
) -> dict[str, float]:
    """Estimate group index from n_eff versus wavelength.

    Uses a quadratic fit around the target wavelength:

        n_eff(lambda) ≈ a lambda^2 + b lambda + c

    Then:

        n_g = n_eff - lambda * dn_eff/dlambda
    """
    successful_results = [
        row for row in wavelength_results
        if row["status"] == "ok"
    ]

    if len(successful_results) < 3:
        raise ValueError("At least 3 successful wavelength points are required.")

    wavelengths = np.array(
        [float(row["wavelength_um"]) for row in successful_results]
    )
    neffs = np.array(
        [float(row["numerical_neff"]) for row in successful_results]
    )

    fit_order = min(2, len(successful_results) - 1)
    coefficients = np.polyfit(wavelengths, neffs, deg=fit_order)
    polynomial = np.poly1d(coefficients)
    derivative = np.polyder(polynomial)

    neff_target = float(polynomial(target_wavelength_um))
    dneff_dlambda = float(derivative(target_wavelength_um))

    group_index = neff_target - target_wavelength_um * dneff_dlambda

    return {
        "target_wavelength_um": float(target_wavelength_um),
        "neff_fit": neff_target,
        "dneff_dlambda": dneff_dlambda,
        "group_index": float(group_index),
        "fit_order": float(fit_order),
        "num_points": float(len(successful_results)),
    }

def sweep_padding_with_polarization_mpb(
    base_problem: NumericalWaveguideProblem,
    paddings_um: list[float],
    band_num: int = 1,
) -> list[dict[str, float | str]]:
    """Sweep padding and record n_eff plus polarization fractions.

    This checks whether the selected band remains the same TE-like physical
    mode as the simulation-domain padding changes.
    """
    results = []

    for padding_um in paddings_um:
        config = NumericalModeConfig(
            padding_um=padding_um,
            resolution_px_per_um=base_problem.config.resolution_px_per_um,
        )

        problem = NumericalWaveguideProblem(
            spec=base_problem.spec,
            config=config,
            n_core=base_problem.n_core,
            n_clad=base_problem.n_clad,
        )

        field_data = extract_band_field_intensity(
            problem=problem,
            band_num=band_num,
        )

        polarization = compute_field_component_fractions(field_data)

        results.append(
            {
                "padding_um": float(padding_um),
                "numerical_neff": float(field_data["neff"]),
                "ex_fraction": float(polarization["ex_fraction"]),
                "ey_fraction": float(polarization["ey_fraction"]),
                "ez_fraction": float(polarization["ez_fraction"]),
                "dominant_component": str(polarization["dominant_component"]),
                "classification": str(polarization["classification"]),
                "cell_width_um": problem.cell_width_um,
                "cell_height_um": problem.cell_height_um,
            }
        )

    return results

def plot_field_intensity(
    field_data: dict[str, np.ndarray | float],
    problem: NumericalWaveguideProblem,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot electric-field intensity with silicon core outline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_um = field_data["y_um"]
    z_um = field_data["z_um"]
    intensity = field_data["intensity"]

    extent = [
        float(y_um[0]),
        float(y_um[-1]),
        float(z_um[0]),
        float(z_um[-1]),
    ]

    plt.figure()
    plt.imshow(
        intensity.T,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(label="|E|^2, arbitrary units")

    # Silicon core outline.
    half_width = problem.spec.width_um / 2
    half_thickness = problem.spec.thickness_um / 2

    y_outline = [
        -half_width,
        half_width,
        half_width,
        -half_width,
        -half_width,
    ]
    z_outline = [
        -half_thickness,
        -half_thickness,
        half_thickness,
        half_thickness,
        -half_thickness,
    ]

    plt.plot(y_outline, z_outline, "w--", linewidth=1.5)

    plt.xlabel("Horizontal coordinate y (um)")
    plt.ylabel("Vertical coordinate z (um)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_wavelength_sweep(
    results: list[dict[str, float | str]],
    output_path: str | Path,
) -> None:
    """Plot numerical n_eff versus wavelength."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    successful_results = [
        row for row in results
        if row["status"] == "ok"
    ]

    wavelengths = [row["wavelength_um"] for row in successful_results]
    neffs = [row["numerical_neff"] for row in successful_results]

    plt.figure()
    plt.plot(wavelengths, neffs, marker="o")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Numerical n_eff")
    plt.title("MPB wavelength sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_field_components(
    field_data: dict[str, np.ndarray | float],
    problem: NumericalWaveguideProblem,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot |Ex|^2, |Ey|^2, and |Ez|^2 field components."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_um = field_data["y_um"]
    z_um = field_data["z_um"]

    ex = field_data["ex"]
    ey = field_data["ey"]
    ez = field_data["ez"]

    component_data = [
        ("|Ex|^2", np.abs(ex) ** 2),
        ("|Ey|^2", np.abs(ey) ** 2),
        ("|Ez|^2", np.abs(ez) ** 2),
    ]

    extent = [
        float(y_um[0]),
        float(y_um[-1]),
        float(z_um[0]),
        float(z_um[-1]),
    ]

    half_width = problem.spec.width_um / 2
    half_thickness = problem.spec.thickness_um / 2

    y_outline = [
        -half_width,
        half_width,
        half_width,
        -half_width,
        -half_width,
    ]
    z_outline = [
        -half_thickness,
        -half_thickness,
        half_thickness,
        half_thickness,
        -half_thickness,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    for ax, (component_name, values) in zip(axes, component_data):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        ax.plot(y_outline, z_outline, "w--", linewidth=1.2)
        ax.set_title(component_name)
        ax.set_xlabel("y (um)")
        ax.set_ylabel("z (um)")
        fig.colorbar(image, ax=ax)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def plot_padding_field_comparison(
    base_problem: NumericalWaveguideProblem,
    paddings_um: list[float],
    output_path: str | Path,
    band_num: int = 1,
) -> None:
    """Plot |E|^2 field profiles for several padding values.

    This checks whether the spatial mode profile changes as the simulation
    domain grows. It is a field-shape diagnostic for padding/domain convergence.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_cases = []

    for padding_um in paddings_um:
        config = NumericalModeConfig(
            padding_um=padding_um,
            resolution_px_per_um=base_problem.config.resolution_px_per_um,
        )

        problem = NumericalWaveguideProblem(
            spec=base_problem.spec,
            config=config,
            n_core=base_problem.n_core,
            n_clad=base_problem.n_clad,
        )

        field_data = extract_band_field_intensity(
            problem=problem,
            band_num=band_num,
        )

        field_cases.append((padding_um, problem, field_data))

    # Use a common color scale so visual comparisons are meaningful.
    max_intensity = max(
        float(np.max(field_data["intensity"]))
        for _, _, field_data in field_cases
    )

    fig, axes = plt.subplots(
        1,
        len(field_cases),
        figsize=(4 * len(field_cases), 3.5),
        constrained_layout=True,
    )

    if len(field_cases) == 1:
        axes = [axes]

    for ax, (padding_um, problem, field_data) in zip(axes, field_cases):
        y_um = field_data["y_um"]
        z_um = field_data["z_um"]
        intensity = field_data["intensity"]

        extent = [
            float(y_um[0]),
            float(y_um[-1]),
            float(z_um[0]),
            float(z_um[-1]),
        ]

        image = ax.imshow(
            intensity.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=0,
            vmax=max_intensity,
        )

        # Silicon core outline.
        half_width = problem.spec.width_um / 2
        half_thickness = problem.spec.thickness_um / 2

        y_outline = [
            -half_width,
            half_width,
            half_width,
            -half_width,
            -half_width,
        ]
        z_outline = [
            -half_thickness,
            -half_thickness,
            half_thickness,
            half_thickness,
            -half_thickness,
        ]

        ax.plot(y_outline, z_outline, "w--", linewidth=1.2)

        ax.set_title(
            f"padding={padding_um:.1f} um\n"
            f"n_eff={field_data['neff']:.4f}"
        )
        ax.set_xlabel("y (um)")
        ax.set_ylabel("z (um)")

    fig.colorbar(image, ax=axes, label="|E|^2, common scale")
    fig.suptitle("MPB band 1 field profile versus padding")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_numeric_sweep_csv(
    results: list[dict[str, float | str]],
    output_path: str | Path,
) -> None:
    """Save numerical convergence sweep results to CSV."""
    if not results:
        raise ValueError("Cannot save empty sweep results.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def plot_resolution_sweep(
    results: list[dict[str, float]],
    output_path: str | Path,
) -> None:
    """Plot numerical n_eff versus MPB resolution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolutions = [row["resolution_px_per_um"] for row in results]
    neffs = [row["numerical_neff"] for row in results]

    plt.figure()
    plt.plot(resolutions, neffs, marker="o")
    plt.xlabel("Resolution (px/um)")
    plt.ylabel("Numerical n_eff")
    plt.title("MPB resolution convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_padding_sweep(
    results: list[dict[str, float]],
    output_path: str | Path,
) -> None:
    """Plot numerical n_eff versus cladding padding."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paddings = [row["padding_um"] for row in results]
    neffs = [row["numerical_neff"] for row in results]

    plt.figure()
    plt.plot(paddings, neffs, marker="o")
    plt.xlabel("Padding (um)")
    plt.ylabel("Numerical n_eff")
    plt.title("MPB padding convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_band_diagnostic(
    results: list[dict[str, float | str]],
    output_path: str | Path,
) -> None:
    """Plot numerical n_eff for successful MPB band roots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    successful_results = [row for row in results if row["status"] == "ok"]

    bands = [row["band_num"] for row in successful_results]
    neffs = [row["numerical_neff"] for row in successful_results]

    plt.figure()
    plt.plot(bands, neffs, marker="o")
    plt.xlabel("MPB band number")
    plt.ylabel("Numerical n_eff")
    plt.title("MPB band diagnostic at target wavelength")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """Build and summarize the numerical mode problem."""
    spec = StripWaveguideSpec(width_um=0.5)
    config = NumericalModeConfig(padding_um=1.5, resolution_px_per_um=70)
    problem = NumericalWaveguideProblem(spec=spec, config=config)

    print_problem_summary(problem)

    geometry = build_meep_geometry(problem)
    core, clad = build_meep_materials(problem)

    print()
    print("Meep objects")
    print("------------")
    print(f"geometry objects: {len(geometry)}")
    print(f"core epsilon:     {core.epsilon(1 / spec.wavelength_um)[0][0].real:.4f}")
    print(f"clad epsilon:     {clad.epsilon(1 / spec.wavelength_um)[0][0].real:.4f}")

    compare_against_eim(problem)

    print()
    print("Numerical MPB estimate")
    print("----------------------")

    numerical_neff = solve_mpb_waveguide_neff(problem)
    print(f"MPB numerical n_eff: {numerical_neff:.4f}")

    print()
    print("Band diagnostic at target wavelength")
    print("------------------------------------")

    band_results = sweep_bands_mpb(
        problem=problem,
        band_nums=[1, 2, 3, 4],
    )

    print("band_num, numerical_n_eff")
    for row in band_results:
        print(
            f"{row['band_num']:.0f}, "
            f"{row['numerical_neff']:.6f}"
        )

    band_csv = Path("data/sweeps/waveguide_mpb_band_diagnostic.csv")
    save_numeric_sweep_csv(band_results, band_csv)
    print(f"Saved band diagnostic to: {band_csv}")

    band_plot = Path("results/figures/waveguide_mpb_band_diagnostic.png")
    plot_band_diagnostic(band_results, band_plot)
    print(f"Saved band diagnostic plot to: {band_plot}")

    print()
    print("Resolution convergence sweep")
    print("----------------------------")

    resolution_results = sweep_resolution_mpb(
        base_problem=problem,
        resolutions_px_per_um=[20, 30, 40, 50],
    )

    print("resolution_px_per_um, numerical_n_eff")
    for row in resolution_results:
        print(
            f"{row['resolution_px_per_um']:.0f}, "
            f"{row['numerical_neff']:.6f}"
        )

    resolution_csv = Path("data/sweeps/waveguide_mpb_resolution_sweep.csv")
    save_numeric_sweep_csv(resolution_results, resolution_csv)
    print(f"Saved resolution sweep to: {resolution_csv}")

    resolution_plot = Path("results/figures/waveguide_mpb_resolution_sweep.png")
    plot_resolution_sweep(resolution_results, resolution_plot)
    print(f"Saved resolution sweep plot to: {resolution_plot}")
    
    print()
    print("Resolution + polarization sweep")
    print("-------------------------------")

    resolution_polarization_results = sweep_resolution_with_polarization_mpb(
        base_problem=problem,
        resolutions_px_per_um=[30, 40, 50, 60, 70, 80],
        band_num=1,
    )

    print(
        "resolution_px_per_um, numerical_n_eff, "
        "ex_fraction, ey_fraction, ez_fraction, classification"
    )
    for row in resolution_polarization_results:
        print(
            f"{row['resolution_px_per_um']:.0f}, "
            f"{row['numerical_neff']:.6f}, "
            f"{row['ex_fraction']:.4f}, "
            f"{row['ey_fraction']:.4f}, "
            f"{row['ez_fraction']:.4f}, "
            f"{row['classification']}"
        )

    resolution_polarization_csv = Path(
        "data/sweeps/waveguide_mpb_resolution_polarization_sweep.csv"
    )
    save_numeric_sweep_csv(
        resolution_polarization_results,
        resolution_polarization_csv,
    )
    print(
        "Saved resolution + polarization sweep to: "
        f"{resolution_polarization_csv}"
    )

    print()
    print("Field profile diagnostic")
    print("------------------------")

    band1_field = extract_band_field_intensity(problem, band_num=1)

    field_npz = Path("data/fields/waveguide_mpb_band1_field.npz")
    save_field_data_npz(band1_field, field_npz)

    field_plot = Path("results/figures/waveguide_mpb_band1_field.png")
    plot_field_intensity(
        field_data=band1_field,
        problem=problem,
        output_path=field_plot,
        title=f"MPB band 1 |E|^2, n_eff={band1_field['neff']:.4f}",
    )

    print(f"Saved band 1 field data to: {field_npz}")
    print(f"Saved band 1 field plot to: {field_plot}")

    components_plot = Path("results/figures/waveguide_mpb_band1_components.png")
    plot_field_components(
        field_data=band1_field,
        problem=problem,
        output_path=components_plot,
        title=f"MPB band 1 field components, n_eff={band1_field['neff']:.4f}",
    )

    print(f"Saved band 1 component plot to: {components_plot}")

    polarization = compute_field_component_fractions(band1_field)

    print()
    print("Polarization diagnostic for band 1")
    print("----------------------------------")
    print(f"Ex fraction:        {polarization['ex_fraction']:.4f}")
    print(f"Ey fraction:        {polarization['ey_fraction']:.4f}")
    print(f"Ez fraction:        {polarization['ez_fraction']:.4f}")
    print(f"Dominant component: {polarization['dominant_component']}")
    print(f"Classification:     {polarization['classification']}")

    polarization_csv = Path("data/sweeps/waveguide_mpb_band1_polarization.csv")
    save_numeric_sweep_csv([polarization], polarization_csv)
    print(f"Saved polarization diagnostic to: {polarization_csv}")

    print()
    print("Padding convergence sweep")
    print("-------------------------")

    padding_results = sweep_padding_mpb(
        base_problem=problem,
        paddings_um=[1.0, 1.5, 2.0, 2.5],
    )

    print("padding_um, numerical_n_eff, cell_width_um, cell_height_um")
    for row in padding_results:
        print(
            f"{row['padding_um']:.1f}, "
            f"{row['numerical_neff']:.6f}, "
            f"{row['cell_width_um']:.3f}, "
            f"{row['cell_height_um']:.3f}"
        )

    padding_csv = Path("data/sweeps/waveguide_mpb_padding_sweep.csv")
    save_numeric_sweep_csv(padding_results, padding_csv)
    print(f"Saved padding sweep to: {padding_csv}")

    padding_plot = Path("results/figures/waveguide_mpb_padding_sweep.png")
    plot_padding_sweep(padding_results, padding_plot)
    print(f"Saved padding sweep plot to: {padding_plot}")
    
    print()
    print("Wavelength sweep")
    print("----------------")

    wavelength_results = sweep_wavelength_mpb(
        base_problem=problem,
        wavelengths_um=[1.50, 1.525, 1.55, 1.575, 1.60],
        band_num=1,
    )

    print("wavelength_um, numerical_n_eff, status")
    for row in wavelength_results:
        print(
            f"{row['wavelength_um']:.3f}, "
            f"{row['numerical_neff']:.6f}, "
            f"{row['status']}"
        )

    wavelength_csv = Path("data/sweeps/waveguide_mpb_wavelength_sweep.csv")
    save_numeric_sweep_csv(wavelength_results, wavelength_csv)
    print(f"Saved wavelength sweep to: {wavelength_csv}")

    wavelength_plot = Path("results/figures/waveguide_mpb_wavelength_sweep.png")
    plot_wavelength_sweep(wavelength_results, wavelength_plot)
    print(f"Saved wavelength sweep plot to: {wavelength_plot}")

    group_index_result = estimate_group_index_from_wavelength_sweep(
        wavelength_results=wavelength_results,
        target_wavelength_um=problem.spec.wavelength_um,
    )

    print()
    print("Group index estimate")
    print("--------------------")
    print(f"target wavelength: {group_index_result['target_wavelength_um']:.4f} um")
    print(f"n_eff from fit:    {group_index_result['neff_fit']:.6f}")
    print(f"dn_eff/dlambda:    {group_index_result['dneff_dlambda']:.6f} 1/um")
    print(f"group index:       {group_index_result['group_index']:.6f}")

    group_index_csv = Path("data/sweeps/waveguide_mpb_group_index.csv")
    save_numeric_sweep_csv([group_index_result], group_index_csv)
    print(f"Saved group index estimate to: {group_index_csv}")

    print()
    print("Padding + polarization sweep")
    print("----------------------------")

    padding_polarization_results = sweep_padding_with_polarization_mpb(
        base_problem=problem,
        paddings_um=[1.0, 1.5, 2.0, 2.5, 3.0],
        band_num=1,
    )

    print(
        "padding_um, numerical_n_eff, "
        "ex_fraction, ey_fraction, ez_fraction, classification"
    )
    for row in padding_polarization_results:
        print(
            f"{row['padding_um']:.1f}, "
            f"{row['numerical_neff']:.6f}, "
            f"{row['ex_fraction']:.4f}, "
            f"{row['ey_fraction']:.4f}, "
            f"{row['ez_fraction']:.4f}, "
            f"{row['classification']}"
        )

    padding_polarization_csv = Path(
        "data/sweeps/waveguide_mpb_padding_polarization_sweep.csv"
    )
    save_numeric_sweep_csv(
        padding_polarization_results,
        padding_polarization_csv,
    )
    print(
        "Saved padding + polarization sweep to: "
        f"{padding_polarization_csv}"
    )

    padding_field_comparison_plot = Path(
        "results/figures/waveguide_mpb_padding_field_comparison.png"
    )
    plot_padding_field_comparison(
        base_problem=problem,
        paddings_um=[1.5, 2.0, 2.5, 3.0],
        output_path=padding_field_comparison_plot,
        band_num=1,
    )
    print(
        "Saved padding field comparison plot to: "
        f"{padding_field_comparison_plot}"
    )


if __name__ == "__main__":
    main()