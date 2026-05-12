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

import meep as mp
from meep import mpb
from scipy.optimize import brentq
import contextlib
import os

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

    padding_um: float = 1.5
    resolution_px_per_um: int = 40


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

def solve_mpb_waveguide_neff(
    problem: NumericalWaveguideProblem,
    band_num: int = 1,
) -> float:
    """Estimate waveguide n_eff using MPB by solving for beta at target frequency.

    MPB normally solves for frequency at a given propagation constant k.
    But for photonic design, we usually know the wavelength/frequency and want
    the propagation constant beta, or equivalently n_eff.

    So we:
        1. Pick a trial k.
        2. Ask MPB for the mode frequency.
        3. Root-find k such that MPB frequency = target frequency.
        4. Convert k to n_eff using n_eff = k / f.

    Notes
    -----
    Units:
        length unit = microns
        target frequency = 1 / wavelength_um

    Coordinate convention:
        x: propagation direction
        y: waveguide width direction
        z: waveguide thickness direction
    """
    target_freq = 1 / problem.spec.wavelength_um

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

    def mode_frequency_for_k(kx: float) -> float:
        """Return MPB frequency of the selected band for a trial kx."""
        mode_solver = mpb.ModeSolver(
            geometry_lattice=geometry_lattice,
            geometry=geometry,
            default_material=clad,
            resolution=problem.config.resolution_px_per_um,
            num_bands=max(4, band_num),
            k_points=[mp.Vector3(kx, 0, 0)],
        )

        run_mpb_quietly(mode_solver)

        return float(mode_solver.all_freqs[0][band_num - 1])

    k_min = problem.n_clad * target_freq * 1.001
    k_max = problem.n_core * target_freq * 0.999

    def residual(kx: float) -> float:
        return mode_frequency_for_k(kx) - target_freq

    k_solution = brentq(residual, k_min, k_max, xtol=1e-5, rtol=1e-5)
    numerical_neff = k_solution / target_freq

    return float(numerical_neff)

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


def save_numeric_sweep_csv(
    results: list[dict[str, float]],
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


def main() -> None:
    """Build and summarize the numerical mode problem."""
    spec = StripWaveguideSpec(width_um=0.5)
    config = NumericalModeConfig(padding_um=1.5, resolution_px_per_um=40)
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

    print()
    print("Status")
    print("------")
    print("Numerical problem scaffold created successfully.")
    print("Next step: add plots and reduce MPB verbosity.")


if __name__ == "__main__":
    main()