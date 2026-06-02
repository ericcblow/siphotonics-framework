"""
Finite directional coupler FDTD length sweep.

Purpose:
- Move beyond infinite supermode theory.
- Simulate a finite-length directional coupler in 2D effective-index form.
- Extract through/cross flux versus coupling length.
- Compare FDTD output to ideal supermode prediction.

This is a learning model, not a final 3D silicon-photonics design solver.

Coordinate convention:
    x = propagation direction
    y = lateral waveguide/gap direction

2D effective-index approximation:
    core index uses approximate vertical slab effective index
    cladding index uses oxide index
"""

# WARNING:
# This file is a flux-diagnostic learning script.
# It does not extract true modal S-parameters.
# Use directional_coupler_sparameters.py for real port-based extraction.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectionalCouplerFDTDSpec:
    """2D effective-index finite directional coupler specification."""

    width_um: float = 0.5
    gap_um: float = 0.20
    wavelength_um: float = 1.55

    # Effective-index approximation for 2D top-view model.
    # Use the EIM vertical slab value, not the full rectangular n_eff.
    core_index_2d: float = 2.85
    cladding_index: float = 1.444

    # From MPB supermode extraction at gap ~= 0.20 um, resolution 50.
    l_full_um: float = 38.2


@dataclass(frozen=True)
class FDTDSettings:
    """Numerical settings for 2D Meep simulation."""

    resolution: int = 30
    pml_um: float = 1.0
    input_straight_um: float = 4.0
    output_straight_um: float = 4.0
    y_padding_um: float = 2.0
    runtime: float = 250.0

    max_coupler_length_um: float = 40.0
    monitor_width_um: float = 2.0
    source_width_um: float = 2.0


def ideal_cross_power(length_um: float, l_full_um: float) -> float:
    """Ideal supermode-predicted cross power."""
    if length_um < 0:
        raise ValueError("length_um must be nonnegative.")
    if l_full_um <= 0:
        raise ValueError("l_full_um must be positive.")

    return float(np.sin(np.pi * length_um / (2.0 * l_full_um)) ** 2)


def ideal_through_power(length_um: float, l_full_um: float) -> float:
    """Ideal supermode-predicted through power."""
    return 1.0 - ideal_cross_power(length_um=length_um, l_full_um=l_full_um)


def make_coupler_geometry(
    coupler_length_um: float,
    spec: DirectionalCouplerFDTDSpec,
    settings: FDTDSettings,
    include_top_waveguide: bool = True,
) -> tuple[list[mp.GeometricObject], mp.Vector3, dict[str, float]]:
    """
    Build a fixed-cell 2D finite directional-coupler geometry.

    Geometry:
        bottom waveguide:
            full fixed device length

        top waveguide:
            only the finite coupling section of length coupler_length_um

    This avoids changing the source-monitor distance as the swept coupling
    length changes.
    """
    if coupler_length_um < 0:
        raise ValueError("coupler_length_um must be nonnegative.")

    total_length_um = (
        settings.input_straight_um
        + settings.max_coupler_length_um
        + settings.output_straight_um
    )

    sx = total_length_um + 2.0 * settings.pml_um
    sy = (
        2.0 * settings.y_padding_um
        + 2.0 * spec.width_um
        + spec.gap_um
    )

    center_to_center = spec.width_um + spec.gap_um

    y_bottom = -center_to_center / 2.0
    y_top = +center_to_center / 2.0

    core = mp.Medium(index=spec.core_index_2d)

    geometry: list[mp.GeometricObject] = [
        mp.Block(
            size=mp.Vector3(total_length_um, spec.width_um, mp.inf),
            center=mp.Vector3(0, y_bottom, 0),
            material=core,
        )
    ]

    if include_top_waveguide and coupler_length_um > 0:
        geometry.append(
            mp.Block(
                size=mp.Vector3(coupler_length_um, spec.width_um, mp.inf),
                center=mp.Vector3(0, y_top, 0),
                material=core,
            )
        )

    cell_size = mp.Vector3(sx, sy, 0)

    positions = {
        "x_source": -total_length_um / 2.0 + 1.0,
        "x_monitor": +total_length_um / 2.0 - 1.0,
        "y_bottom": y_bottom,
        "y_top": y_top,
        "total_length_um": total_length_um,
    }

    return geometry, cell_size, positions

def run_straight_reference_flux(
    spec: DirectionalCouplerFDTDSpec,
    settings: FDTDSettings,
) -> dict[str, float]:
    """
    Run a straight bottom-waveguide reference simulation.

    This is diagnostic normalization data, not a true modal S-parameter
    normalization. It records both local bottom-guide flux and a wider
    output-plane flux.
    """
    geometry, cell_size, positions = make_coupler_geometry(
        coupler_length_um=0.0,
        spec=spec,
        settings=settings,
        include_top_waveguide=False,
    )

    frequency = 1.0 / spec.wavelength_um
    fwidth = 0.10 * frequency

    source = make_mode_source(
        frequency=frequency,
        fwidth=fwidth,
        x_source=positions["x_source"],
        y_source=positions["y_bottom"],
        source_width_um=settings.source_width_um,
    )

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(settings.pml_um)],
        geometry=geometry,
        sources=[source],
        default_material=mp.Medium(index=spec.cladding_index),
        resolution=settings.resolution,
        dimensions=2,
    )

    local_reference_monitor = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(positions["x_monitor"], positions["y_bottom"], 0),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )

    total_reference_monitor = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(positions["x_monitor"], 0, 0),
            size=mp.Vector3(0, cell_size.y - 2.0 * settings.pml_um, 0),
        ),
    )

    sim.run(until=settings.runtime)

    local_reference_flux = float(mp.get_fluxes(local_reference_monitor)[0])
    total_reference_flux = float(mp.get_fluxes(total_reference_monitor)[0])

    if abs(total_reference_flux) < 1e-12:
        raise RuntimeError(f"Total reference flux is too small: {total_reference_flux}")

    if abs(local_reference_flux) < 1e-12:
        print(
            "Warning: local reference flux is very small "
            f"({local_reference_flux:.3e}). "
            "Local reference-normalized diagnostics will be NaN."
        )

    return {
        "local_reference_flux": local_reference_flux,
        "total_reference_flux": total_reference_flux,
    }

def make_mode_source(
    frequency: float,
    fwidth: float,
    x_source: float,
    y_source: float,
    source_width_um: float,
) -> mp.Source:
    """
    Create a simple 2D Ez line source.

    This is intentionally not a true guided-mode port source.
    It is only for flux-diagnostic FDTD experiments.
    """
    return mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=fwidth),
        component=mp.Ez,
        center=mp.Vector3(x_source, y_source, 0),
        size=mp.Vector3(0, source_width_um, 0),
    )

def run_single_length_fdtd(
    coupler_length_um: float,
    spec: DirectionalCouplerFDTDSpec,
    settings: FDTDSettings,
    reference: dict[str, float],
) -> dict[str, float]:
    """
    Run one finite directional-coupler FDTD simulation.

    Important:
    This is still a flux-diagnostic simulation, not a true modal
    S-parameter extraction.

    The bounded through_power/cross_power columns are local output fractions:

        through_power = through_flux / (through_flux + cross_flux)
        cross_power   = cross_flux   / (through_flux + cross_flux)

    Reference-normalized raw fluxes are also saved for diagnostics, but should
    not be interpreted as true guided-mode powers.
    """
    geometry, cell_size, positions = make_coupler_geometry(
        coupler_length_um=coupler_length_um,
        spec=spec,
        settings=settings,
        include_top_waveguide=True,
    )

    frequency = 1.0 / spec.wavelength_um
    fwidth = 0.10 * frequency

    source = make_mode_source(
        frequency=frequency,
        fwidth=fwidth,
        x_source=positions["x_source"],
        y_source=positions["y_bottom"],
        source_width_um=settings.source_width_um,
    )

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(settings.pml_um)],
        geometry=geometry,
        sources=[source],
        default_material=mp.Medium(index=spec.cladding_index),
        resolution=settings.resolution,
        dimensions=2,
    )

    through_monitor = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(positions["x_monitor"], positions["y_bottom"], 0),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )

    cross_monitor = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(positions["x_monitor"], positions["y_top"], 0),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )

    total_output_monitor = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(positions["x_monitor"], 0, 0),
            size=mp.Vector3(0, cell_size.y - 2.0 * settings.pml_um, 0),
        ),
    )

    sim.run(until=settings.runtime)

    through_flux = float(mp.get_fluxes(through_monitor)[0])
    cross_flux = float(mp.get_fluxes(cross_monitor)[0])
    total_output_flux = float(mp.get_fluxes(total_output_monitor)[0])

    local_output_flux = through_flux + cross_flux

    if abs(local_output_flux) < 1e-18:
        through_power = np.nan
        cross_power = np.nan
    else:
        through_power = through_flux / local_output_flux
        cross_power = cross_flux / local_output_flux

    # Diagnostic reference-normalized values.
    # These are not true S-parameter powers.
    local_ref = reference.get("local_reference_flux", np.nan)
    total_ref = reference.get("total_reference_flux", np.nan)

    if not np.isfinite(local_ref) or abs(local_ref) < 1e-12:
        through_flux_ref_norm = np.nan
        cross_flux_ref_norm = np.nan
        local_output_flux_ref_norm = np.nan
    else:
        through_flux_ref_norm = through_flux / local_ref
        cross_flux_ref_norm = cross_flux / local_ref
        local_output_flux_ref_norm = local_output_flux / local_ref

    if not np.isfinite(total_ref) or abs(total_ref) < 1e-12:
        total_output_flux_ref_norm = np.nan
    else:
        total_output_flux_ref_norm = total_output_flux / total_ref

    ideal_cross = ideal_cross_power(
        length_um=coupler_length_um,
        l_full_um=spec.l_full_um,
    )
    ideal_through = 1.0 - ideal_cross

    return {
        "gap_um": spec.gap_um,
        "coupler_length_um": coupler_length_um,
        "through_flux": through_flux,
        "cross_flux": cross_flux,
        "local_output_flux": local_output_flux,
        "total_output_flux": total_output_flux,
        "local_reference_flux": reference["local_reference_flux"],
        "total_reference_flux": reference["total_reference_flux"],

        # Bounded local output fractions. These are what we plot for now.
        "through_flux_fraction": through_power,
        "cross_flux_fraction": cross_power,
        "local_fraction_sum": through_power + cross_power,

        # Diagnostic raw-flux normalization.
        "through_flux_ref_norm": through_flux_ref_norm,
        "cross_flux_ref_norm": cross_flux_ref_norm,
        "local_output_flux_ref_norm": local_output_flux_ref_norm,
        "total_output_flux_ref_norm": total_output_flux_ref_norm,

        # This is no longer a physical loss estimate. It is left as NaN to avoid
        # pretending flux diagnostics are guided-mode S-parameters.
        "excess_loss": np.nan,

        "ideal_through_power": ideal_through,
        "ideal_cross_power": ideal_cross,
        "resolution": settings.resolution,
        "runtime": settings.runtime,
        "max_coupler_length_um": settings.max_coupler_length_um,
    }

def run_length_sweep(
    lengths_um: np.ndarray,
    spec: DirectionalCouplerFDTDSpec,
    settings: FDTDSettings,
) -> pd.DataFrame:
    """
    Run finite-coupler FDTD simulations over a list of lengths.

    This diagnostic version does not use straight-reference normalization.
    It reports local through/cross flux fractions only.
    """
    rows: list[dict[str, float]] = []

    print()
    print("Running finite-coupler flux-fraction diagnostic")
    print("------------------------------------------------")
    print(
        "No straight-reference normalization is used here. "
        "Reported through/cross values are local flux fractions, "
        "not modal S-parameters."
    )

    for length_um in lengths_um:
        print()
        print(f"Running finite coupler FDTD: L = {length_um:.3f} um")
        print("--------------------------------------------------")

        row = run_single_length_fdtd(
            coupler_length_um=float(length_um),
            spec=spec,
            settings=settings,
            reference={
                "local_reference_flux": np.nan,
                "total_reference_flux": np.nan,
            },
        )
        rows.append(row)

        print(
            f"through_fraction={row['through_flux_fraction']:.4f}, "
            f"cross_fraction={row['cross_flux_fraction']:.4f}, "
            f"total_output_flux={row['total_output_flux']:.4e}, "
            f"ideal_cross={row['ideal_cross_power']:.4f}"
        )

    return pd.DataFrame(rows)

def plot_length_sweep(df: pd.DataFrame, output_path: Path) -> None:
    """Plot FDTD through/cross power versus ideal supermode prediction."""
    required_columns = {
        "coupler_length_um",
        "through_flux_fraction",
        "cross_flux_fraction",
        "ideal_through_power",
        "ideal_cross_power",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required plot columns: {missing}")

    df = df.sort_values("coupler_length_um")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.plot(
        df["coupler_length_um"],
        df["cross_flux_fraction"],
        marker="o",
        label="FDTD cross fraction",
    )
    ax.plot(
        df["coupler_length_um"],
        df["through_flux_fraction"],
        marker="o",
        label="FDTD through fraction",
    )
    ax.plot(
        df["coupler_length_um"],
        df["ideal_cross_power"],
        linestyle="--",
        label="ideal cross",
    )
    ax.plot(
        df["coupler_length_um"],
        df["ideal_through_power"],
        linestyle="--",
        label="ideal through",
    )

    ax.set_xlabel("Coupler length (um)")
    ax.set_ylabel("Local output fraction")
    ax.set_title("Finite Directional Coupler: Flux Fractions vs Supermode Prediction")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-0.05, 1.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if not output_path.exists():
        raise RuntimeError(f"Expected plot was not saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite directional coupler FDTD length sweep."
    )

    parser.add_argument("--gap", type=float, default=0.20)
    parser.add_argument("--l-full", type=float, default=38.2)
    parser.add_argument("--resolution", type=int, default=30)
    parser.add_argument("--runtime", type=float, default=250.0)

    parser.add_argument(
        "--length-max",
        type=float,
        default=80.0,
        help="Maximum coupler length in microns for FDTD sweep.",
    )

    parser.add_argument(
        "--num-lengths",
        type=int,
        default=17,
        help="Number of coupler lengths to simulate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec = DirectionalCouplerFDTDSpec(
        gap_um=args.gap,
        l_full_um=args.l_full,
    )

    settings = FDTDSettings(
        resolution=args.resolution,
        runtime=args.runtime,
        max_coupler_length_um=args.length_max,
    )

    lengths_um = np.linspace(
        0.0,
        args.length_max,
        args.num_lengths,
    )

    df = run_length_sweep(
        lengths_um=lengths_um,
        spec=spec,
        settings=settings,
    )

    data_dir = Path("data/sweeps")
    fig_dir = Path("results/figures")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "directional_coupler_fdtd_length_sweep.csv"
    fig_path = fig_dir / "directional_coupler_fdtd_length_sweep.png"

    df.to_csv(csv_path, index=False)
    plot_length_sweep(df=df, output_path=fig_path)

    print()
    print("Finite directional coupler FDTD length sweep")
    print("--------------------------------------------")
    print(df.to_string(index=False))
    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {fig_path}")

    print()
    print("WARNING")
    print("-------")
    print(
        "This FDTD script reports crude flux fractions, not modal S-parameters. "
        "Use it only as a diagnostic learning model. True coupler metrics require "
        "mode-decomposition monitors / S-parameter extraction."
    )    


if __name__ == "__main__":
    main()