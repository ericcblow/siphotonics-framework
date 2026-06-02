"""
Directional coupler modal S-parameter diagnostic extraction.

Purpose:
- Move beyond crude flux monitors.
- Use Meep mode decomposition to estimate modal through/cross/reflection powers.
- Support two learning geometries:
    1. finite-segment coupler: fixed cell, finite top-guide segment
    2. full-parallel beating test: both guides exist along the whole variable-length cell

Important limitations:
- This is a 2D effective-index learning model, not a production 3D SOI solver.
- Raw modal powers are reported for debugging.
- Incident-normalized values are the most useful diagnostic columns, but they are
  still not guaranteed final process-ready S-parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectionalCouplerSParamSpec:
    """2D effective-index directional-coupler specification."""

    width_um: float = 0.5
    gap_um: float = 0.20
    wavelength_um: float = 1.55
    core_index_2d: float = 2.85
    cladding_index: float = 1.444
    coupler_length_um: float = 19.1


@dataclass(frozen=True)
class SParamSettings:
    """Numerical settings for modal diagnostic extraction."""

    resolution: int = 20
    pml_um: float = 1.0
    input_straight_um: float = 4.0
    output_straight_um: float = 4.0
    y_padding_um: float = 2.0
    runtime: float = 150.0

    source_width_um: float = 0.6
    monitor_width_um: float = 0.6

    max_coupler_length_um: float = 40.0
    use_full_parallel_guides: bool = False

    eig_band: int = 1
    eig_parity: int = mp.ODD_Z

    decay_check_interval: float = 50.0
    decay_by: float = 1e-6


def ideal_cross_power(length_um: float, l_full_um: float = 38.2) -> float:
    """Ideal supermode cross power for comparison only."""
    if length_um < 0:
        raise ValueError("length_um must be nonnegative.")
    if l_full_um <= 0:
        raise ValueError("l_full_um must be positive.")
    return float(np.sin(np.pi * length_um / (2.0 * l_full_um)) ** 2)

def ideal_cross_power_with_lfull(length_um: float, l_full_um: float) -> float:
    """Ideal cross power for a specified complete-transfer length."""
    if length_um < 0:
        raise ValueError("length_um must be nonnegative.")
    if l_full_um <= 0:
        raise ValueError("l_full_um must be positive.")

    return float(np.sin(np.pi * length_um / (2.0 * l_full_um)) ** 2)


def add_empirical_full_parallel_fit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an empirical ideal curve for full-parallel FDTD beating.

    The fitted complete-transfer length is estimated as the length where
    cross_incident_norm is maximum.
    """
    required_columns = {"coupler_length_um", "cross_incident_norm"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required fit columns: {missing}")

    df = df.copy()

    valid = df[
        np.isfinite(df["cross_incident_norm"])
        & np.isfinite(df["coupler_length_um"])
    ]

    if valid.empty:
        df["l_full_fit_um"] = np.nan
        df["fit_cross_power"] = np.nan
        df["fit_through_power"] = np.nan
        return df

    peak_idx = valid["cross_incident_norm"].idxmax()
    l_full_fit_um = float(valid.loc[peak_idx, "coupler_length_um"])

    df["l_full_fit_um"] = l_full_fit_um
    df["fit_cross_power"] = df["coupler_length_um"].apply(
        lambda length_um: ideal_cross_power_with_lfull(
            length_um=float(length_um),
            l_full_um=l_full_fit_um,
        )
    )
    df["fit_through_power"] = 1.0 - df["fit_cross_power"]

    return df

def make_geometry(
    spec: DirectionalCouplerSParamSpec,
    settings: SParamSettings,
) -> tuple[list[mp.GeometricObject], mp.Vector3, dict[str, float]]:
    """
    Build the 2D geometry.

    finite-segment mode:
        fixed cell length = input + settings.max_coupler_length_um + output
        bottom guide length = full fixed device length
        top guide length = spec.coupler_length_um

    full-parallel mode:
        variable cell length = input + spec.coupler_length_um + output
        bottom guide length = full variable device length
        top guide length = full variable device length
    """
    if spec.coupler_length_um < 0:
        raise ValueError("coupler_length_um must be nonnegative.")
    if settings.max_coupler_length_um <= 0:
        raise ValueError("max_coupler_length_um must be positive.")

    if settings.use_full_parallel_guides:
        active_length_um = spec.coupler_length_um
    else:
        if spec.coupler_length_um > settings.max_coupler_length_um:
            raise ValueError(
                "coupler_length_um cannot exceed settings.max_coupler_length_um "
                "in finite-segment mode. "
                f"Got L={spec.coupler_length_um}, Lmax={settings.max_coupler_length_um}."
            )
        active_length_um = settings.max_coupler_length_um

    total_length_um = (
        settings.input_straight_um
        + active_length_um
        + settings.output_straight_um
    )

    sx = total_length_um + 2.0 * settings.pml_um
    sy = 2.0 * settings.y_padding_um + 2.0 * spec.width_um + spec.gap_um

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

    if settings.use_full_parallel_guides:
        top_length_um = total_length_um
    elif spec.coupler_length_um > 0:
        top_length_um = spec.coupler_length_um
    else:
        top_length_um = 0.0

    if top_length_um > 0:
        geometry.append(
            mp.Block(
                size=mp.Vector3(top_length_um, spec.width_um, mp.inf),
                center=mp.Vector3(0, y_top, 0),
                material=core,
            )
        )

    cell_size = mp.Vector3(sx, sy, 0)

    positions = {
        "total_length_um": total_length_um,
        "active_length_um": active_length_um,
        "x_source": -total_length_um / 2.0 + 1.0,
        "x_reflection_monitor": -total_length_um / 2.0 + 1.8,
        "x_output_monitor": +total_length_um / 2.0 - 1.0,
        "y_bottom": y_bottom,
        "y_top": y_top,
    }

    return geometry, cell_size, positions


def make_eigenmode_source(
    frequency: float,
    fwidth: float,
    positions: dict[str, float],
    settings: SParamSettings,
) -> mp.EigenModeSource:
    """Create an eigenmode source in the bottom input guide."""
    return mp.EigenModeSource(
        src=mp.GaussianSource(frequency=frequency, fwidth=fwidth),
        center=mp.Vector3(positions["x_source"], positions["y_bottom"], 0),
        size=mp.Vector3(0, settings.source_width_um, 0),
        eig_band=settings.eig_band,
        eig_parity=settings.eig_parity,
        eig_match_freq=True,
    )


def _coefficients_from_monitor(
    sim: mp.Simulation,
    monitor,
    band: int,
    eig_parity,
) -> dict[str, float]:
    """Extract two directional mode coefficients from a mode monitor."""
    coeffs = sim.get_eigenmode_coefficients(
        monitor,
        [band],
        eig_parity=eig_parity,
    )

    alpha = np.asarray(coeffs.alpha)
    if alpha.size < 2:
        raise RuntimeError(
            f"Expected at least two directional coefficients, got shape {alpha.shape}"
        )

    flat = np.ravel(alpha)
    a0 = complex(flat[0])
    a1 = complex(flat[1])

    return {
        "alpha_0_real": a0.real,
        "alpha_0_imag": a0.imag,
        "alpha_1_real": a1.real,
        "alpha_1_imag": a1.imag,
        "alpha_0_power": float(abs(a0) ** 2),
        "alpha_1_power": float(abs(a1) ** 2),
    }


def _build_simulation(
    spec: DirectionalCouplerSParamSpec,
    settings: SParamSettings,
    include_top_waveguide: bool = True,
):
    """Build simulation, source, and common geometry data."""
    geometry, cell_size, positions = make_geometry(spec=spec, settings=settings)
    if not include_top_waveguide:
        geometry = geometry[:1]

    frequency = 1.0 / spec.wavelength_um
    fwidth = 0.10 * frequency

    source = make_eigenmode_source(
        frequency=frequency,
        fwidth=fwidth,
        positions=positions,
        settings=settings,
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

    return sim, frequency, positions


def run_until_output_decayed(
    sim: mp.Simulation,
    positions: dict[str, float],
    settings: SParamSettings,
) -> None:
    """Run until the source is off and output-port fields have decayed."""
    output_point = mp.Vector3(
        positions["x_output_monitor"],
        positions["y_bottom"],
        0,
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            settings.decay_check_interval,
            mp.Ez,
            output_point,
            settings.decay_by,
        )
    )


def _mode_power_summary(
    through_coeffs: dict[str, float],
    cross_coeffs: dict[str, float],
    reflection_coeffs: dict[str, float],
) -> dict[str, float]:
    """Compute raw and incident-normalized diagnostic powers."""
    incident_mode_power = max(
        reflection_coeffs["alpha_0_power"],
        reflection_coeffs["alpha_1_power"],
    )
    through_raw_power = max(
        through_coeffs["alpha_0_power"],
        through_coeffs["alpha_1_power"],
    )
    cross_raw_power = max(
        cross_coeffs["alpha_0_power"],
        cross_coeffs["alpha_1_power"],
    )
    reflection_raw_power = min(
        reflection_coeffs["alpha_0_power"],
        reflection_coeffs["alpha_1_power"],
    )

    if incident_mode_power <= 0:
        through_incident_norm = np.nan
        cross_incident_norm = np.nan
        reflection_incident_norm = np.nan
    else:
        through_incident_norm = through_raw_power / incident_mode_power
        cross_incident_norm = cross_raw_power / incident_mode_power
        reflection_incident_norm = reflection_raw_power / incident_mode_power

    through_mode_power = through_raw_power
    cross_mode_power = cross_raw_power
    reflection_mode_power = reflection_raw_power
    guided_power_sum = through_mode_power + cross_mode_power + reflection_mode_power

    return {
        "through_mode_power": through_mode_power,
        "cross_mode_power": cross_mode_power,
        "reflection_mode_power": reflection_mode_power,
        "guided_power_sum": guided_power_sum,
        "residual_power_estimate": 1.0 - guided_power_sum,
        "incident_mode_power": incident_mode_power,
        "through_incident_norm": through_incident_norm,
        "cross_incident_norm": cross_incident_norm,
        "reflection_incident_norm": reflection_incident_norm,
        "through_raw_power": through_raw_power,
        "cross_raw_power": cross_raw_power,
        "reflection_raw_power": reflection_raw_power,
    }


def run_single_sparameter_case(
    spec: DirectionalCouplerSParamSpec,
    settings: SParamSettings,
) -> dict[str, float]:
    """Run one 2D coupler simulation and extract modal diagnostics."""
    sim, frequency, positions = _build_simulation(
        spec=spec,
        settings=settings,
        include_top_waveguide=True,
    )

    through_monitor = sim.add_mode_monitor(
        frequency,
        0,
        1,
        mp.ModeRegion(
            center=mp.Vector3(positions["x_output_monitor"], positions["y_bottom"], 0),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )
    cross_monitor = sim.add_mode_monitor(
        frequency,
        0,
        1,
        mp.ModeRegion(
            center=mp.Vector3(positions["x_output_monitor"], positions["y_top"], 0),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )
    reflection_monitor = sim.add_mode_monitor(
        frequency,
        0,
        1,
        mp.ModeRegion(
            center=mp.Vector3(
                positions["x_reflection_monitor"],
                positions["y_bottom"],
                0,
            ),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )

    run_until_output_decayed(sim=sim, positions=positions, settings=settings)

    through_coeffs = _coefficients_from_monitor(
        sim=sim,
        monitor=through_monitor,
        band=settings.eig_band,
        eig_parity=settings.eig_parity,
    )
    cross_coeffs = _coefficients_from_monitor(
        sim=sim,
        monitor=cross_monitor,
        band=settings.eig_band,
        eig_parity=settings.eig_parity,
    )
    reflection_coeffs = _coefficients_from_monitor(
        sim=sim,
        monitor=reflection_monitor,
        band=settings.eig_band,
        eig_parity=settings.eig_parity,
    )

    powers = _mode_power_summary(
        through_coeffs=through_coeffs,
        cross_coeffs=cross_coeffs,
        reflection_coeffs=reflection_coeffs,
    )

    return {
        "gap_um": spec.gap_um,
        "coupler_length_um": spec.coupler_length_um,
        "wavelength_um": spec.wavelength_um,
        **powers,
        "through_alpha0_power": through_coeffs["alpha_0_power"],
        "through_alpha1_power": through_coeffs["alpha_1_power"],
        "cross_alpha0_power": cross_coeffs["alpha_0_power"],
        "cross_alpha1_power": cross_coeffs["alpha_1_power"],
        "reflection_alpha0_power": reflection_coeffs["alpha_0_power"],
        "reflection_alpha1_power": reflection_coeffs["alpha_1_power"],
        "ideal_cross_power": ideal_cross_power(spec.coupler_length_um),
        "resolution": settings.resolution,
        "runtime": settings.runtime,
        "full_parallel": settings.use_full_parallel_guides,
    }


def run_straight_port_test(
    spec: DirectionalCouplerSParamSpec,
    settings: SParamSettings,
) -> dict[str, float]:
    """Run a straight bottom-waveguide port sanity test."""
    straight_spec = DirectionalCouplerSParamSpec(
        width_um=spec.width_um,
        gap_um=spec.gap_um,
        wavelength_um=spec.wavelength_um,
        core_index_2d=spec.core_index_2d,
        cladding_index=spec.cladding_index,
        coupler_length_um=settings.max_coupler_length_um,
    )

    straight_settings = SParamSettings(
        resolution=settings.resolution,
        pml_um=settings.pml_um,
        input_straight_um=settings.input_straight_um,
        output_straight_um=settings.output_straight_um,
        y_padding_um=settings.y_padding_um,
        runtime=settings.runtime,
        source_width_um=settings.source_width_um,
        monitor_width_um=settings.monitor_width_um,
        max_coupler_length_um=settings.max_coupler_length_um,
        use_full_parallel_guides=True,
        eig_band=settings.eig_band,
        eig_parity=settings.eig_parity,
        decay_check_interval=settings.decay_check_interval,
        decay_by=settings.decay_by,
    )

    sim, frequency, positions = _build_simulation(
        spec=straight_spec,
        settings=straight_settings,
        include_top_waveguide=False,
    )

    output_monitor = sim.add_mode_monitor(
        frequency,
        0,
        1,
        mp.ModeRegion(
            center=mp.Vector3(positions["x_output_monitor"], positions["y_bottom"], 0),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )
    reflection_monitor = sim.add_mode_monitor(
        frequency,
        0,
        1,
        mp.ModeRegion(
            center=mp.Vector3(
                positions["x_reflection_monitor"],
                positions["y_bottom"],
                0,
            ),
            size=mp.Vector3(0, settings.monitor_width_um, 0),
        ),
    )

    run_until_output_decayed(
        sim=sim,
        positions=positions,
        settings=straight_settings,
    )

    output_coeffs = _coefficients_from_monitor(
        sim=sim,
        monitor=output_monitor,
        band=settings.eig_band,
        eig_parity=settings.eig_parity,
    )
    reflection_coeffs = _coefficients_from_monitor(
        sim=sim,
        monitor=reflection_monitor,
        band=settings.eig_band,
        eig_parity=settings.eig_parity,
    )

    output_forward_power = max(
        output_coeffs["alpha_0_power"],
        output_coeffs["alpha_1_power"],
    )
    output_backward_power = min(
        output_coeffs["alpha_0_power"],
        output_coeffs["alpha_1_power"],
    )
    reflection_incident_power = max(
        reflection_coeffs["alpha_0_power"],
        reflection_coeffs["alpha_1_power"],
    )
    reflection_backward_power = min(
        reflection_coeffs["alpha_0_power"],
        reflection_coeffs["alpha_1_power"],
    )

    if reflection_incident_power <= 0:
        straight_transmission_norm = np.nan
        straight_reflection_norm = np.nan
    else:
        straight_transmission_norm = output_forward_power / reflection_incident_power
        straight_reflection_norm = reflection_backward_power / reflection_incident_power

    return {
        "wavelength_um": spec.wavelength_um,
        "length_um": settings.max_coupler_length_um,
        "output_forward_power": output_forward_power,
        "output_backward_power": output_backward_power,
        "reflection_incident_power": reflection_incident_power,
        "reflection_backward_power": reflection_backward_power,
        "output_alpha0_power": output_coeffs["alpha_0_power"],
        "output_alpha1_power": output_coeffs["alpha_1_power"],
        "reflection_alpha0_power": reflection_coeffs["alpha_0_power"],
        "reflection_alpha1_power": reflection_coeffs["alpha_1_power"],
        "straight_transmission_norm": straight_transmission_norm,
        "straight_reflection_norm": straight_reflection_norm,
        "resolution": settings.resolution,
        "runtime": settings.runtime,
    }


def get_straight_output_reference_power(
    spec: DirectionalCouplerSParamSpec,
    settings: SParamSettings,
) -> float:
    """Return raw straight-waveguide output alpha power for diagnostics."""
    row = run_straight_port_test(spec=spec, settings=settings)
    reference_power = float(row["output_forward_power"])
    if reference_power <= 0:
        raise RuntimeError(f"Invalid straight output reference power: {reference_power}")
    return reference_power


def run_length_sweep(
    lengths_um: np.ndarray,
    base_spec: DirectionalCouplerSParamSpec,
    settings: SParamSettings,
) -> pd.DataFrame:
    """Run modal diagnostic extraction over coupler length."""
    rows: list[dict[str, float]] = []

    for length_um in lengths_um:
        print()
        print(f"Running modal diagnostic case: L = {length_um:.3f} um")
        print("-------------------------------------------------")

        spec = DirectionalCouplerSParamSpec(
            width_um=base_spec.width_um,
            gap_um=base_spec.gap_um,
            wavelength_um=base_spec.wavelength_um,
            core_index_2d=base_spec.core_index_2d,
            cladding_index=base_spec.cladding_index,
            coupler_length_um=float(length_um),
        )

        row = run_single_sparameter_case(spec=spec, settings=settings)
        rows.append(row)

        print(
            f"through_norm={row['through_incident_norm']:.4g}, "
            f"cross_norm={row['cross_incident_norm']:.4g}, "
            f"reflection_norm={row['reflection_incident_norm']:.4g}, "
            f"ideal_cross={row['ideal_cross_power']:.4g}"
        )

    return pd.DataFrame(rows)


def _plot_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Prefer incident-normalized diagnostics for plots."""
    preferred = [
        "through_incident_norm",
        "cross_incident_norm",
        "reflection_incident_norm",
        "ideal_cross_power",
    ]
    if all(column in df.columns for column in preferred):
        return tuple(preferred)  # type: ignore[return-value]
    return (
        "through_mode_power",
        "cross_mode_power",
        "reflection_mode_power",
        "ideal_cross_power",
    )


def plot_length_sweep(df: pd.DataFrame, output_path: Path) -> None:
    """Plot modal diagnostic powers versus coupler length."""
    through_col, cross_col, reflection_col, ideal_col = _plot_columns(df)

    required_columns = {
        "coupler_length_um",
        through_col,
        cross_col,
        reflection_col,
        ideal_col,
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")

    df = df.sort_values("coupler_length_um")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(
        df["coupler_length_um"],
        df[through_col],
        marker="o",
        label="through diagnostic",
    )
    ax.plot(
        df["coupler_length_um"],
        df[cross_col],
        marker="o",
        label="cross diagnostic",
    )
    ax.plot(
        df["coupler_length_um"],
        df[reflection_col],
        marker="o",
        label="reflection diagnostic",
    )
    ax.plot(
        df["coupler_length_um"],
        df[ideal_col],
        linestyle="--",
        label="old ideal cross reference",
    )

    if {"fit_cross_power", "fit_through_power"}.issubset(df.columns):
        ax.plot(
            df["coupler_length_um"],
            df["fit_cross_power"],
            linestyle="--",
            label="fit ideal cross",
        )
        ax.plot(
            df["coupler_length_um"],
            df["fit_through_power"],
            linestyle="--",
            label="fit ideal through",
        )

    ax.set_xlabel("Coupler length (um)")
    ax.set_ylabel("Incident-normalized diagnostic power")
    ax.set_title("Directional Coupler Modal Diagnostic Length Sweep")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-0.05, 1.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_case(row: dict[str, float], output_path: Path) -> None:
    """Bar plot of extracted modal diagnostic powers for one coupler."""
    labels = ["through", "cross", "reflection", "ideal cross"]
    values = [
        row.get("through_incident_norm", row["through_mode_power"]),
        row.get("cross_incident_norm", row["cross_mode_power"]),
        row.get("reflection_incident_norm", row["reflection_mode_power"]),
        row["ideal_cross_power"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("Diagnostic power")
    ax.set_title("Directional Coupler Modal Diagnostic")
    ax.grid(True, axis="y")
    ax.set_ylim(-0.05, 1.2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directional coupler modal diagnostic."
    )

    parser.add_argument("--gap", type=float, default=0.20)
    parser.add_argument("--length", type=float, default=19.1)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--runtime", type=float, default=150.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a coupler-length modal diagnostic sweep.",
    )
    parser.add_argument(
        "--length-max",
        type=float,
        default=40.0,
        help="Maximum coupler length in microns for sweep or straight test.",
    )
    parser.add_argument(
        "--num-lengths",
        type=int,
        default=9,
        help="Number of coupler lengths in sweep.",
    )
    parser.add_argument(
        "--straight-test",
        action="store_true",
        help="Run a straight-waveguide port sanity test.",
    )
    parser.add_argument(
        "--full-parallel",
        action="store_true",
        help=(
            "Use two full-length parallel waveguides. In sweep mode this makes "
            "the simulation cell length vary with the swept length."
        ),
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include L=0 in --full-parallel sweeps. Default skips zero.",
    )

    return parser.parse_args()


def make_sweep_lengths(args: argparse.Namespace) -> np.ndarray:
    """Create sweep lengths, skipping zero for full-parallel mode by default."""
    if args.num_lengths <= 0:
        raise ValueError("--num-lengths must be positive.")
    if args.length_max <= 0:
        raise ValueError("--length-max must be positive.")

    if args.full_parallel and not args.include_zero:
        return np.linspace(
            args.length_max / args.num_lengths,
            args.length_max,
            args.num_lengths,
        )

    return np.linspace(0.0, args.length_max, args.num_lengths)


def main() -> None:
    args = parse_args()

    spec = DirectionalCouplerSParamSpec(
        gap_um=args.gap,
        coupler_length_um=args.length,
    )

    if args.sweep or args.straight_test:
        max_coupler_length_um = args.length_max
    else:
        max_coupler_length_um = args.length

    settings = SParamSettings(
        resolution=args.resolution,
        runtime=args.runtime,
        max_coupler_length_um=max_coupler_length_um,
        use_full_parallel_guides=args.full_parallel,
    )

    data_dir = Path("data/sparameters")
    fig_dir = Path("results/figures")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.straight_test:
        row = run_straight_port_test(spec=spec, settings=settings)
        csv_path = data_dir / "directional_coupler_straight_port_test.csv"
        pd.DataFrame([row]).to_csv(csv_path, index=False)

        print()
        print("Straight waveguide port sanity test")
        print("-----------------------------------")
        print(pd.DataFrame([row]).to_string(index=False))
        print()
        print(f"Saved CSV: {csv_path}")
        print()
        print("Expected diagnostic behavior:")
        print("  straight_transmission_norm should be near order unity")
        print("  straight_reflection_norm should be small")
        return

    if args.sweep:
        lengths_um = make_sweep_lengths(args)
        df = run_length_sweep(
            lengths_um=lengths_um,
            base_spec=spec,
            settings=settings,
        )
        if args.full_parallel:
            df = add_empirical_full_parallel_fit(df)

        csv_path = data_dir / "directional_coupler_sparameter_length_sweep.csv"
        fig_path = fig_dir / "directional_coupler_sparameter_length_sweep.png"

        df.to_csv(csv_path, index=False)
        plot_length_sweep(df=df, output_path=fig_path)

        print()
        print("Directional coupler modal diagnostic length sweep")
        print("-------------------------------------------------")
        print(df.to_string(index=False))
        print()
        print(f"Saved CSV:  {csv_path}")
        print(f"Saved plot: {fig_path}")
        print()
        print("Interpretation warning:")
        print(
            "This is a modal-coefficient diagnostic sweep in a 2D effective-index "
            "model. Use incident-normalized diagnostics for plots; raw values are "
            "not final normalized S-parameters."
        )
        return

    row = run_single_sparameter_case(spec=spec, settings=settings)

    csv_path = data_dir / "directional_coupler_sparameter_single_case.csv"
    fig_path = fig_dir / "directional_coupler_sparameter_single_case.png"

    pd.DataFrame([row]).to_csv(csv_path, index=False)
    plot_single_case(row=row, output_path=fig_path)

    print()
    print("Directional coupler modal diagnostic")
    print("------------------------------------")
    print(pd.DataFrame([row]).to_string(index=False))
    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {fig_path}")
    print()
    print("Interpretation warning:")
    print(
        "This is a modal-coefficient diagnostic. Raw values are not final "
        "normalized S-parameters; use the straight-port test to debug ports."
    )


if __name__ == "__main__":
    main()
