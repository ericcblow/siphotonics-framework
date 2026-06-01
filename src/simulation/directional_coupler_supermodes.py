"""
Directional coupler supermode sweep.

This module bridges abstract ring coupling coefficients to physical
directional-coupler geometry.

It supports two solver modes:

1. mock:
   - Uses a temporary exponential model for supermode splitting.
   - Useful for testing the data pipeline quickly.

2. mpb:
   - Uses Meep/MPB to solve the two-waveguide eigenmode problem.
   - Extracts even and odd TE-like supermodes.
   - Computes delta_neff and coupling lengths versus gap.

Coordinate convention:

    x = propagation direction
    y = horizontal direction across waveguide width and coupler gap
    z = vertical direction through silicon thickness

For this convention, the TE-like strip-waveguide mode is expected to have
dominant Ey field.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import meep as mp
    from meep import mpb

    HAS_MEEP = True
except ImportError:
    HAS_MEEP = False


@dataclass(frozen=True)
class DirectionalCouplerSpec:
    """Basic SOI directional-coupler geometry and material parameters."""

    width_um: float = 0.5
    thickness_um: float = 0.22
    wavelength_um: float = 1.55
    si_n: float = 3.476
    sio2_n: float = 1.444


@dataclass(frozen=True)
class MpbCouplerSolveSettings:
    """Numerical settings for MPB directional-coupler supermode extraction."""

    resolution: int = 50
    padding_y_um: float = 2.0
    padding_z_um: float = 2.0
    num_bands: int = 6
    k_min_factor: float = 1.1
    k_max_factor: float = 3.6
    k_guess_neff: float = 2.45
    suppress_output: bool = True


def coupling_lengths_from_indices(
    wavelength_um: float,
    n_even: float,
    n_odd: float,
) -> dict[str, float]:
    """
    Compute directional-coupler supermode splitting and coupling lengths.

    Definitions used here:

        delta_neff = n_even - n_odd

        L_full = wavelength / (2 * delta_neff)

        L_3dB = L_full / 2

    L_full is the ideal complete-transfer length.

    L_3dB is the ideal 50/50 splitting length.
    """
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")

    delta_neff = n_even - n_odd

    if delta_neff <= 0:
        raise ValueError(
            "Expected n_even > n_odd. "
            f"Got n_even={n_even}, n_odd={n_odd}, delta={delta_neff}."
        )

    l_full_um = wavelength_um / (2.0 * delta_neff)
    l_3db_um = l_full_um / 2.0

    return {
        "delta_neff": float(delta_neff),
        "L_full_um": float(l_full_um),
        "L_3dB_um": float(l_3db_um),
    }


def cross_coupled_power(
    length_um: float,
    l_full_um: float,
) -> float:
    """
    Ideal cross-coupled power for a symmetric lossless directional coupler.

        kappa^2 = sin^2(pi * L / (2 * L_full))
    """
    if length_um < 0:
        raise ValueError("length_um must be nonnegative.")

    if l_full_um <= 0:
        raise ValueError("l_full_um must be positive.")

    return float(np.sin(np.pi * length_um / (2.0 * l_full_um)) ** 2)


def length_for_target_kappa_power(
    l_full_um: float,
    kappa_power: float,
) -> float:
    """
    Return physical coupler length for target power coupling kappa^2.

        L = 2 L_full / pi * asin(sqrt(kappa^2))
    """
    if l_full_um <= 0:
        raise ValueError("l_full_um must be positive.")

    if not 0.0 <= kappa_power <= 1.0:
        raise ValueError("kappa_power must be between 0 and 1.")

    return float(
        2.0
        * l_full_um
        / np.pi
        * np.arcsin(np.sqrt(kappa_power))
    )


def solve_even_odd_supermodes_mock(
    gap_um: float,
    spec: DirectionalCouplerSpec,
) -> tuple[float, float, dict[str, float]]:
    """
    Temporary mock model for even/odd supermode indices.

    This is not a real EM solver. It validates the data pipeline before
    using MPB.

        delta_neff(g) = A * exp(-g / g0)
    """
    if gap_um <= 0:
        raise ValueError("gap_um must be positive.")

    n_single = 2.444
    a = 0.05
    g0_um = 0.15

    delta_neff = a * np.exp(-gap_um / g0_um)

    n_even = n_single + delta_neff / 2.0
    n_odd = n_single - delta_neff / 2.0

    diagnostics = {
        "even_ey_fraction": np.nan,
        "odd_ey_fraction": np.nan,
        "even_symmetry_score": np.nan,
        "odd_symmetry_score": np.nan,
    }

    return float(n_even), float(n_odd), diagnostics


def _suppress_stdout_stderr(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()

    return contextlib.redirect_stdout(io.StringIO())


def build_mpb_mode_solver(
    gap_um: float,
    spec: DirectionalCouplerSpec,
    settings: MpbCouplerSolveSettings,
):
    """
    Build an MPB mode solver for two parallel strip waveguides.

    Geometry:
        two silicon blocks in oxide background

    Propagation:
        along x

    Cross-section:
        y-z plane
    """
    if not HAS_MEEP:
        raise ImportError(
            "Meep/MPB is not available. Install meep or run with --solver mock."
        )

    if gap_um <= 0:
        raise ValueError("gap_um must be positive.")

    width = spec.width_um
    thickness = spec.thickness_um
    spacing = width + gap_um

    cell_y = 2.0 * settings.padding_y_um + 2.0 * width + gap_um
    cell_z = 2.0 * settings.padding_z_um + thickness

    silicon = mp.Medium(index=spec.si_n)
    oxide = mp.Medium(index=spec.sio2_n)

    geometry_lattice = mp.Lattice(size=mp.Vector3(0, cell_y, cell_z))

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, width, thickness),
            center=mp.Vector3(0, -spacing / 2.0, 0),
            material=silicon,
        ),
        mp.Block(
            size=mp.Vector3(mp.inf, width, thickness),
            center=mp.Vector3(0, +spacing / 2.0, 0),
            material=silicon,
        ),
    ]

    frequency = 1.0 / spec.wavelength_um

    k_guess = settings.k_guess_neff * frequency
    k_min = settings.k_min_factor * frequency
    k_max = settings.k_max_factor * frequency

    mode_solver = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        default_material=oxide,
        resolution=settings.resolution,
        num_bands=settings.num_bands,
        k_points=[mp.Vector3(k_guess, 0, 0)],
    )

    return mode_solver, frequency, k_guess, k_min, k_max


def find_mpb_neffs(
    mode_solver,
    frequency: float,
    k_guess: float,
    k_min: float,
    k_max: float,
    settings: MpbCouplerSolveSettings,
) -> list[float]:
    """
    Use MPB find_k to solve for propagation constants at target frequency.

    MPB returns k values. In Meep/MPB normalized units, effective index is:

        n_eff = k / frequency
    """
    if not HAS_MEEP:
        raise ImportError("Meep/MPB is not available.")

    with _suppress_stdout_stderr(settings.suppress_output):
        k_values = mode_solver.find_k(
            mp.NO_PARITY,
            frequency,
            1,
            settings.num_bands,
            mp.Vector3(1, 0, 0),
            1e-7,
            k_guess,
            k_min,
            k_max,
        )

    neffs = [float(k / frequency) for k in k_values]
    return neffs


def _as_component_array(efield, component_index: int) -> np.ndarray:
    """
    Convert MPB efield output to a 2D complex array for one field component.

    Expected MPB output often has shape like:

        (Nx, Ny, Nz, 3)

    For our cross-section, Nx may be 1. We squeeze singleton dimensions and
    keep the requested vector component.
    """
    arr = np.asarray(efield)

    if arr.ndim < 1:
        raise ValueError("Unexpected empty efield array.")

    if arr.shape[-1] < 3:
        raise ValueError(f"Expected last efield axis to contain vector components, got {arr.shape}.")

    comp = arr[..., component_index]
    comp = np.squeeze(comp)

    if comp.ndim != 2:
        raise ValueError(
            f"Expected squeezed field component to be 2D, got shape {comp.shape}."
        )

    return comp


def get_band_field_components(mode_solver, band: int, suppress_output: bool = True):
    """
    Return Ex, Ey, Ez component arrays for a solved MPB band.

    Band index follows MPB convention and starts from 1.
    """
    with _suppress_stdout_stderr(suppress_output):
        mode_solver.get_dfield(band)
        efield = mode_solver.get_efield(band, bloch_phase=True)

    ex = _as_component_array(efield, 0)
    ey = _as_component_array(efield, 1)
    ez = _as_component_array(efield, 2)

    return ex, ey, ez

def field_energy_fraction(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
) -> tuple[float, float, float]:
    """Return Ex, Ey, Ez energy fractions from field component arrays."""
    ex_energy = float(np.sum(np.abs(ex) ** 2))
    ey_energy = float(np.sum(np.abs(ey) ** 2))
    ez_energy = float(np.sum(np.abs(ez) ** 2))

    total = ex_energy + ey_energy + ez_energy
    if total <= 0:
        raise ValueError("Total field energy is zero.")

    return ex_energy / total, ey_energy / total, ez_energy / total


def symmetry_score_y(field: np.ndarray) -> float:
    """
    Compute a simple symmetry score about y = 0.

    Score convention:

        +1 -> even symmetry
        -1 -> odd symmetry

    We compare the dominant component field to its mirror image.

    Because MPB eigenvectors can have arbitrary complex phase, we first rotate
    the field by the phase of its largest-magnitude element. This makes the
    comparison more stable for mostly real mode profiles.
    """
    if field.ndim != 2:
        raise ValueError("field must be a 2D array.")

    idx = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    phase = np.angle(field[idx])
    f = field * np.exp(-1j * phase)

    # Axis 0 is assumed to be y after squeezing. If your plots show the mirror
    # axis is wrong, swap this to axis=1.
    f_mirror = np.flip(f, axis=0)

    numerator = np.real(np.vdot(f_mirror, f))
    denominator = float(np.sum(np.abs(f) ** 2))

    if denominator <= 0:
        raise ValueError("Cannot compute symmetry of zero field.")

    return float(numerator / denominator)


def classify_supermodes_from_bands(
    mode_solver,
    neffs: list[float],
    settings: MpbCouplerSolveSettings,
    spec: DirectionalCouplerSpec | None = None,
) -> tuple[float, float, dict[str, float]]:
    """
    Classify directional-coupler supermodes from effective indices only.

    First robust MPB version:
    - Do not call get_efield() here.
    - Select the two largest distinct guided n_eff values.
    - Treat the larger one as n_even and the smaller one as n_odd.

    This is a practical first pass. We will add field-profile validation
    separately after the index sweep is working.
    """
    if spec is None:
        n_clad = 1.444
        n_core = 3.476
    else:
        n_clad = spec.sio2_n
        n_core = spec.si_n

    # Keep finite, physically guided candidates.
    guided = [
        float(n)
        for n in neffs
        if np.isfinite(n) and n_clad < float(n) < n_core
    ]

    if len(guided) < 2:
        raise ValueError(
            "Could not find at least two guided supermode candidates. "
            f"Raw neffs: {neffs}"
        )

    # Sort descending and remove near-duplicates.
    guided_sorted = sorted(guided, reverse=True)

    distinct: list[float] = []
    tol = 1e-5

    for n in guided_sorted:
        if not distinct or abs(n - distinct[-1]) > tol:
            distinct.append(n)

    if len(distinct) < 2:
        raise ValueError(
            "Could not find two distinct guided supermode indices. "
            f"Guided neffs: {guided_sorted}. "
            "This usually means the solver returned duplicate roots or only one usable band."
        )

    n_even = distinct[0]
    n_odd = distinct[1]

    diagnostics = {
        "even_band": np.nan,
        "odd_band": np.nan,
        "even_ey_fraction": np.nan,
        "odd_ey_fraction": np.nan,
        "even_symmetry_score": np.nan,
        "odd_symmetry_score": np.nan,
        "even_ex_fraction": np.nan,
        "odd_ex_fraction": np.nan,
        "even_ez_fraction": np.nan,
        "odd_ez_fraction": np.nan,
    }

    return n_even, n_odd, diagnostics

def solve_even_odd_supermodes_mpb(
    gap_um: float,
    spec: DirectionalCouplerSpec,
    settings: MpbCouplerSolveSettings,
) -> tuple[float, float, dict[str, float]]:
    """
    Solve even/odd TE-like supermode indices using MPB.
    """
    mode_solver, frequency, k_guess, k_min, k_max = build_mpb_mode_solver(
        gap_um=gap_um,
        spec=spec,
        settings=settings,
    )

    neffs = find_mpb_neffs(
        mode_solver=mode_solver,
        frequency=frequency,
        k_guess=k_guess,
        k_min=k_min,
        k_max=k_max,
        settings=settings,
    )

    n_even, n_odd, diagnostics = classify_supermodes_from_bands(
        mode_solver=mode_solver,
        neffs=neffs,
        settings=settings,
        spec=spec,
    )

    return n_even, n_odd, diagnostics

def run_resolution_convergence_sweep(
    gaps_um: np.ndarray,
    resolutions: list[int],
    spec: DirectionalCouplerSpec,
    base_settings: MpbCouplerSolveSettings,
) -> pd.DataFrame:
    """
    Run MPB directional-coupler sweeps at several resolutions.

    This is used to quantify numerical convergence of:
        - n_even
        - n_odd
        - delta_neff
        - L_full_um

    MPB runs can be slow, so use this intentionally rather than inside pytest.
    """
    rows: list[dict[str, float]] = []

    for resolution in resolutions:
        settings = MpbCouplerSolveSettings(
            resolution=resolution,
            padding_y_um=base_settings.padding_y_um,
            padding_z_um=base_settings.padding_z_um,
            num_bands=base_settings.num_bands,
            k_min_factor=base_settings.k_min_factor,
            k_max_factor=base_settings.k_max_factor,
            k_guess_neff=base_settings.k_guess_neff,
            suppress_output=base_settings.suppress_output,
        )

        print()
        print(f"Running directional-coupler MPB convergence sweep at resolution={resolution}")
        print("--------------------------------------------------------------------------")

        df = run_gap_sweep(
            gaps_um=gaps_um,
            spec=spec,
            solver="mpb",
            settings=settings,
        )

        df["resolution"] = resolution

        rows.extend(df.to_dict(orient="records"))

    return pd.DataFrame(rows)


def plot_resolution_convergence(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot directional-coupler convergence versus MPB resolution.

    Shows:
        - delta_neff versus gap
        - L_full versus gap
    """
    required_columns = {
        "gap_um",
        "resolution",
        "delta_neff",
        "L_full_um",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required convergence columns: {missing}")

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    for resolution, group in df.groupby("resolution"):
        group = group.sort_values("gap_um")

        axes[0].plot(
            group["gap_um"],
            group["delta_neff"],
            marker="o",
            label=f"res={resolution}",
        )

        axes[1].plot(
            group["gap_um"],
            group["L_full_um"],
            marker="o",
            label=f"res={resolution}",
        )

    axes[0].set_ylabel(r"$\Delta n_\mathrm{eff}$")
    axes[0].set_title("Directional Coupler Resolution Convergence")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Gap (um)")
    axes[1].set_ylabel(r"$L_\mathrm{full}$ (um)")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if not output_path.exists():
        raise RuntimeError(f"Expected convergence plot was not saved: {output_path}")

def summarize_resolution_difference(
    df: pd.DataFrame,
    low_resolution: int,
    high_resolution: int,
) -> pd.DataFrame:
    """
    Compare low-resolution and high-resolution MPB results gap by gap.

    Percent difference is computed relative to the high-resolution result.
    """
    low = (
        df[df["resolution"] == low_resolution]
        .set_index("gap_um")
        .sort_index()
    )
    high = (
        df[df["resolution"] == high_resolution]
        .set_index("gap_um")
        .sort_index()
    )

    common_gaps = low.index.intersection(high.index)

    rows: list[dict[str, float]] = []

    for gap_um in common_gaps:
        low_delta = float(low.loc[gap_um, "delta_neff"])
        high_delta = float(high.loc[gap_um, "delta_neff"])

        low_lfull = float(low.loc[gap_um, "L_full_um"])
        high_lfull = float(high.loc[gap_um, "L_full_um"])

        rows.append(
            {
                "gap_um": float(gap_um),
                f"delta_neff_res_{low_resolution}": low_delta,
                f"delta_neff_res_{high_resolution}": high_delta,
                "delta_neff_pct_diff_vs_high": 100.0
                * (low_delta - high_delta)
                / high_delta,
                f"L_full_res_{low_resolution}_um": low_lfull,
                f"L_full_res_{high_resolution}_um": high_lfull,
                "L_full_pct_diff_vs_high": 100.0
                * (low_lfull - high_lfull)
                / high_lfull,
            }
        )

    return pd.DataFrame(rows)

def compute_kappa_vs_length(
    df: pd.DataFrame,
    lengths_um: np.ndarray,
) -> pd.DataFrame:
    """
    Compute ideal directional-coupler power coupling versus length.

    Uses:
        kappa^2(L) = sin^2(pi L / (2 L_full))

    Input df must contain:
        gap_um
        L_full_um
    """
    required_columns = {"gap_um", "L_full_um"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[dict[str, float]] = []

    for _, row in df.iterrows():
        gap_um = float(row["gap_um"])
        l_full_um = float(row["L_full_um"])

        for length_um in lengths_um:
            kappa_power = cross_coupled_power(
                length_um=float(length_um),
                l_full_um=l_full_um,
            )

            rows.append(
                {
                    "gap_um": gap_um,
                    "length_um": float(length_um),
                    "kappa_power": kappa_power,
                    "through_power": 1.0 - kappa_power,
                    "L_full_um": l_full_um,
                }
            )

    return pd.DataFrame(rows)

def make_practical_design_table(
    df: pd.DataFrame,
    target_kappa_powers: list[float] | None = None,
    min_length_um: float = 2.0,
    max_length_um: float = 100.0,
) -> pd.DataFrame:
    """
    Create a practical directional-coupler design table.

    For each gap and target kappa^2, compute the required coupler length,
    then flag whether the length is inside a practical range.
    """
    if target_kappa_powers is None:
        target_kappa_powers = [0.025, 0.05, 0.10, 0.20, 0.50]

    required_columns = {"gap_um", "L_full_um"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[dict[str, float | bool | str]] = []

    for _, row in df.iterrows():
        gap_um = float(row["gap_um"])
        l_full_um = float(row["L_full_um"])

        for kappa_power in target_kappa_powers:
            length_um = length_for_target_kappa_power(
                l_full_um=l_full_um,
                kappa_power=kappa_power,
            )

            is_practical = min_length_um <= length_um <= max_length_um

            if length_um < min_length_um:
                note = "very short; fabrication/transition sensitive"
            elif length_um > max_length_um:
                note = "long; layout-area penalty"
            else:
                note = "reasonable first-pass length"

            rows.append(
                {
                    "gap_um": gap_um,
                    "target_kappa_power": kappa_power,
                    "required_length_um": length_um,
                    "L_full_um": l_full_um,
                    "is_practical": is_practical,
                    "note": note,
                }
            )

    return pd.DataFrame(rows)


def _apply_center_zoom(ax, array_shape: tuple[int, int], zoom_fraction: float) -> None:
    """
    Zoom an imshow axis into the center of a 2D array.

    array_shape is the plotted array shape after transpose, i.e. ey_power.T.shape.
    zoom_fraction = 1.0 means full view.
    zoom_fraction = 0.35 means show central 35% of each axis.
    """
    if not 0.0 < zoom_fraction <= 1.0:
        raise ValueError("zoom_fraction must be in (0, 1].")

    ny, nx = array_shape

    if zoom_fraction >= 1.0:
        return

    x_center = (nx - 1) / 2.0
    y_center = (ny - 1) / 2.0

    half_x = nx * zoom_fraction / 2.0
    half_y = ny * zoom_fraction / 2.0

    ax.set_xlim(x_center - half_x, x_center + half_x)
    ax.set_ylim(y_center - half_y, y_center + half_y)

def plot_fields_for_gaps_and_bands(
    gaps_um: list[float],
    bands: list[int],
    spec: DirectionalCouplerSpec,
    settings: MpbCouplerSolveSettings,
    output_path: Path,
    zoom_fraction: float = 1.0,
) -> None:
    """
    Save MPB |Ey| field profiles for selected bands across multiple gaps.

    Grid layout:
        rows    = gaps
        columns = selected bands

    Example:
        gaps_um = [0.10, 0.20, 0.30, 0.40]
        bands = [1, 2, 3, 4]

    This is a validation plot. It helps verify how candidate supermodes
    evolve as the directional-coupler gap changes.
    """
    if not HAS_MEEP:
        raise ImportError("Meep/MPB is not available.")

    if not gaps_um:
        raise ValueError("gaps_um must contain at least one gap.")

    if not bands:
        raise ValueError("bands must contain at least one band index.")

    if any(band < 1 for band in bands):
        raise ValueError("MPB band indices start at 1.")

    if max(bands) > settings.num_bands:
        raise ValueError(
            f"Requested band {max(bands)}, but settings.num_bands={settings.num_bands}."
        )

    nrows = len(gaps_um)
    ncols = len(bands)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, gap_um in enumerate(gaps_um):
        mode_solver, frequency, k_guess, k_min, k_max = build_mpb_mode_solver(
            gap_um=gap_um,
            spec=spec,
            settings=settings,
        )

        with _suppress_stdout_stderr(settings.suppress_output):
            mode_solver.run()

        for col, band in enumerate(bands):
            ax = axes[row, col]

            ex, ey, ez = get_band_field_components(
                mode_solver=mode_solver,
                band=band,
                suppress_output=settings.suppress_output,
            )

            _, ey_fraction, _ = field_energy_fraction(ex, ey, ez)
            sym_score = symmetry_score_y(ey)

            ey_power = np.abs(ey) ** 2
            image = ey_power.T

            im = ax.imshow(
                image,
                origin="lower",
                aspect="auto",
            )

            _apply_center_zoom(
                ax=ax,
                array_shape=image.shape,
                zoom_fraction=zoom_fraction,
            )

            if row == 0:
                ax.set_title(f"Band {band}")

            if col == 0:
                ax.set_ylabel(
                    f"gap={gap_um:.2f} um\n"
                    rf"$f_y$={ey_fraction:.2f}, sym={sym_score:.2f}"
                )
            else:
                ax.set_ylabel(
                    rf"$f_y$={ey_fraction:.2f}, sym={sym_score:.2f}"
                )

            ax.set_xlabel("y index")

            # Keep colorbars small because many subplots can get crowded.
            fig.colorbar(im, ax=ax, shrink=0.75)

    zoom_label = "full view" if zoom_fraction == 1.0 else f"center zoom {zoom_fraction:.2f}"
    fig.suptitle(
        rf"Directional coupler |Ey| fields across gap, {zoom_label}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if not output_path.exists():
        raise RuntimeError(f"Expected field plot was not saved: {output_path}")

def plot_representative_fields(
    gap_um: float,
    spec: DirectionalCouplerSpec,
    settings: MpbCouplerSolveSettings,
    output_path: Path,
) -> None:
    """
    Backward-compatible wrapper.

    The gap_um argument is kept for compatibility, but this function now
    plots selected bands across several gaps.

    Saves:
    - full-view gap/band field grid to output_path
    - zoomed gap/band field grid next to it with '_zoom' suffix
    """
    selected_gaps_um = [0.10, 0.20, 0.30, 0.40, 0.50]
    selected_bands = [1, 2, 3, 4]

    plot_fields_for_gaps_and_bands(
        gaps_um=selected_gaps_um,
        bands=selected_bands,
        spec=spec,
        settings=settings,
        output_path=output_path,
        zoom_fraction=1.0,
    )

    zoom_path = output_path.with_name(f"{output_path.stem}_zoom{output_path.suffix}")

    plot_fields_for_gaps_and_bands(
        gaps_um=selected_gaps_um,
        bands=selected_bands,
        spec=spec,
        settings=settings,
        output_path=zoom_path,
        zoom_fraction=0.35,
    )

def solve_even_odd_supermodes(
    gap_um: float,
    spec: DirectionalCouplerSpec,
    solver: str,
    settings: MpbCouplerSolveSettings,
) -> tuple[float, float, dict[str, float]]:
    """Dispatch to mock or MPB supermode solver."""
    if solver == "mock":
        return solve_even_odd_supermodes_mock(gap_um=gap_um, spec=spec)

    if solver == "mpb":
        return solve_even_odd_supermodes_mpb(
            gap_um=gap_um,
            spec=spec,
            settings=settings,
        )

    raise ValueError(f"Unknown solver: {solver}")


def run_gap_sweep(
    gaps_um: np.ndarray,
    spec: DirectionalCouplerSpec,
    solver: str = "mock",
    settings: MpbCouplerSolveSettings | None = None,
) -> pd.DataFrame:
    """Run a gap sweep and return a table of directional-coupler metrics."""
    if settings is None:
        settings = MpbCouplerSolveSettings()

    rows: list[dict[str, float]] = []

    for gap_um in gaps_um:
        n_even, n_odd, diagnostics = solve_even_odd_supermodes(
            gap_um=float(gap_um),
            spec=spec,
            solver=solver,
            settings=settings,
        )

        lengths = coupling_lengths_from_indices(
            wavelength_um=spec.wavelength_um,
            n_even=n_even,
            n_odd=n_odd,
        )

        rows.append(
            {
                "gap_um": float(gap_um),
                "n_even": n_even,
                "n_odd": n_odd,
                **lengths,
                **diagnostics,
            }
        )

    return pd.DataFrame(rows)


def plot_kappa_vs_length(
    df: pd.DataFrame,
    output_path: Path,
    max_length_um: float | None = None,
) -> None:
    """
    Plot ideal power coupling versus physical coupler length for each gap.
    """
    required_columns = {"gap_um", "length_um", "kappa_power"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for gap_um, group in df.groupby("gap_um"):
        group = group.sort_values("length_um")

        ax.plot(
            group["length_um"],
            group["kappa_power"],
            label=f"gap={gap_um:.2f} um",
        )

    for target in [0.025, 0.05, 0.10, 0.20, 0.50]:
        ax.axhline(target, linestyle="--", linewidth=0.8)
        ax.text(
            0.01,
            target,
            rf"$\kappa^2={target:.3f}$",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Coupler length (um)")
    ax.set_ylabel(r"Cross-coupled power $\kappa^2$")
    ax.set_title("Ideal Directional-Coupler Power Coupling")
    ax.grid(True)
    ax.legend()

    if max_length_um is not None:
        ax.set_xlim(0, max_length_um)

    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if not output_path.exists():
        raise RuntimeError(f"Expected kappa plot was not saved: {output_path}")
    
def plot_gap_sweep(df: pd.DataFrame, output_path: Path) -> None:
    """Plot delta_neff and coupling length versus directional-coupler gap."""
    required_columns = {
        "gap_um",
        "delta_neff",
        "L_full_um",
        "L_3dB_um",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    axes[0].plot(df["gap_um"], df["delta_neff"], marker="o")
    axes[0].set_ylabel(r"$\Delta n_\mathrm{eff}$")
    axes[0].set_title("Directional Coupler Supermode Splitting")
    axes[0].grid(True)

    axes[1].plot(
        df["gap_um"],
        df["L_full_um"],
        marker="o",
        label=r"$L_\mathrm{full}$",
    )
    axes[1].plot(
        df["gap_um"],
        df["L_3dB_um"],
        marker="s",
        label=r"$L_{3\mathrm{dB}}$",
    )
    axes[1].set_xlabel("Gap (um)")
    axes[1].set_ylabel("Length (um)")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_diagnostics(df: pd.DataFrame, output_path: Path) -> None:
    """Plot MPB mode-classification diagnostics if available."""
    required_columns = {
        "gap_um",
        "even_ey_fraction",
        "odd_ey_fraction",
        "even_symmetry_score",
        "odd_symmetry_score",
    }

    if not required_columns.issubset(df.columns):
        return

    if df["even_ey_fraction"].isna().all():
        return

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    axes[0].plot(
        df["gap_um"],
        df["even_ey_fraction"],
        marker="o",
        label="even candidate",
    )
    axes[0].plot(
        df["gap_um"],
        df["odd_ey_fraction"],
        marker="s",
        label="odd candidate",
    )
    axes[0].set_ylabel(r"$E_y$ energy fraction")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        df["gap_um"],
        df["even_symmetry_score"],
        marker="o",
        label="even candidate",
    )
    axes[1].plot(
        df["gap_um"],
        df["odd_symmetry_score"],
        marker="s",
        label="odd candidate",
    )
    axes[1].set_xlabel("Gap (um)")
    axes[1].set_ylabel("Symmetry score")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def add_target_coupling_lengths(
    df: pd.DataFrame,
    target_kappa_powers: list[float] | None = None,
) -> pd.DataFrame:
    """Add required coupler lengths for target power coupling values."""
    if target_kappa_powers is None:
        target_kappa_powers = [0.025, 0.05, 0.10, 0.20, 0.50]

    df = df.copy()

    for kappa_power in target_kappa_powers:
        label = str(kappa_power).replace(".", "p")
        column = f"L_kappa2_{label}_um"

        df[column] = df["L_full_um"].apply(
            lambda l_full: length_for_target_kappa_power(
                l_full_um=float(l_full),
                kappa_power=kappa_power,
            )
        )

    return df

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Directional coupler supermode gap sweep."
    )

    parser.add_argument(
        "--solver",
        choices=["mock", "mpb"],
        default="mock",
        help="Solver backend to use.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="MPB resolution in pixels per micron.",
    )

    parser.add_argument(
        "--padding-y",
        type=float,
        default=2.0,
        help="Horizontal cladding padding in microns.",
    )

    parser.add_argument(
        "--padding-z",
        type=float,
        default=2.0,
        help="Vertical cladding padding in microns.",
    )

    parser.add_argument(
        "--num-bands",
        type=int,
        default=6,
        help="Number of MPB bands to solve.",
    )

    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Run a resolution convergence comparison instead of a single sweep.",
    )

    parser.add_argument(
        "--convergence-resolutions",
        type=int,
        nargs="+",
        default=[30, 50],
        help="List of MPB resolutions for convergence sweep.",
    )

    parser.add_argument(
        "--kappa-length-plot",
        action="store_true",
        help="After the sweep, save kappa^2 versus coupler length plot.",
    )

    parser.add_argument(
        "--max-coupler-length",
        type=float,
        default=100.0,
        help="Maximum coupler length in microns for kappa^2 length plot.",
    )

    parser.add_argument(
    "--design-table",
    action="store_true",
    help="Save a practical coupler design table for target kappa^2 values.",
    )

    parser.add_argument(
        "--min-practical-length",
        type=float,
        default=2.0,
        help="Minimum practical coupler length in microns for design table.",
    )

    parser.add_argument(
        "--max-practical-length",
        type=float,
        default=100.0,
        help="Maximum practical coupler length in microns for design table.",
    )

    return parser.parse_args()



def main() -> None:
    """Run the directional-coupler gap sweep."""
    args = parse_args()

    spec = DirectionalCouplerSpec()

    settings = MpbCouplerSolveSettings(
        resolution=args.resolution,
        padding_y_um=args.padding_y,
        padding_z_um=args.padding_z,
        num_bands=args.num_bands,
    )

    gaps_um = np.array(
        [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
        ]
    )

    if args.convergence:
        if args.solver != "mpb":
            raise ValueError("--convergence should be used with --solver mpb.")

        data_dir = Path("data/sweeps")
        fig_dir = Path("results/figures")
        data_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        convergence_df = run_resolution_convergence_sweep(
            gaps_um=gaps_um,
            resolutions=args.convergence_resolutions,
            spec=spec,
            base_settings=settings,
        )

        convergence_csv_path = data_dir / "directional_coupler_resolution_convergence.csv"
        convergence_fig_path = fig_dir / "directional_coupler_resolution_convergence.png"

        convergence_df.to_csv(convergence_csv_path, index=False)
        plot_resolution_convergence(
            df=convergence_df,
            output_path=convergence_fig_path,
        )

        print()
        print("Directional coupler resolution convergence")
        print("------------------------------------------")
        print(convergence_df.to_string(index=False))
        print()
        print(f"Saved convergence CSV:  {convergence_csv_path}")
        print(f"Saved convergence plot: {convergence_fig_path}")

        if len(args.convergence_resolutions) >= 2:
            low_resolution = min(args.convergence_resolutions)
            high_resolution = max(args.convergence_resolutions)

            summary_df = summarize_resolution_difference(
                df=convergence_df,
                low_resolution=low_resolution,
                high_resolution=high_resolution,
            )

            summary_csv_path = (
                data_dir
                / f"directional_coupler_resolution_{low_resolution}_vs_{high_resolution}.csv"
            )

            summary_df.to_csv(summary_csv_path, index=False)

            print()
            print(f"Resolution {low_resolution} vs {high_resolution} summary")
            print("-------------------------------------------------------")
            print(summary_df.to_string(index=False))
            print()
            print(f"Saved summary CSV: {summary_csv_path}")

        return

    df = run_gap_sweep(
        gaps_um=gaps_um,
        spec=spec,
        solver=args.solver,
        settings=settings,
    )

    df = add_target_coupling_lengths(df)

    data_dir = Path("data/sweeps")
    fig_dir = Path("results/figures")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.solver

    csv_path = data_dir / f"directional_coupler_gap_sweep_{suffix}.csv"
    fig_path = fig_dir / f"directional_coupler_gap_sweep_{suffix}.png"
    diag_path = fig_dir / f"directional_coupler_diagnostics_{suffix}.png"

    # Also write the historical default filename for downstream convenience.
    default_csv_path = data_dir / "directional_coupler_gap_sweep.csv"
    default_fig_path = fig_dir / "directional_coupler_gap_sweep.png"

    df.to_csv(csv_path, index=False)
    df.to_csv(default_csv_path, index=False)

    plot_gap_sweep(df=df, output_path=fig_path)
    plot_gap_sweep(df=df, output_path=default_fig_path)
    plot_diagnostics(df=df, output_path=diag_path)

    if args.kappa_length_plot:
        lengths_um = np.linspace(0.0, args.max_coupler_length, 501)

        kappa_df = compute_kappa_vs_length(
            df=df,
            lengths_um=lengths_um,
        )

        kappa_csv_path = data_dir / f"directional_coupler_kappa_vs_length_{suffix}.csv"
        kappa_fig_path = fig_dir / f"directional_coupler_kappa_vs_length_{suffix}.png"

        kappa_df.to_csv(kappa_csv_path, index=False)
        plot_kappa_vs_length(
            df=kappa_df,
            output_path=kappa_fig_path,
            max_length_um=args.max_coupler_length,
        )

        print(f"Saved kappa-vs-length CSV:  {kappa_csv_path}")
        print(f"Saved kappa-vs-length plot: {kappa_fig_path}")

    if args.design_table:
        design_df = make_practical_design_table(
            df=df,
            min_length_um=args.min_practical_length,g
            max_length_um=args.max_practical_length,
        )

        design_csv_path = data_dir / f"directional_coupler_design_table_{suffix}.csv"
        design_df.to_csv(design_csv_path, index=False)

        print(f"Saved design table: {design_csv_path}")
        
    if args.solver == "mpb":
        field_plot_path = fig_dir / "directional_coupler_fields_gap_sweep.png"

        plot_representative_fields(
            gap_um=0.20,  # kept only for backward-compatible function signature
            spec=spec,
            settings=settings,
            output_path=field_plot_path,
        )

    print()
    print(f"Directional coupler gap sweep using solver: {args.solver}")
    print("----------------------------------------------------------")
    print(df.to_string(index=False))
    print()
    print(f"Saved CSV:        {csv_path}")
    print(f"Saved CSV alias:  {default_csv_path}")
    print(f"Saved plot:       {fig_path}")
    print(f"Saved plot alias: {default_fig_path}")

    if args.solver == "mpb":
        field_zoom_path = field_plot_path.with_name(
            f"{field_plot_path.stem}_zoom{field_plot_path.suffix}"
        )

        print(f"Saved diagnostics:     {diag_path}")
        print(f"Saved field plot:      {field_plot_path}")
        print(f"Saved zoom field plot: {field_zoom_path}")

        target_kappa_power = 0.10
        first_row = df.iloc[0]
        length_um = length_for_target_kappa_power(
            l_full_um=float(first_row["L_full_um"]),
            kappa_power=target_kappa_power,
        )

    print()
    print("Example compact-model bridge")
    print("----------------------------")
    print(f"At gap = {first_row['gap_um']:.3f} um:")
    print(f"  L_full = {first_row['L_full_um']:.3f} um")
    print(f"  target kappa^2 = {target_kappa_power:.3f}")
    print(f"  required coupler length = {length_um:.3f} um")


if __name__ == "__main__":
    main()