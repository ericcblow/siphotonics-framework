# src/compact_models/ring.py

"""Simple ring-resonator compact-model estimates.

This module uses waveguide-level quantities such as group index to estimate
ring-level quantities such as free spectral range, spectra, resonance metrics,
and simple multi-ring behavior.

Units:
    length: microns
"""

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.pdk.materials import WAVELENGTH_UM


@dataclass(frozen=True)
class RingResonatorSpec:
    """Basic ring resonator specification."""

    radius_um: float = 8.0
    wavelength_um: float = WAVELENGTH_UM
    group_index: float = 4.0497


def ring_round_trip_length_um(spec: RingResonatorSpec) -> float:
    """Return ring round-trip length."""
    return float(2 * np.pi * spec.radius_um)


def estimate_ring_fsr_um(spec: RingResonatorSpec) -> float:
    """Estimate ring free spectral range in microns.

    Approximation:
        FSR ~= lambda^2 / (n_g * L_rt)

    This is valid for small wavelength spacing near the target wavelength.
    """
    round_trip_length_um = ring_round_trip_length_um(spec)

    fsr_um = spec.wavelength_um**2 / (
        spec.group_index * round_trip_length_um
    )

    return float(fsr_um)


def ring_round_trip_length_cm(spec: RingResonatorSpec) -> float:
    """Return ring round-trip length in centimeters.

    Since:
        1 cm = 10,000 um
    """
    return ring_round_trip_length_um(spec) / 10_000


def ring_round_trip_loss_db_from_loss_budget(
    spec: RingResonatorSpec,
    propagation_loss_db_per_cm: float,
    bend_loss_db_per_turn: float = 0.0,
    coupler_excess_loss_db: float = 0.0,
) -> float:
    """Estimate total intrinsic round-trip loss in dB.

    Parameters
    ----------
    spec:
        Ring resonator specification.
    propagation_loss_db_per_cm:
        Straight-equivalent waveguide propagation loss.
    bend_loss_db_per_turn:
        Additional bend/radiation loss for one full ring round trip.
    coupler_excess_loss_db:
        Additional lumped excess loss per round trip from coupler regions.

    Returns
    -------
    float
        Total intrinsic round-trip loss in dB.

    Notes
    -----
    This is an intrinsic loss budget. It does not include intentional bus-ring
    coupling loss, because coupling is an external loading channel.
    """
    if propagation_loss_db_per_cm < 0:
        raise ValueError("propagation_loss_db_per_cm must be nonnegative.")

    if bend_loss_db_per_turn < 0:
        raise ValueError("bend_loss_db_per_turn must be nonnegative.")

    if coupler_excess_loss_db < 0:
        raise ValueError("coupler_excess_loss_db must be nonnegative.")

    propagation_loss_db = (
        propagation_loss_db_per_cm * ring_round_trip_length_cm(spec)
    )

    total_loss_db = (
        propagation_loss_db
        + bend_loss_db_per_turn
        + coupler_excess_loss_db
    )

    return float(total_loss_db)


def round_trip_power_loss_from_loss_db(round_trip_loss_db: float) -> float:
    """Convert round-trip loss in dB to round-trip power loss fraction."""
    if round_trip_loss_db < 0:
        raise ValueError("round_trip_loss_db must be nonnegative.")

    round_trip_power_transmission = 10 ** (-round_trip_loss_db / 10)
    round_trip_power_loss = 1 - round_trip_power_transmission

    return float(round_trip_power_loss)


def round_trip_power_loss_from_loss_budget(
    spec: RingResonatorSpec,
    propagation_loss_db_per_cm: float,
    bend_loss_db_per_turn: float = 0.0,
    coupler_excess_loss_db: float = 0.0,
) -> float:
    """Estimate intrinsic round-trip power loss from a dB loss budget."""
    round_trip_loss_db = ring_round_trip_loss_db_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    return round_trip_power_loss_from_loss_db(round_trip_loss_db)


def estimate_intrinsic_q_from_loss_budget(
    spec: RingResonatorSpec,
    propagation_loss_db_per_cm: float,
    bend_loss_db_per_turn: float = 0.0,
    coupler_excess_loss_db: float = 0.0,
) -> float:
    """Estimate intrinsic Q from a propagation/bend/coupler loss budget."""
    round_trip_power_loss = round_trip_power_loss_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    if round_trip_power_loss <= 0:
        return float("inf")

    q_factors = estimate_ring_q_factors(
        spec=spec,
        power_coupling=0.01,  # dummy valid value; only intrinsic_q is used
        round_trip_power_loss=round_trip_power_loss,
    )

    return float(q_factors["intrinsic_q"])

def estimate_ring_fsr_nm(spec: RingResonatorSpec) -> float:
    """Estimate ring free spectral range in nanometers."""
    return 1000 * estimate_ring_fsr_um(spec)


def estimate_ring_q_factors(
    spec: RingResonatorSpec,
    power_coupling: float,
    round_trip_power_loss: float,
) -> dict[str, float]:
    """Estimate intrinsic, coupling, and loaded Q for a ring.

    Uses small-loss approximations:

        Q_intrinsic ~= 2 pi n_g L_rt / (lambda * loss)
        Q_coupling  ~= 2 pi n_g L_rt / (lambda * coupling)
        1/Q_loaded  = 1/Q_intrinsic + 1/Q_coupling

    where loss and coupling are power fractions per round trip/pass.
    """
    if not 0 < power_coupling < 1:
        raise ValueError("power_coupling must be between 0 and 1.")

    if not 0 < round_trip_power_loss < 1:
        raise ValueError("round_trip_power_loss must be between 0 and 1.")

    round_trip_length_um = ring_round_trip_length_um(spec)

    q_scale = (
        2
        * np.pi
        * spec.group_index
        * round_trip_length_um
        / spec.wavelength_um
    )

    intrinsic_q = q_scale / round_trip_power_loss
    coupling_q = q_scale / power_coupling
    loaded_q = 1 / (1 / intrinsic_q + 1 / coupling_q)

    return {
        "intrinsic_q": float(intrinsic_q),
        "coupling_q": float(coupling_q),
        "analytic_loaded_q": float(loaded_q),
    }


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


def all_pass_ring_through_field(
    wavelengths_um: np.ndarray,
    spec: RingResonatorSpec,
    power_coupling: float = 0.1,
    round_trip_power_loss: float = 0.02,
) -> np.ndarray:
    """Return all-pass ring through-port complex field transfer function.

    This is the field transfer H(lambda) = E_through / E_in.

    Cascaded resonators should multiply field transfer functions first, then
    convert to power:

        H_total = H1 * H2 * ...
        T_total = |H_total|^2
    """
    if not 0 <= power_coupling <= 1:
        raise ValueError("power_coupling must be between 0 and 1.")

    if not 0 <= round_trip_power_loss < 1:
        raise ValueError("round_trip_power_loss must be between 0 and 1.")

    wavelengths_um = np.asarray(wavelengths_um)

    round_trip_length_um = ring_round_trip_length_um(spec)

    t = np.sqrt(1 - power_coupling)
    a = np.sqrt(1 - round_trip_power_loss)

    phase = 2 * np.pi * spec.group_index * round_trip_length_um * (
        1 / wavelengths_um - 1 / spec.wavelength_um
    )

    field_transfer = (t - a * np.exp(-1j * phase)) / (
        1 - a * t * np.exp(-1j * phase)
    )

    return field_transfer


def all_pass_ring_through_power(
    wavelengths_um: np.ndarray,
    spec: RingResonatorSpec,
    power_coupling: float = 0.1,
    round_trip_power_loss: float = 0.02,
) -> np.ndarray:
    """Return all-pass ring through-port power transmission."""
    field_transfer = all_pass_ring_through_field(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=power_coupling,
        round_trip_power_loss=round_trip_power_loss,
    )

    return np.abs(field_transfer) ** 2


def cascade_all_pass_ring_field(
    wavelengths_um: np.ndarray,
    specs: list[RingResonatorSpec],
    power_couplings: list[float],
    round_trip_power_losses: list[float],
) -> np.ndarray:
    """Return cascaded all-pass ring complex field transfer function.

    Each ring is modeled as a through-port all-pass transfer function.
    The total field transfer is the product of each ring's field transfer.
    """
    if not (
        len(specs)
        == len(power_couplings)
        == len(round_trip_power_losses)
    ):
        raise ValueError(
            "specs, power_couplings, and round_trip_power_losses "
            "must have the same length."
        )

    wavelengths_um = np.asarray(wavelengths_um)
    total_field = np.ones_like(wavelengths_um, dtype=complex)

    for spec, power_coupling, round_trip_power_loss in zip(
        specs,
        power_couplings,
        round_trip_power_losses,
    ):
        total_field *= all_pass_ring_through_field(
            wavelengths_um=wavelengths_um,
            spec=spec,
            power_coupling=power_coupling,
            round_trip_power_loss=round_trip_power_loss,
        )

    return total_field


def cascade_all_pass_ring_power(
    wavelengths_um: np.ndarray,
    specs: list[RingResonatorSpec],
    power_couplings: list[float],
    round_trip_power_losses: list[float],
) -> np.ndarray:
    """Return cascaded all-pass ring through-port power transmission."""
    total_field = cascade_all_pass_ring_field(
        wavelengths_um=wavelengths_um,
        specs=specs,
        power_couplings=power_couplings,
        round_trip_power_losses=round_trip_power_losses,
    )

    return np.abs(total_field) ** 2


def add_drop_ring_power(
    wavelengths_um: np.ndarray,
    spec: RingResonatorSpec,
    input_power_coupling: float = 0.1,
    drop_power_coupling: float = 0.1,
    round_trip_power_loss: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return through/drop power transmission for a simple add-drop ring.

    This is a compact learning model for a two-bus ring resonator.
    """
    if not 0 <= input_power_coupling <= 1:
        raise ValueError("input_power_coupling must be between 0 and 1.")

    if not 0 <= drop_power_coupling <= 1:
        raise ValueError("drop_power_coupling must be between 0 and 1.")

    if not 0 <= round_trip_power_loss < 1:
        raise ValueError("round_trip_power_loss must be between 0 and 1.")

    wavelengths_um = np.asarray(wavelengths_um)

    round_trip_length_um = ring_round_trip_length_um(spec)

    t1 = np.sqrt(1 - input_power_coupling)
    t2 = np.sqrt(1 - drop_power_coupling)

    k1 = np.sqrt(input_power_coupling)
    k2 = np.sqrt(drop_power_coupling)

    a = np.sqrt(1 - round_trip_power_loss)

    phase = 2 * np.pi * spec.group_index * round_trip_length_um * (
        1 / wavelengths_um - 1 / spec.wavelength_um
    )

    denominator = 1 - a * t1 * t2 * np.exp(-1j * phase)

    through_field = (
        t1 - a * t2 * np.exp(-1j * phase)
    ) / denominator

    # The sqrt(a) factor represents half-round-trip propagation loss in a
    # symmetric lumped-loss approximation.
    drop_field = (
        -k1 * k2 * np.sqrt(a) * np.exp(-0.5j * phase)
    ) / denominator

    through_power = np.abs(through_field) ** 2
    drop_power = np.abs(drop_field) ** 2

    return through_power, drop_power


def estimate_dip_linewidth_um(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
    dip_index: int,
) -> float:
    """Estimate full-width at half-depth linewidth of a resonance dip."""
    wavelengths_um = np.asarray(wavelengths_um)
    transmission = np.asarray(transmission)

    if not 0 < dip_index < len(transmission) - 1:
        raise ValueError("dip_index must be inside the wavelength array.")

    t_min = transmission[dip_index]
    t_max = np.max(transmission)
    t_half = t_min + 0.5 * (t_max - t_min)

    left_index = None
    for i in range(dip_index, 0, -1):
        if transmission[i] <= t_half <= transmission[i - 1]:
            left_index = i
            break

    right_index = None
    for i in range(dip_index, len(transmission) - 1):
        if transmission[i] <= t_half <= transmission[i + 1]:
            right_index = i
            break

    if left_index is None or right_index is None:
        return float("nan")

    def interpolate_crossing(i_low: int, i_high: int) -> float:
        wl_low = wavelengths_um[i_low]
        wl_high = wavelengths_um[i_high]
        t_low = transmission[i_low]
        t_high = transmission[i_high]

        if np.isclose(t_high, t_low):
            return float(0.5 * (wl_low + wl_high))

        fraction = (t_half - t_low) / (t_high - t_low)
        return float(wl_low + fraction * (wl_high - wl_low))

    left_wavelength_um = interpolate_crossing(left_index, left_index - 1)
    right_wavelength_um = interpolate_crossing(right_index, right_index + 1)

    return float(right_wavelength_um - left_wavelength_um)


def extract_ring_resonance_metrics(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
) -> dict[str, float]:
    """Extract simple resonance metrics from an all-pass ring spectrum."""
    wavelengths_um = np.asarray(wavelengths_um)
    transmission = np.asarray(transmission)

    if wavelengths_um.shape != transmission.shape:
        raise ValueError("wavelengths_um and transmission must have the same shape.")

    if len(wavelengths_um) < 3:
        raise ValueError("Need at least 3 wavelength points.")

    local_min_indices = []
    for i in range(1, len(transmission) - 1):
        if transmission[i] < transmission[i - 1] and transmission[i] < transmission[i + 1]:
            local_min_indices.append(i)

    if not local_min_indices:
        raise ValueError("No resonance dips found.")

    deepest_index = min(local_min_indices, key=lambda i: transmission[i])
    resonance_wavelength_um = float(wavelengths_um[deepest_index])
    min_transmission = float(transmission[deepest_index])
    max_transmission = float(np.max(transmission))

    transmission_floor = 1e-15
    safe_min_transmission = max(min_transmission, transmission_floor)

    extinction_ratio_db = float(
        10 * np.log10(max_transmission / safe_min_transmission)
    )

    linewidth_um = estimate_dip_linewidth_um(
        wavelengths_um=wavelengths_um,
        transmission=transmission,
        dip_index=deepest_index,
    )
    linewidth_nm = 1000 * linewidth_um

    if np.isfinite(linewidth_um) and linewidth_um > 0:
        loaded_q = resonance_wavelength_um / linewidth_um
    else:
        loaded_q = float("nan")

    resonance_wavelengths_um = np.array(
        [wavelengths_um[i] for i in local_min_indices],
        dtype=float,
    )

    if len(resonance_wavelengths_um) >= 2:
        fsr_values_um = np.diff(resonance_wavelengths_um)
        mean_fsr_um = float(np.mean(fsr_values_um))
        mean_fsr_nm = 1000 * mean_fsr_um
    else:
        mean_fsr_um = float("nan")
        mean_fsr_nm = float("nan")

    return {
        "num_resonances_found": float(len(local_min_indices)),
        "deepest_resonance_wavelength_um": resonance_wavelength_um,
        "min_transmission": min_transmission,
        "max_transmission": max_transmission,
        "extinction_ratio_db": extinction_ratio_db,
        "linewidth_um": linewidth_um,
        "linewidth_nm": linewidth_nm,
        "loaded_q": float(loaded_q),
        "mean_fsr_um": mean_fsr_um,
        "mean_fsr_nm": mean_fsr_nm,
    }


def extract_add_drop_metrics(
    wavelengths_um: np.ndarray,
    through_power: np.ndarray,
    drop_power: np.ndarray,
) -> dict[str, float]:
    """Extract simple metrics from an add-drop ring spectrum."""
    wavelengths_um = np.asarray(wavelengths_um)
    through_power = np.asarray(through_power)
    drop_power = np.asarray(drop_power)

    if wavelengths_um.shape != through_power.shape:
        raise ValueError("wavelengths_um and through_power must have the same shape.")

    if wavelengths_um.shape != drop_power.shape:
        raise ValueError("wavelengths_um and drop_power must have the same shape.")

    if len(wavelengths_um) < 3:
        raise ValueError("Need at least 3 wavelength points.")

    drop_peak_indices = []
    for i in range(1, len(drop_power) - 1):
        if drop_power[i] > drop_power[i - 1] and drop_power[i] > drop_power[i + 1]:
            drop_peak_indices.append(i)

    if not drop_peak_indices:
        raise ValueError("No drop peaks found.")

    strongest_drop_index = max(drop_peak_indices, key=lambda i: drop_power[i])

    drop_peak_wavelength_um = float(wavelengths_um[strongest_drop_index])
    max_drop_power = float(drop_power[strongest_drop_index])
    min_through_power = float(np.min(through_power))
    max_through_power = float(np.max(through_power))

    power_floor = 1e-15
    safe_max_drop_power = max(max_drop_power, power_floor)
    safe_min_through_power = max(min_through_power, power_floor)

    drop_insertion_loss_db = float(-10 * np.log10(safe_max_drop_power))
    through_extinction_ratio_db = float(
        10 * np.log10(max_through_power / safe_min_through_power)
    )

    drop_peak_wavelengths_um = np.array(
        [wavelengths_um[i] for i in drop_peak_indices],
        dtype=float,
    )

    if len(drop_peak_wavelengths_um) >= 2:
        fsr_values_um = np.diff(drop_peak_wavelengths_um)
        mean_fsr_um = float(np.mean(fsr_values_um))
        mean_fsr_nm = 1000 * mean_fsr_um
    else:
        mean_fsr_um = float("nan")
        mean_fsr_nm = float("nan")

    return {
        "num_drop_peaks_found": float(len(drop_peak_indices)),
        "drop_peak_wavelength_um": drop_peak_wavelength_um,
        "max_drop_power": max_drop_power,
        "drop_insertion_loss_db": drop_insertion_loss_db,
        "min_through_power": min_through_power,
        "max_through_power": max_through_power,
        "through_extinction_ratio_db": through_extinction_ratio_db,
        "mean_fsr_um": mean_fsr_um,
        "mean_fsr_nm": mean_fsr_nm,
    }


def sweep_ring_coupling(
    spec: RingResonatorSpec,
    power_couplings: list[float],
    round_trip_power_loss: float = 0.02,
    span_nm: float = 40.0,
    num_points: int = 2001,
) -> list[dict[str, float]]:
    """Sweep bus-to-ring power coupling and extract ring metrics."""
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=spec.wavelength_um,
        span_nm=span_nm,
        num_points=num_points,
    )

    results = []

    for power_coupling in power_couplings:
        transmission = all_pass_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=spec,
            power_coupling=power_coupling,
            round_trip_power_loss=round_trip_power_loss,
        )

        metrics = extract_ring_resonance_metrics(
            wavelengths_um=wavelengths_um,
            transmission=transmission,
        )

        q_factors = estimate_ring_q_factors(
            spec=spec,
            power_coupling=power_coupling,
            round_trip_power_loss=round_trip_power_loss,
        )

        results.append(
            {
                "power_coupling": float(power_coupling),
                "round_trip_power_loss": float(round_trip_power_loss),
                "extinction_ratio_db": float(metrics["extinction_ratio_db"]),
                "linewidth_nm": float(metrics["linewidth_nm"]),
                "loaded_q": float(metrics["loaded_q"]),
                "intrinsic_q": float(q_factors["intrinsic_q"]),
                "coupling_q": float(q_factors["coupling_q"]),
                "analytic_loaded_q": float(q_factors["analytic_loaded_q"]),
                "spectrum_loaded_q": float(metrics["loaded_q"]),
                "mean_fsr_nm": float(metrics["mean_fsr_nm"]),
                "min_transmission": float(metrics["min_transmission"]),
                "max_transmission": float(metrics["max_transmission"]),
            }
        )

    return results


def sweep_add_drop_coupling_balance(
    spec: RingResonatorSpec,
    input_power_couplings: list[float],
    drop_power_couplings: list[float],
    round_trip_power_loss: float = 0.02,
    span_nm: float = 40.0,
    num_points: int = 2001,
) -> list[dict[str, float]]:
    """Sweep input/drop coupling values and extract add-drop metrics."""
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=spec.wavelength_um,
        span_nm=span_nm,
        num_points=num_points,
    )

    results = []

    for input_power_coupling in input_power_couplings:
        for drop_power_coupling in drop_power_couplings:
            through_power, drop_power = add_drop_ring_power(
                wavelengths_um=wavelengths_um,
                spec=spec,
                input_power_coupling=input_power_coupling,
                drop_power_coupling=drop_power_coupling,
                round_trip_power_loss=round_trip_power_loss,
            )

            metrics = extract_add_drop_metrics(
                wavelengths_um=wavelengths_um,
                through_power=through_power,
                drop_power=drop_power,
            )

            results.append(
                {
                    "input_power_coupling": float(input_power_coupling),
                    "drop_power_coupling": float(drop_power_coupling),
                    "round_trip_power_loss": float(round_trip_power_loss),
                    "max_drop_power": float(metrics["max_drop_power"]),
                    "drop_insertion_loss_db": float(
                        metrics["drop_insertion_loss_db"]
                    ),
                    "min_through_power": float(metrics["min_through_power"]),
                    "through_extinction_ratio_db": float(
                        metrics["through_extinction_ratio_db"]
                    ),
                    "mean_fsr_nm": float(metrics["mean_fsr_nm"]),
                }
            )

    return results

def extract_multiple_spectrum_metrics(
    wavelengths_um: np.ndarray,
    spectra: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    """Extract simple resonance metrics for multiple through spectra."""
    results = []

    for label, transmission in spectra.items():
        metrics = extract_ring_resonance_metrics(
            wavelengths_um=wavelengths_um,
            transmission=transmission,
        )

        results.append(
            {
                "spectrum_label": label,
                "num_resonances_found": float(
                    metrics["num_resonances_found"]
                ),
                "deepest_resonance_wavelength_um": float(
                    metrics["deepest_resonance_wavelength_um"]
                ),
                "min_transmission": float(metrics["min_transmission"]),
                "max_transmission": float(metrics["max_transmission"]),
                "extinction_ratio_db": float(metrics["extinction_ratio_db"]),
                "linewidth_nm": float(metrics["linewidth_nm"]),
                "loaded_q": float(metrics["loaded_q"]),
                "mean_fsr_nm": float(metrics["mean_fsr_nm"]),
            }
        )

    return results

def save_ring_spectrum_csv(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
    output_path,
) -> None:
    """Save ring spectrum to CSV."""
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


def save_table_csv(
    rows: list[dict[str, object]],
    output_path,
) -> None:
    """Save a list of dictionary rows to CSV."""
    if not rows:
        raise ValueError("Cannot save empty table.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_add_drop_spectrum_csv(
    wavelengths_um: np.ndarray,
    through_power: np.ndarray,
    drop_power: np.ndarray,
    output_path,
) -> None:
    """Save add-drop ring spectrum to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["wavelength_um", "through_power", "drop_power"],
        )
        writer.writeheader()

        for wavelength_um, through, drop in zip(
            wavelengths_um,
            through_power,
            drop_power,
        ):
            writer.writerow(
                {
                    "wavelength_um": float(wavelength_um),
                    "through_power": float(through),
                    "drop_power": float(drop),
                }
            )


def save_ring_sweep_csv(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Save ring sweep results to CSV."""
    if not results:
        raise ValueError("Cannot save empty sweep results.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def save_ring_metrics_csv(
    metrics: dict[str, float],
    output_path,
) -> None:
    """Save ring resonance metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def plot_ring_spectrum(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
    output_path,
) -> None:
    """Plot all-pass ring through-port spectrum."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    plt.plot(wavelengths_um * 1000, transmission)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title("All-pass ring through-port spectrum")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cascaded_ring_spectra(
    wavelengths_um: np.ndarray,
    spectra: dict[str, np.ndarray],
    output_path,
) -> None:
    """Plot several cascaded all-pass ring spectra."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))

    for label, transmission in spectra.items():
        plt.plot(
            wavelengths_um * 1000,
            transmission,
            label=label,
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title("Cascaded all-pass ring spectra")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ring_coupling_sweep(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Plot ring extinction ratio and loaded Q versus power coupling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    couplings = [row["power_coupling"] for row in results]
    extinction = [row["extinction_ratio_db"] for row in results]
    loaded_q = [row["spectrum_loaded_q"] for row in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    extinction_color = "tab:blue"
    q_color = "tab:orange"

    extinction_line = ax1.plot(
        couplings,
        extinction,
        marker="o",
        color=extinction_color,
        label="Extinction ratio",
    )
    ax1.set_xlabel("Power coupling")
    ax1.set_ylabel("Extinction ratio (dB)", color=extinction_color)
    ax1.tick_params(axis="y", labelcolor=extinction_color)
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    q_line = ax2.plot(
        couplings,
        loaded_q,
        marker="s",
        linestyle="--",
        color=q_color,
        label="Loaded Q",
    )
    ax2.set_ylabel("Loaded Q", color=q_color)
    ax2.tick_params(axis="y", labelcolor=q_color)

    lines = extinction_line + q_line
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    title = "All-pass ring coupling sweep"
    subtitle = "Extinction peaks near critical coupling; loaded Q decreases with stronger coupling"
    plt.title(f"{title}\n{subtitle}", fontsize=11)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ring_coupling_linewidth_sweep(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Plot ring linewidth versus power coupling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    couplings = [row["power_coupling"] for row in results]
    linewidth_nm = [row["linewidth_nm"] for row in results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        couplings,
        linewidth_nm,
        marker="o",
        color="tab:green",
        label="Linewidth",
    )
    plt.xlabel("Power coupling")
    plt.ylabel("Linewidth (nm)")
    plt.title("All-pass ring linewidth versus coupling")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ring_spectra_for_couplings(
    spec: RingResonatorSpec,
    power_couplings: list[float],
    round_trip_power_loss: float,
    output_path,
    span_nm: float = 20.0,
    num_points: int = 2001,
) -> None:
    """Plot all-pass ring spectra for several coupling values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=spec.wavelength_um,
        span_nm=span_nm,
        num_points=num_points,
    )

    plt.figure(figsize=(8, 4.8))

    for power_coupling in power_couplings:
        transmission = all_pass_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=spec,
            power_coupling=power_coupling,
            round_trip_power_loss=round_trip_power_loss,
        )

        metrics = extract_ring_resonance_metrics(
            wavelengths_um=wavelengths_um,
            transmission=transmission,
        )

        label = (
            f"k2={power_coupling:.3f}, "
            f"ER={metrics['extinction_ratio_db']:.1f} dB"
        )

        plt.plot(
            wavelengths_um * 1000,
            transmission,
            label=label,
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Through power")
    plt.title("All-pass ring spectra versus coupling")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ring_coupling_min_transmission_sweep(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Plot minimum through transmission versus power coupling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    couplings = [row["power_coupling"] for row in results]
    min_transmission = [row["min_transmission"] for row in results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        couplings,
        min_transmission,
        marker="o",
        color="tab:purple",
        label="Minimum through power",
    )
    plt.xlabel("Power coupling")
    plt.ylabel("Minimum through power")
    plt.title("All-pass ring minimum transmission versus coupling")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ring_loaded_q_comparison(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Plot analytic loaded Q versus spectrum-extracted loaded Q."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    couplings = [row["power_coupling"] for row in results]
    spectrum_loaded_q = [row["spectrum_loaded_q"] for row in results]
    analytic_loaded_q = [row["analytic_loaded_q"] for row in results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        couplings,
        spectrum_loaded_q,
        marker="o",
        label="Spectrum-extracted loaded Q",
    )
    plt.plot(
        couplings,
        analytic_loaded_q,
        marker="s",
        linestyle="--",
        label="Analytic loaded Q",
    )
    plt.xlabel("Power coupling")
    plt.ylabel("Loaded Q")
    plt.title("Loaded Q comparison versus coupling")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_add_drop_spectrum(
    wavelengths_um: np.ndarray,
    through_power: np.ndarray,
    drop_power: np.ndarray,
    output_path,
) -> None:
    """Plot add-drop ring through/drop spectra."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        wavelengths_um * 1000,
        through_power,
        label="Through port",
        color="tab:blue",
    )
    plt.plot(
        wavelengths_um * 1000,
        drop_power,
        label="Drop port",
        color="tab:orange",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power transmission")
    plt.title("Add-drop ring spectrum")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_add_drop_coupling_heatmap(
    results: list[dict[str, float]],
    metric_name: str,
    output_path,
    title: str,
    colorbar_label: str,
) -> None:
    """Plot a heatmap for an add-drop coupling-balance metric."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_values = sorted({row["input_power_coupling"] for row in results})
    drop_values = sorted({row["drop_power_coupling"] for row in results})

    metric_grid = np.empty((len(drop_values), len(input_values)))

    for row in results:
        x_index = input_values.index(row["input_power_coupling"])
        y_index = drop_values.index(row["drop_power_coupling"])
        metric_grid[y_index, x_index] = row[metric_name]

    extent = [
        min(input_values),
        max(input_values),
        min(drop_values),
        max(drop_values),
    ]

    plt.figure(figsize=(6.5, 5.2))
    image = plt.imshow(
        metric_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(image, label=colorbar_label)
    plt.xlabel("Input power coupling k1^2")
    plt.ylabel("Drop power coupling k2^2")
    plt.title(title)
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

    propagation_loss_db_per_cm = 2.0
    bend_loss_db_per_turn = 0.05
    coupler_excess_loss_db = 0.02

    round_trip_loss_db = ring_round_trip_loss_db_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    budget_round_trip_power_loss = round_trip_power_loss_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    budget_intrinsic_q = estimate_intrinsic_q_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    print()
    print("Ring intrinsic loss budget")
    print("--------------------------")
    print(f"propagation loss:       {propagation_loss_db_per_cm:.3f} dB/cm")
    print(f"bend loss per turn:     {bend_loss_db_per_turn:.3f} dB/turn")
    print(f"coupler excess loss:    {coupler_excess_loss_db:.3f} dB/round trip")
    print(f"round-trip loss:        {round_trip_loss_db:.6f} dB")
    print(f"round-trip power loss:  {budget_round_trip_power_loss:.6f}")
    print(f"intrinsic Q estimate:   {budget_intrinsic_q:.1f}")

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

    metrics = extract_ring_resonance_metrics(
        wavelengths_um=wavelengths_um,
        transmission=transmission,
    )

    metrics_csv = "data/sweeps/ring_all_pass_metrics.csv"
    save_ring_metrics_csv(metrics, metrics_csv)

    print()
    print("Ring spectrum metrics")
    print("---------------------")
    print(f"resonances found:      {metrics['num_resonances_found']:.0f}")
    print(
        "deepest resonance:     "
        f"{metrics['deepest_resonance_wavelength_um'] * 1000:.3f} nm"
    )
    print(f"min transmission:      {metrics['min_transmission']:.6f}")
    print(f"max transmission:      {metrics['max_transmission']:.6f}")
    print(f"extinction ratio:      {metrics['extinction_ratio_db']:.3f} dB")
    print(f"linewidth:             {metrics['linewidth_nm']:.4f} nm")
    print(f"loaded Q:              {metrics['loaded_q']:.1f}")
    print(f"mean FSR:              {metrics['mean_fsr_nm']:.3f} nm")
    print(f"Saved ring spectrum to: {spectrum_csv}")
    print(f"Saved ring spectrum plot to: {spectrum_plot}")
    print(f"Saved ring metrics to: {metrics_csv}")

    budget_transmission = all_pass_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.02,
        round_trip_power_loss=budget_round_trip_power_loss,
    )

    budget_spectrum_csv = "data/sweeps/ring_loss_budget_all_pass_spectrum.csv"
    save_ring_spectrum_csv(
        wavelengths_um,
        budget_transmission,
        budget_spectrum_csv,
    )

    budget_spectrum_plot = "results/figures/ring_loss_budget_all_pass_spectrum.png"
    plot_ring_spectrum(
        wavelengths_um,
        budget_transmission,
        budget_spectrum_plot,
    )

    print(f"Saved loss-budget spectrum to: {budget_spectrum_csv}")
    print(f"Saved loss-budget spectrum plot to: {budget_spectrum_plot}")

    propagation_loss_db_per_cm = 2.0
    bend_loss_db_per_turn = 0.05
    coupler_excess_loss_db = 0.02

    round_trip_loss_db = ring_round_trip_loss_db_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    budget_round_trip_power_loss = round_trip_power_loss_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    budget_intrinsic_q = estimate_intrinsic_q_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=propagation_loss_db_per_cm,
        bend_loss_db_per_turn=bend_loss_db_per_turn,
        coupler_excess_loss_db=coupler_excess_loss_db,
    )

    print()
    print("Ring intrinsic loss budget")
    print("--------------------------")
    print(f"propagation loss:       {propagation_loss_db_per_cm:.3f} dB/cm")
    print(f"bend loss per turn:     {bend_loss_db_per_turn:.3f} dB/turn")
    print(f"coupler excess loss:    {coupler_excess_loss_db:.3f} dB/round trip")
    print(f"round-trip loss:        {round_trip_loss_db:.6f} dB")
    print(f"round-trip power loss:  {budget_round_trip_power_loss:.6f}")
    print(f"intrinsic Q estimate:   {budget_intrinsic_q:.1f}")

    coupling_results = sweep_ring_coupling(
        spec=spec,
        power_couplings=[
            0.002,
            0.005,
            0.008,
            0.010,
            0.012,
            0.014,
            0.016,
            0.018,
            0.020,
            0.022,
            0.024,
            0.026,
            0.028,
            0.030,
            0.035,
            0.040,
            0.050,
            0.075,
            0.100,
            0.150,
            0.200,
            0.300,
        ],
        round_trip_power_loss=0.02,
        span_nm=40.0,
        num_points=2001,
    )

    coupling_csv = "data/sweeps/ring_coupling_sweep.csv"
    save_ring_sweep_csv(coupling_results, coupling_csv)

    coupling_plot = "results/figures/ring_coupling_sweep.png"
    plot_ring_coupling_sweep(coupling_results, coupling_plot)

    linewidth_plot = "results/figures/ring_coupling_linewidth_sweep.png"
    plot_ring_coupling_linewidth_sweep(coupling_results, linewidth_plot)

    spectra_vs_coupling_plot = "results/figures/ring_spectra_vs_coupling.png"
    plot_ring_spectra_for_couplings(
        spec=spec,
        power_couplings=[0.005, 0.010, 0.020, 0.050, 0.100],
        round_trip_power_loss=0.02,
        output_path=spectra_vs_coupling_plot,
        span_nm=20.0,
        num_points=2001,
    )

    min_transmission_plot = "results/figures/ring_coupling_min_transmission_sweep.png"
    plot_ring_coupling_min_transmission_sweep(
        coupling_results,
        min_transmission_plot,
    )

    q_comparison_plot = "results/figures/ring_loaded_q_comparison.png"
    plot_ring_loaded_q_comparison(coupling_results, q_comparison_plot)

    print()
    print("Ring coupling sweep")
    print("-------------------")
    print(
        "power_coupling, extinction_ratio_db, linewidth_nm, "
        "spectrum_loaded_q, analytic_loaded_q, coupling_q"
    )
    for row in coupling_results:
        print(
            f"{row['power_coupling']:.3f}, "
            f"{row['extinction_ratio_db']:.3f}, "
            f"{row['linewidth_nm']:.4f}, "
            f"{row['spectrum_loaded_q']:.1f}, "
            f"{row['analytic_loaded_q']:.1f}, "
            f"{row['coupling_q']:.1f}"
        )

    print(f"Saved coupling sweep to: {coupling_csv}")
    print(f"Saved coupling sweep plot to: {coupling_plot}")
    print(f"Saved linewidth sweep plot to: {linewidth_plot}")
    print(f"Saved spectra versus coupling plot to: {spectra_vs_coupling_plot}")
    print(f"Saved min-transmission sweep plot to: {min_transmission_plot}")
    print(f"Saved loaded-Q comparison plot to: {q_comparison_plot}")

    add_drop_wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=spec.wavelength_um,
        span_nm=40.0,
        num_points=2001,
    )

    through_power, drop_power = add_drop_ring_power(
        wavelengths_um=add_drop_wavelengths_um,
        spec=spec,
        input_power_coupling=0.05,
        drop_power_coupling=0.05,
        round_trip_power_loss=0.02,
    )

    add_drop_csv = "data/sweeps/ring_add_drop_spectrum.csv"
    save_add_drop_spectrum_csv(
        add_drop_wavelengths_um,
        through_power,
        drop_power,
        add_drop_csv,
    )

    add_drop_plot = "results/figures/ring_add_drop_spectrum.png"
    plot_add_drop_spectrum(
        add_drop_wavelengths_um,
        through_power,
        drop_power,
        add_drop_plot,
    )

    add_drop_metrics = extract_add_drop_metrics(
        wavelengths_um=add_drop_wavelengths_um,
        through_power=through_power,
        drop_power=drop_power,
    )

    add_drop_metrics_csv = "data/sweeps/ring_add_drop_metrics.csv"
    save_ring_metrics_csv(add_drop_metrics, add_drop_metrics_csv)

    print()
    print("Add-drop ring metrics")
    print("---------------------")
    print(f"drop peaks found:       {add_drop_metrics['num_drop_peaks_found']:.0f}")
    print(
        "drop peak wavelength:   "
        f"{add_drop_metrics['drop_peak_wavelength_um'] * 1000:.3f} nm"
    )
    print(f"max drop power:         {add_drop_metrics['max_drop_power']:.6f}")
    print(
        "drop insertion loss:    "
        f"{add_drop_metrics['drop_insertion_loss_db']:.3f} dB"
    )
    print(f"min through power:      {add_drop_metrics['min_through_power']:.6f}")
    print(
        "through extinction:     "
        f"{add_drop_metrics['through_extinction_ratio_db']:.3f} dB"
    )
    print(f"mean FSR:               {add_drop_metrics['mean_fsr_nm']:.3f} nm")
    print(f"Saved add-drop spectrum to: {add_drop_csv}")
    print(f"Saved add-drop spectrum plot to: {add_drop_plot}")
    print(f"Saved add-drop metrics to: {add_drop_metrics_csv}")

    add_drop_coupling_results = sweep_add_drop_coupling_balance(
        spec=spec,
        input_power_couplings=[0.01, 0.02, 0.05, 0.10, 0.20],
        drop_power_couplings=[0.01, 0.02, 0.05, 0.10, 0.20],
        round_trip_power_loss=0.02,
        span_nm=40.0,
        num_points=2001,
    )

    add_drop_coupling_csv = "data/sweeps/ring_add_drop_coupling_balance.csv"
    save_ring_sweep_csv(add_drop_coupling_results, add_drop_coupling_csv)

    drop_power_heatmap = "results/figures/ring_add_drop_max_drop_power_heatmap.png"
    plot_add_drop_coupling_heatmap(
        results=add_drop_coupling_results,
        metric_name="max_drop_power",
        output_path=drop_power_heatmap,
        title="Add-drop ring max drop power",
        colorbar_label="Max drop power",
    )

    drop_il_heatmap = "results/figures/ring_add_drop_insertion_loss_heatmap.png"
    plot_add_drop_coupling_heatmap(
        results=add_drop_coupling_results,
        metric_name="drop_insertion_loss_db",
        output_path=drop_il_heatmap,
        title="Add-drop ring drop insertion loss",
        colorbar_label="Drop insertion loss (dB)",
    )

    print()
    print("Add-drop coupling balance sweep")
    print("-------------------------------")
    print(f"Saved add-drop coupling sweep to: {add_drop_coupling_csv}")
    print(f"Saved max-drop-power heatmap to: {drop_power_heatmap}")
    print(f"Saved drop-insertion-loss heatmap to: {drop_il_heatmap}")

    cascade_wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=spec.wavelength_um,
        span_nm=30.0,
        num_points=2001,
    )
    wl_detune = .3
    one_ring_specs = [
        RingResonatorSpec(radius_um=8.0, wavelength_um=1.55, group_index=4.0497)
    ]

    two_identical_specs = [
        RingResonatorSpec(radius_um=8.0, wavelength_um=1.55, group_index=4.0497),
        RingResonatorSpec(radius_um=8.0, wavelength_um=1.55, group_index=4.0497),
    ]

    three_identical_specs = [
        RingResonatorSpec(radius_um=8.0, wavelength_um=1.55, group_index=4.0497),
        RingResonatorSpec(radius_um=8.0, wavelength_um=1.55, group_index=4.0497),
        RingResonatorSpec(radius_um=8.0, wavelength_um=1.55, group_index=4.0497),
    ]

    three_detuned_specs = [
        RingResonatorSpec(radius_um=8.00, wavelength_um=1.55, group_index=4.0497),
        RingResonatorSpec(radius_um=8.00+1*wl_detune, wavelength_um=1.55, group_index=4.0497),
        RingResonatorSpec(radius_um=8.00-1*wl_detune, wavelength_um=1.55, group_index=4.0497),
    ]

    cascade_power_coupling = 0.20
    cascade_round_trip_loss = 0.05

    cascade_spectra = {
        "1 ring": cascade_all_pass_ring_power(
            wavelengths_um=cascade_wavelengths_um,
            specs=one_ring_specs,
            power_couplings=[cascade_power_coupling],
            round_trip_power_losses=[cascade_round_trip_loss],
        ),
        "2 identical rings": cascade_all_pass_ring_power(
            wavelengths_um=cascade_wavelengths_um,
            specs=two_identical_specs,
            power_couplings=[
                cascade_power_coupling,
                cascade_power_coupling,
            ],
            round_trip_power_losses=[
                cascade_round_trip_loss,
                cascade_round_trip_loss,
            ],
        ),
        "3 identical rings": cascade_all_pass_ring_power(
            wavelengths_um=cascade_wavelengths_um,
            specs=three_identical_specs,
            power_couplings=[
                cascade_power_coupling,
                cascade_power_coupling,
                cascade_power_coupling,
            ],
            round_trip_power_losses=[
                cascade_round_trip_loss,
                cascade_round_trip_loss,
                cascade_round_trip_loss,
            ],
        ),
        "3 detuned rings": cascade_all_pass_ring_power(
            wavelengths_um=cascade_wavelengths_um,
            specs=three_detuned_specs,
            power_couplings=[
                cascade_power_coupling,
                cascade_power_coupling,
                cascade_power_coupling,
            ],
            round_trip_power_losses=[
                cascade_round_trip_loss,
                cascade_round_trip_loss,
                cascade_round_trip_loss,
            ],
        ),
    }

    cascade_plot = "results/figures/ring_cascade_spectrum.png"
    plot_cascaded_ring_spectra(
        wavelengths_um=cascade_wavelengths_um,
        spectra=cascade_spectra,
        output_path=cascade_plot,
    )

    print()
    print("Cascaded all-pass ring spectra")
    print("------------------------------")
    print(f"Saved cascaded ring spectrum plot to: {cascade_plot}")

    cascade_metrics = extract_multiple_spectrum_metrics(
        wavelengths_um=cascade_wavelengths_um,
        spectra=cascade_spectra,
    )

    cascade_metrics_csv = "data/sweeps/ring_cascade_metrics.csv"
    save_table_csv(cascade_metrics, cascade_metrics_csv)

    print(f"Saved cascaded ring metrics to: {cascade_metrics_csv}")

    print()
    print("Cascaded all-pass ring metrics")
    print("------------------------------")
    print("label, extinction_ratio_db, linewidth_nm, loaded_q")
    for row in cascade_metrics:
        print(
            f"{row['spectrum_label']}, "
            f"{row['extinction_ratio_db']:.3f}, "
            f"{row['linewidth_nm']:.4f}, "
            f"{row['loaded_q']:.1f}"
        )