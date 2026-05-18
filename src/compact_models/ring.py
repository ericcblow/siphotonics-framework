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


def estimate_ring_q_factors(
    spec: RingResonatorSpec,
    power_coupling: float,
    round_trip_power_loss: float,
) -> dict[str, float]:
    """Estimate intrinsic, coupling, and loaded Q for a ring.

    Uses small-loss approximations:

        Q_intrinsic ≈ 2π n_g L_rt / (λ * loss)
        Q_coupling  ≈ 2π n_g L_rt / (λ * coupling)
        1/Q_loaded  = 1/Q_intrinsic + 1/Q_coupling

    where loss and coupling are power fractions per round trip/pass.

    These estimates are approximate and are most accurate for small loss and
    small coupling.
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


def estimate_dip_linewidth_um(
    wavelengths_um: np.ndarray,
    transmission: np.ndarray,
    dip_index: int,
) -> float:
    """Estimate full-width at half-depth linewidth of a resonance dip.

    For an all-pass through-port dip:
        half-depth level = T_min + (T_max - T_min) / 2

    The linewidth is the wavelength spacing between the left and right
    half-depth crossings around the selected dip.
    """
    wavelengths_um = np.asarray(wavelengths_um)
    transmission = np.asarray(transmission)

    if not 0 < dip_index < len(transmission) - 1:
        raise ValueError("dip_index must be inside the wavelength array.")

    t_min = transmission[dip_index]
    t_max = np.max(transmission)
    t_half = t_min + 0.5 * (t_max - t_min)

    # Search left crossing.
    left_index = None
    for i in range(dip_index, 0, -1):
        if transmission[i] <= t_half <= transmission[i - 1]:
            left_index = i
            break

    # Search right crossing.
    right_index = None
    for i in range(dip_index, len(transmission) - 1):
        if transmission[i] <= t_half <= transmission[i + 1]:
            right_index = i
            break

    if left_index is None or right_index is None:
        return float("nan")

    def interpolate_crossing(i_low: int, i_high: int) -> float:
        """Linearly interpolate wavelength where transmission crosses t_half."""
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
    """Extract simple resonance metrics from an all-pass ring spectrum.

    This function finds the deepest transmission dip and estimates:
        - resonance wavelength
        - extinction ratio
        - approximate FSR from neighboring dips if available

    Notes
    -----
    This is a simple first-pass metric extractor. It assumes the spectrum has
    visible resonance dips and reasonably smooth sampling.
    """
    wavelengths_um = np.asarray(wavelengths_um)
    transmission = np.asarray(transmission)

    if wavelengths_um.shape != transmission.shape:
        raise ValueError("wavelengths_um and transmission must have the same shape.")

    if len(wavelengths_um) < 3:
        raise ValueError("Need at least 3 wavelength points.")

    # Find local minima: T[i] lower than both neighbors.
    local_min_indices = []
    for i in range(1, len(transmission) - 1):
        if transmission[i] < transmission[i - 1] and transmission[i] < transmission[i + 1]:
            local_min_indices.append(i)

    if not local_min_indices:
        raise ValueError("No resonance dips found.")

    # Deepest dip.
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

    # Estimate FSR from adjacent local minima if possible.
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


def save_ring_sweep_csv(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Save ring sweep results to CSV."""
    import csv
    from pathlib import Path

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
    import csv
    from pathlib import Path

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

def plot_ring_coupling_sweep(
    results: list[dict[str, float]],
    output_path,
) -> None:
    """Plot ring extinction ratio and loaded Q versus power coupling."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    couplings = [row["power_coupling"] for row in results]
    extinction = [row["extinction_ratio_db"] for row in results]
    loaded_q = [row["spectrum_loaded_q"] for row in results]
    linewidth_nm = [row["linewidth_nm"] for row in results]

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
    from pathlib import Path

    import matplotlib.pyplot as plt

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
    """Plot all-pass ring spectra for several coupling values.

    This visually shows how extinction depth and linewidth change with coupling.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

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
            f"k²={power_coupling:.3f}, "
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
    from pathlib import Path

    import matplotlib.pyplot as plt

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
    print(f"mean FSR:              {metrics['mean_fsr_nm']:.3f} nm")
    print(f"Saved ring metrics to: {metrics_csv}")

    print(f"linewidth:             {metrics['linewidth_nm']:.4f} nm")
    print(f"loaded Q:              {metrics['loaded_q']:.1f}")

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
    print(f"Saved spectra versus coupling plot to: {spectra_vs_coupling_plot}")
    print(f"Saved min-transmission sweep plot to: {min_transmission_plot}")