# src/compact_models/stacked_rings.py

"""Compact models for directly coupled / stacked ring resonators.

This module starts with a normalized temporal-coupled-mode-theory model for
two directly coupled rings.

The goal is not yet a fully calibrated foundry model. The goal is to capture
the key physics:

    ring-ring coupling -> resonance splitting
    loss/coupling      -> linewidth
    ring detuning      -> asymmetric split resonances

Units:
    wavelength: microns
"""

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.compact_models.ring import wavelength_grid_around_center


@dataclass(frozen=True)
class TwoStackedRingSpec:
    """Normalized two-stacked-ring compact-model parameters.

    Parameters
    ----------
    center_wavelength_um:
        Nominal resonance wavelength of ring 1.
    linewidth_nm:
        Approximate linewidth scale for each uncoupled ring.
    bus_coupling_rate:
        Normalized coupling rate from bus into ring 1.
    intrinsic_loss_rate:
        Normalized internal loss rate in each ring.
    ring_ring_coupling_rate:
        Normalized coupling rate between ring 1 and ring 2.
    ring2_detuning_nm:
        Resonance wavelength offset of ring 2 relative to ring 1.
        Positive means ring 2 is resonant at a longer wavelength.
    """

    center_wavelength_um: float = 1.55
    linewidth_nm: float = 0.20
    bus_coupling_rate: float = 1.0
    intrinsic_loss_rate: float = 0.5
    ring_ring_coupling_rate: float = 0.0
    ring2_detuning_nm: float = 0.0


def wavelength_detuning_units(
    wavelengths_um: np.ndarray,
    center_wavelength_um: float,
    linewidth_nm: float,
) -> np.ndarray:
    """Return normalized wavelength detuning.

    The normalized detuning is approximately:

        delta = (lambda - lambda0) / linewidth

    This is a learning model, so we use wavelength-normalized detuning rather
    than angular-frequency calibrated detuning.
    """
    wavelengths_um = np.asarray(wavelengths_um)
    linewidth_um = linewidth_nm / 1000

    if linewidth_um <= 0:
        raise ValueError("linewidth_nm must be positive.")

    return (wavelengths_um - center_wavelength_um) / linewidth_um


def two_stacked_ring_through_power(
    wavelengths_um: np.ndarray,
    spec: TwoStackedRingSpec,
) -> np.ndarray:
    """Return through-port power for two directly coupled rings.

    This uses a normalized steady-state coupled-mode model.

    Ring 1 is coupled to the bus. Ring 2 is coupled only to ring 1.
    The through field is modeled as:

        s_out = s_in - sqrt(2 gamma_e) a1

    where a1 is the solved ring-1 amplitude.

    This model is intended to show resonance splitting and detuning effects.
    """
    if spec.bus_coupling_rate < 0:
        raise ValueError("bus_coupling_rate must be nonnegative.")

    if spec.intrinsic_loss_rate < 0:
        raise ValueError("intrinsic_loss_rate must be nonnegative.")

    if spec.ring_ring_coupling_rate < 0:
        raise ValueError("ring_ring_coupling_rate must be nonnegative.")

    wavelengths_um = np.asarray(wavelengths_um)

    delta1 = wavelength_detuning_units(
        wavelengths_um=wavelengths_um,
        center_wavelength_um=spec.center_wavelength_um,
        linewidth_nm=spec.linewidth_nm,
    )

    ring2_center_um = spec.center_wavelength_um + spec.ring2_detuning_nm / 1000
    delta2 = wavelength_detuning_units(
        wavelengths_um=wavelengths_um,
        center_wavelength_um=ring2_center_um,
        linewidth_nm=spec.linewidth_nm,
    )

    gamma_e = spec.bus_coupling_rate
    gamma_i = spec.intrinsic_loss_rate
    gamma_total = gamma_e + gamma_i

    mu = spec.ring_ring_coupling_rate

    # Input field amplitude is normalized to 1.
    s_in = 1.0

    through_field = np.empty_like(wavelengths_um, dtype=complex)

    for index, (d1, d2) in enumerate(zip(delta1, delta2)):
        # Linear system:
        # (gamma_total + j*d1) a1 + j*mu a2 = sqrt(2 gamma_e) s_in
        # j*mu a1 + (gamma_i + j*d2) a2 = 0
        #
        # Ring 2 has intrinsic loss only in this simplest model.
        matrix = np.array(
            [
                [gamma_total + 1j * d1, 1j * mu],
                [1j * mu, gamma_i + 1j * d2],
            ],
            dtype=complex,
        )

        rhs = np.array(
            [np.sqrt(2 * gamma_e) * s_in, 0.0],
            dtype=complex,
        )

        a1, _ = np.linalg.solve(matrix, rhs)

        s_out = s_in - np.sqrt(2 * gamma_e) * a1
        through_field[index] = s_out

    return np.abs(through_field) ** 2

def sweep_ring2_detuning(
    wavelengths_um: np.ndarray,
    base_spec: TwoStackedRingSpec,
    detunings_nm: list[float],
) -> dict[str, np.ndarray]:
    """Sweep ring-2 detuning while keeping all other parameters fixed.

    This mimics thermally tuning the second ring while the ring-ring coupling
    rate remains fixed.
    """
    spectra = {}

    for detuning_nm in detunings_nm:
        tuned_spec = TwoStackedRingSpec(
            center_wavelength_um=base_spec.center_wavelength_um,
            linewidth_nm=base_spec.linewidth_nm,
            bus_coupling_rate=base_spec.bus_coupling_rate,
            intrinsic_loss_rate=base_spec.intrinsic_loss_rate,
            ring_ring_coupling_rate=base_spec.ring_ring_coupling_rate,
            ring2_detuning_nm=detuning_nm,
        )

        label = f"detuning={detuning_nm:+.2f} nm"

        spectra[label] = two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=tuned_spec,
        )

    return spectra

def save_stacked_ring_spectrum_csv(
    wavelengths_um: np.ndarray,
    spectra: dict[str, np.ndarray],
    output_path,
) -> None:
    """Save stacked-ring spectra sharing one wavelength grid."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["wavelength_um", *spectra.keys()]

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for index, wavelength_um in enumerate(wavelengths_um):
            row = {"wavelength_um": float(wavelength_um)}

            for label, spectrum in spectra.items():
                row[label] = float(spectrum[index])

            writer.writerow(row)


def plot_stacked_ring_spectra(
    wavelengths_um: np.ndarray,
    spectra: dict[str, np.ndarray],
    output_path,
) -> None:
    """Plot stacked-ring spectra."""
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
    plt.title("Two stacked ring through-port spectra")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_ring2_detuning_sweep(
    wavelengths_um: np.ndarray,
    spectra: dict[str, np.ndarray],
    output_path,
) -> None:
    """Plot two-stacked-ring spectra while sweeping ring-2 detuning."""
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
    plt.title("Two stacked rings: ring-2 detuning sweep at fixed mu")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main() -> None:
    """Generate example stacked-ring spectra."""
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=4.0,
        num_points=2001,
    )

    spectra = {
        "uncoupled ring 2, mu=0": two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(
                center_wavelength_um=1.55,
                linewidth_nm=0.20,
                bus_coupling_rate=0.5,
                intrinsic_loss_rate=0.5,
                ring_ring_coupling_rate=0.0,
                ring2_detuning_nm=0.0,
            ),
        ),
        "weak ring-ring coupling, mu=0.5": two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(
                center_wavelength_um=1.55,
                linewidth_nm=0.20,
                bus_coupling_rate=0.5,
                intrinsic_loss_rate=0.5,
                ring_ring_coupling_rate=0.5,
                ring2_detuning_nm=0.0,
            ),
        ),
        "strong ring-ring coupling, mu=1.2": two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(
                center_wavelength_um=1.55,
                linewidth_nm=0.20,
                bus_coupling_rate=0.5,
                intrinsic_loss_rate=0.5,
                ring_ring_coupling_rate=1.2,
                ring2_detuning_nm=0.0,
            ),
        ),
        "detuned ring 2, mu=1.2": two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(
                center_wavelength_um=1.55,
                linewidth_nm=0.20,
                bus_coupling_rate=0.5,
                intrinsic_loss_rate=0.5,
                ring_ring_coupling_rate=1.2,
                ring2_detuning_nm=0.35,
            ),
        ),
    }

    output_csv = "data/sweeps/two_stacked_ring_spectra.csv"
    save_stacked_ring_spectrum_csv(
        wavelengths_um=wavelengths_um,
        spectra=spectra,
        output_path=output_csv,
    )

    output_plot = "results/figures/two_stacked_ring_spectra.png"
    plot_stacked_ring_spectra(
        wavelengths_um=wavelengths_um,
        spectra=spectra,
        output_path=output_plot,
    )

    print("Two stacked ring spectra")
    print("------------------------")
    print(f"Saved spectrum data to: {output_csv}")
    print(f"Saved spectrum plot to: {output_plot}")

    detuning_wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=5.0,
        num_points=2001,
    )

    fixed_mu_spec = TwoStackedRingSpec(
        center_wavelength_um=1.55,
        linewidth_nm=0.20,
        bus_coupling_rate=0.5,
        intrinsic_loss_rate=0.5,
        ring_ring_coupling_rate=1.0,
        ring2_detuning_nm=0.0,
    )

    detuning_spectra = sweep_ring2_detuning(
        wavelengths_um=detuning_wavelengths_um,
        base_spec=fixed_mu_spec,
        detunings_nm=[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
    )

    detuning_csv = "data/sweeps/two_stacked_ring_detuning_sweep.csv"
    save_stacked_ring_spectrum_csv(
        wavelengths_um=detuning_wavelengths_um,
        spectra=detuning_spectra,
        output_path=detuning_csv,
    )

    detuning_plot = "results/figures/two_stacked_ring_detuning_sweep.png"
    plot_ring2_detuning_sweep(
        wavelengths_um=detuning_wavelengths_um,
        spectra=detuning_spectra,
        output_path=detuning_plot,
    )

    print()
    print("Two stacked ring detuning sweep")
    print("-------------------------------")
    print(f"Fixed ring-ring coupling mu: {fixed_mu_spec.ring_ring_coupling_rate:.3f}")
    print(f"Saved detuning sweep data to: {detuning_csv}")
    print(f"Saved detuning sweep plot to: {detuning_plot}")


if __name__ == "__main__":
    main()
