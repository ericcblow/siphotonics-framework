import numpy as np
import pytest

from src.simulation.directional_coupler_supermodes import (
    DirectionalCouplerSpec,
    coupling_lengths_from_indices,
    cross_coupled_power,
    length_for_target_kappa_power,
    run_gap_sweep,
)


def test_coupling_lengths_positive():
    lengths = coupling_lengths_from_indices(
        wavelength_um=1.55,
        n_even=2.455,
        n_odd=2.435,
    )

    assert lengths["delta_neff"] > 0
    assert lengths["L_full_um"] > 0
    assert lengths["L_3dB_um"] > 0
    assert lengths["L_3dB_um"] == pytest.approx(lengths["L_full_um"] / 2.0)


def test_invalid_supermode_order_raises():
    with pytest.raises(ValueError):
        coupling_lengths_from_indices(
            wavelength_um=1.55,
            n_even=2.435,
            n_odd=2.455,
        )


def test_cross_coupled_power_limits():
    l_full_um = 40.0

    assert cross_coupled_power(length_um=0.0, l_full_um=l_full_um) == pytest.approx(0.0)
    assert cross_coupled_power(length_um=l_full_um / 2.0, l_full_um=l_full_um) == pytest.approx(0.5)
    assert cross_coupled_power(length_um=l_full_um, l_full_um=l_full_um) == pytest.approx(1.0)


def test_length_for_target_kappa_power_inverse():
    l_full_um = 40.0
    target_kappa_power = 0.10

    length_um = length_for_target_kappa_power(
        l_full_um=l_full_um,
        kappa_power=target_kappa_power,
    )

    recovered_kappa_power = cross_coupled_power(
        length_um=length_um,
        l_full_um=l_full_um,
    )

    assert recovered_kappa_power == pytest.approx(target_kappa_power)


def test_gap_sweep_trends():
    spec = DirectionalCouplerSpec()
    gaps_um = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    df = run_gap_sweep(gaps_um=gaps_um, spec=spec)

    assert len(df) == len(gaps_um)

    # Larger gap should reduce supermode splitting in the mock model.
    assert np.all(np.diff(df["delta_neff"]) < 0)

    # Larger gap should increase coupling length.
    assert np.all(np.diff(df["L_full_um"]) > 0)

    # Even mode should have larger effective index than odd mode.
    assert np.all(df["n_even"] > df["n_odd"])