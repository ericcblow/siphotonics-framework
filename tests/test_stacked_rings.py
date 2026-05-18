# tests/test_stacked_rings.py

import numpy as np
import pytest

from src.compact_models.ring import wavelength_grid_around_center
from src.compact_models.stacked_rings import (
    TwoStackedRingSpec,
    sweep_ring2_detuning,
    two_stacked_ring_through_power,
    wavelength_detuning_units,
)


def test_wavelength_detuning_units_has_expected_center():
    wavelengths_um = np.array([1.549, 1.550, 1.551])

    detuning = wavelength_detuning_units(
        wavelengths_um=wavelengths_um,
        center_wavelength_um=1.550,
        linewidth_nm=1.0,
    )

    assert np.isclose(detuning[1], 0.0)
    assert detuning[0] < 0
    assert detuning[2] > 0


def test_two_stacked_ring_power_is_nonnegative():
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=4.0,
        num_points=501,
    )

    transmission = two_stacked_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=TwoStackedRingSpec(),
    )

    assert np.all(transmission >= 0)


def test_ring_ring_coupling_changes_spectrum():
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=4.0,
        num_points=501,
    )

    uncoupled = two_stacked_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=TwoStackedRingSpec(ring_ring_coupling_rate=0.0),
    )

    coupled = two_stacked_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=TwoStackedRingSpec(ring_ring_coupling_rate=1.0),
    )

    assert np.max(np.abs(coupled - uncoupled)) > 1e-3


def test_invalid_stacked_ring_parameters_raise_error():
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=4.0,
        num_points=101,
    )

    with pytest.raises(ValueError):
        two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(bus_coupling_rate=-0.1),
        )

    with pytest.raises(ValueError):
        two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(intrinsic_loss_rate=-0.1),
        )

    with pytest.raises(ValueError):
        two_stacked_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=TwoStackedRingSpec(ring_ring_coupling_rate=-0.1),
        )

    def test_ring2_detuning_sweep_returns_expected_spectra():
        wavelengths_um = wavelength_grid_around_center(
            center_wavelength_um=1.55,
            span_nm=4.0,
            num_points=501,
        )

        base_spec = TwoStackedRingSpec(
            center_wavelength_um=1.55,
            linewidth_nm=0.20,
            bus_coupling_rate=0.5,
            intrinsic_loss_rate=0.5,
            ring_ring_coupling_rate=1.0,
            ring2_detuning_nm=0.0,
        )

        spectra = sweep_ring2_detuning(
            wavelengths_um=wavelengths_um,
            base_spec=base_spec,
            detunings_nm=[-0.5, 0.0, 0.5],
        )

        assert len(spectra) == 3

        for transmission in spectra.values():
            assert transmission.shape == wavelengths_um.shape
            assert np.all(transmission >= 0)