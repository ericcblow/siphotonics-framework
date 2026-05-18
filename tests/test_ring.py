# tests/test_ring.py
import numpy as np
import pytest


from src.compact_models.ring import (
    RingResonatorSpec,
    all_pass_ring_through_power,
    estimate_ring_fsr_nm,
    estimate_ring_fsr_um,
    estimate_ring_q_factors,
    extract_ring_resonance_metrics,
    ring_round_trip_length_um,
    wavelength_grid_around_center,
    sweep_ring_coupling,
    add_drop_ring_power,
    extract_add_drop_metrics,
    sweep_add_drop_coupling_balance,
    all_pass_ring_through_field,
    cascade_all_pass_ring_power,
    extract_multiple_spectrum_metrics,
    estimate_intrinsic_q_from_loss_budget,
    ring_round_trip_length_cm,
    ring_round_trip_loss_db_from_loss_budget,
    round_trip_power_loss_from_loss_budget,
    round_trip_power_loss_from_loss_db,
)



def test_ring_round_trip_length_is_positive():
    spec = RingResonatorSpec(radius_um=10.0)

    assert ring_round_trip_length_um(spec) > 0


def test_ring_fsr_is_positive():
    spec = RingResonatorSpec(
        radius_um=10.0,
        wavelength_um=1.55,
        group_index=4.0,
    )

    assert estimate_ring_fsr_um(spec) > 0
    assert estimate_ring_fsr_nm(spec) > 0


def test_larger_radius_reduces_fsr():
    small_ring = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    large_ring = RingResonatorSpec(radius_um=20.0, group_index=4.0)

    assert estimate_ring_fsr_nm(large_ring) < estimate_ring_fsr_nm(small_ring)


def test_larger_group_index_reduces_fsr():
    low_ng = RingResonatorSpec(radius_um=10.0, group_index=3.5)
    high_ng = RingResonatorSpec(radius_um=10.0, group_index=4.5)

    assert estimate_ring_fsr_nm(high_ng) < estimate_ring_fsr_nm(low_ng)
    

def test_wavelength_grid_has_expected_length():
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=2001,
    )

    assert len(wavelengths_um) == 2001
    assert np.isclose(wavelengths_um[0], 1.53)
    assert np.isclose(wavelengths_um[-1], 1.57)


def test_all_pass_ring_transmission_is_bounded():
    spec = RingResonatorSpec(
        radius_um=10.0,
        wavelength_um=1.55,
        group_index=4.0,
    )
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=501,
    )

    transmission = all_pass_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.1,
        round_trip_power_loss=0.02,
    )

    assert np.all(transmission >= 0)
    assert np.all(transmission <= 1.0 + 1e-12)


def test_all_pass_ring_spectrum_varies_with_wavelength():
    spec = RingResonatorSpec(
        radius_um=10.0,
        wavelength_um=1.55,
        group_index=4.0,
    )
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=501,
    )

    transmission = all_pass_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.1,
        round_trip_power_loss=0.02,
    )

    assert np.max(transmission) - np.min(transmission) > 0.01


def test_invalid_ring_coupling_raises_error():
    spec = RingResonatorSpec()
    wavelengths_um = wavelength_grid_around_center(1.55)

    with pytest.raises(ValueError):
        all_pass_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=spec,
            power_coupling=-0.1,
        )

    with pytest.raises(ValueError):
        all_pass_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=spec,
            power_coupling=1.1,
        )


def test_invalid_ring_loss_raises_error():
    spec = RingResonatorSpec()
    wavelengths_um = wavelength_grid_around_center(1.55)

    with pytest.raises(ValueError):
        all_pass_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=spec,
            round_trip_power_loss=-0.1,
        )

    with pytest.raises(ValueError):
        all_pass_ring_through_power(
            wavelengths_um=wavelengths_um,
            spec=spec,
            round_trip_power_loss=1.0,
        )

def test_ring_metric_extraction_finds_resonances():
    spec = RingResonatorSpec(
        radius_um=10.0,
        wavelength_um=1.55,
        group_index=4.0,
    )
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=2001,
    )

    transmission = all_pass_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.1,
        round_trip_power_loss=0.02,
    )

    metrics = extract_ring_resonance_metrics(wavelengths_um, transmission)

    assert metrics["num_resonances_found"] >= 2
    assert metrics["min_transmission"] < metrics["max_transmission"]
    assert metrics["extinction_ratio_db"] > 0
    assert metrics["mean_fsr_nm"] > 0
    assert metrics["linewidth_nm"] > 0
    assert metrics["loaded_q"] > 0


def test_ring_q_factors_are_positive():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)

    q_factors = estimate_ring_q_factors(
        spec=spec,
        power_coupling=0.1,
        round_trip_power_loss=0.02,
    )

    assert q_factors["intrinsic_q"] > 0
    assert q_factors["coupling_q"] > 0
    assert q_factors["analytic_loaded_q"] > 0


def test_loaded_q_is_less_than_intrinsic_and_coupling_q():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)

    q_factors = estimate_ring_q_factors(
        spec=spec,
        power_coupling=0.1,
        round_trip_power_loss=0.02,
    )

    assert q_factors["analytic_loaded_q"] < q_factors["intrinsic_q"]
    assert q_factors["analytic_loaded_q"] < q_factors["coupling_q"]


def test_stronger_coupling_reduces_coupling_q():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)

    weak = estimate_ring_q_factors(
        spec=spec,
        power_coupling=0.02,
        round_trip_power_loss=0.02,
    )
    strong = estimate_ring_q_factors(
        spec=spec,
        power_coupling=0.2,
        round_trip_power_loss=0.02,
    )

    assert strong["coupling_q"] < weak["coupling_q"]
    assert strong["analytic_loaded_q"] < weak["analytic_loaded_q"]

def test_coupling_sweep_includes_q_decomposition():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)

    results = sweep_ring_coupling(
        spec=spec,
        power_couplings=[0.02, 0.05],
        round_trip_power_loss=0.02,
        span_nm=40.0,
        num_points=1001,
    )

    assert len(results) == 2

    for row in results:
        assert row["intrinsic_q"] > 0
        assert row["coupling_q"] > 0
        assert row["analytic_loaded_q"] > 0
        assert row["spectrum_loaded_q"] > 0
        assert row["loaded_q"] == row["spectrum_loaded_q"]

def test_add_drop_ring_power_is_bounded():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=501,
    )

    through_power, drop_power = add_drop_ring_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        input_power_coupling=0.05,
        drop_power_coupling=0.05,
        round_trip_power_loss=0.02,
    )

    assert np.all(through_power >= 0)
    assert np.all(drop_power >= 0)
    assert np.all(through_power <= 1.0 + 1e-12)
    assert np.all(drop_power <= 1.0 + 1e-12)


def test_add_drop_ring_has_drop_peaks():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=501,
    )

    _, drop_power = add_drop_ring_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        input_power_coupling=0.05,
        drop_power_coupling=0.05,
        round_trip_power_loss=0.02,
    )

    assert np.max(drop_power) - np.min(drop_power) > 0.01

def test_add_drop_metric_extraction_finds_drop_peaks():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=40.0,
        num_points=1001,
    )

    through_power, drop_power = add_drop_ring_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        input_power_coupling=0.05,
        drop_power_coupling=0.05,
        round_trip_power_loss=0.02,
    )

    metrics = extract_add_drop_metrics(
        wavelengths_um=wavelengths_um,
        through_power=through_power,
        drop_power=drop_power,
    )

    assert metrics["num_drop_peaks_found"] >= 2
    assert metrics["max_drop_power"] > 0
    assert metrics["drop_insertion_loss_db"] >= 0
    assert metrics["through_extinction_ratio_db"] > 0
    assert metrics["mean_fsr_nm"] > 0

def test_add_drop_coupling_balance_sweep_runs():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)

    results = sweep_add_drop_coupling_balance(
        spec=spec,
        input_power_couplings=[0.02, 0.05],
        drop_power_couplings=[0.02, 0.05],
        round_trip_power_loss=0.02,
        span_nm=40.0,
        num_points=501,
    )

    assert len(results) == 4

    for row in results:
        assert row["max_drop_power"] >= 0
        assert row["drop_insertion_loss_db"] >= 0
        assert row["through_extinction_ratio_db"] >= 0
        assert row["mean_fsr_nm"] > 0

def test_all_pass_field_power_matches_power_function():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=20.0,
        num_points=501,
    )

    field = all_pass_ring_through_field(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.05,
        round_trip_power_loss=0.02,
    )

    power_from_field = np.abs(field) ** 2

    power = all_pass_ring_through_power(
        wavelengths_um=wavelengths_um,
        spec=spec,
        power_coupling=0.05,
        round_trip_power_loss=0.02,
    )

    assert np.allclose(power_from_field, power)


def test_cascaded_identical_rings_deepen_notch():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=20.0,
        num_points=501,
    )

    one_ring = cascade_all_pass_ring_power(
        wavelengths_um=wavelengths_um,
        specs=[spec],
        power_couplings=[0.05],
        round_trip_power_losses=[0.02],
    )

    three_rings = cascade_all_pass_ring_power(
        wavelengths_um=wavelengths_um,
        specs=[spec, spec, spec],
        power_couplings=[0.05, 0.05, 0.05],
        round_trip_power_losses=[0.02, 0.02, 0.02],
    )

    assert np.min(three_rings) < np.min(one_ring)

def test_cascade_metric_extraction_runs():
    spec = RingResonatorSpec(radius_um=10.0, group_index=4.0)
    wavelengths_um = wavelength_grid_around_center(
        center_wavelength_um=1.55,
        span_nm=20.0,
        num_points=501,
    )

    spectra = {
        "one_ring": cascade_all_pass_ring_power(
            wavelengths_um=wavelengths_um,
            specs=[spec],
            power_couplings=[0.05],
            round_trip_power_losses=[0.02],
        ),
        "two_rings": cascade_all_pass_ring_power(
            wavelengths_um=wavelengths_um,
            specs=[spec, spec],
            power_couplings=[0.05, 0.05],
            round_trip_power_losses=[0.02, 0.02],
        ),
    }

    metrics = extract_multiple_spectrum_metrics(
        wavelengths_um=wavelengths_um,
        spectra=spectra,
    )

    assert len(metrics) == 2

    for row in metrics:
        assert row["num_resonances_found"] >= 1
        assert row["extinction_ratio_db"] > 0
        assert row["linewidth_nm"] > 0
        assert row["loaded_q"] > 0

def test_ring_round_trip_length_cm_is_consistent():
    spec = RingResonatorSpec(radius_um=10.0)

    assert np.isclose(
        ring_round_trip_length_cm(spec),
        ring_round_trip_length_um(spec) / 10_000,
    )


def test_round_trip_power_loss_from_db_is_bounded():
    loss = round_trip_power_loss_from_loss_db(1.0)

    assert loss > 0
    assert loss < 1


def test_loss_budget_increases_with_bend_loss():
    spec = RingResonatorSpec(radius_um=8.0)

    no_bend = round_trip_power_loss_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=2.0,
        bend_loss_db_per_turn=0.0,
        coupler_excess_loss_db=0.0,
    )

    with_bend = round_trip_power_loss_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=2.0,
        bend_loss_db_per_turn=0.1,
        coupler_excess_loss_db=0.0,
    )

    assert with_bend > no_bend


def test_intrinsic_q_decreases_with_higher_loss_budget():
    spec = RingResonatorSpec(radius_um=8.0)

    low_loss_q = estimate_intrinsic_q_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=1.0,
        bend_loss_db_per_turn=0.0,
        coupler_excess_loss_db=0.0,
    )

    high_loss_q = estimate_intrinsic_q_from_loss_budget(
        spec=spec,
        propagation_loss_db_per_cm=10.0,
        bend_loss_db_per_turn=0.0,
        coupler_excess_loss_db=0.0,
    )

    assert high_loss_q < low_loss_q


def test_loss_budget_rejects_negative_values():
    spec = RingResonatorSpec(radius_um=8.0)

    with pytest.raises(ValueError):
        ring_round_trip_loss_db_from_loss_budget(
            spec=spec,
            propagation_loss_db_per_cm=-1.0,
        )

    with pytest.raises(ValueError):
        ring_round_trip_loss_db_from_loss_budget(
            spec=spec,
            propagation_loss_db_per_cm=1.0,
            bend_loss_db_per_turn=-0.1,
        )

    with pytest.raises(ValueError):
        ring_round_trip_loss_db_from_loss_budget(
            spec=spec,
            propagation_loss_db_per_cm=1.0,
            coupler_excess_loss_db=-0.1,
        )