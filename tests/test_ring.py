# tests/test_ring.py

from src.compact_models.ring import (
    RingResonatorSpec,
    estimate_ring_fsr_nm,
    estimate_ring_fsr_um,
    ring_round_trip_length_um,
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
    