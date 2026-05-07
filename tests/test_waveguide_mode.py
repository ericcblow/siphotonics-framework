# tests/test_waveguide_mode.py

import numpy as np
import pytest

from src.simulation.waveguide_mode import (
    WaveguideCrossSection,
    estimate_effective_index_bounds,
    estimate_rectangular_waveguide_neff_eim,
    solve_symmetric_slab_te0,
    sweep_widths_eim,
)


def test_eim_neff_is_within_physical_bounds():
    xs = WaveguideCrossSection(width_um=0.5)

    n_min, n_max = estimate_effective_index_bounds(xs)
    vertical_neff, rectangular_neff = estimate_rectangular_waveguide_neff_eim(xs)

    assert n_min < vertical_neff < n_max
    assert n_min < rectangular_neff < n_max


def test_width_sweep_neff_increases_with_width():
    xs = WaveguideCrossSection(width_um=0.5)
    widths_um = np.array([0.35, 0.45, 0.55, 0.65])

    results = sweep_widths_eim(widths_um, xs)
    neffs = [row["rectangular_neff"] for row in results]

    assert all(neffs[i] < neffs[i + 1] for i in range(len(neffs) - 1))


def test_invalid_core_cladding_order_raises_error():
    with pytest.raises(ValueError):
        solve_symmetric_slab_te0(
            thickness_um=0.22,
            wavelength_um=1.55,
            n_core=1.44,
            n_clad=3.47,
        )