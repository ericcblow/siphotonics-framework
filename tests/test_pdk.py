# tests/test_pdk.py

from src.pdk.layers import WG
from src.pdk.materials import SI_N_1550, SIO2_N_1550, THICKNESS_SI_UM


def test_indices_make_physical_sense():
    assert SI_N_1550 > SIO2_N_1550
    assert SIO2_N_1550 > 1.0


def test_soi_thickness():
    assert THICKNESS_SI_UM == 0.22


def test_waveguide_layer():
    assert WG == (1, 0)