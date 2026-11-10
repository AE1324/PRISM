import numpy as np
import pandas as pd

from prism_model import fastball_traits as ft


def test_physics_lower_release_is_flatter():
    hi = ft.vaa_physics(90, 6.2, 6.3, 3.0, 15)
    lo = ft.vaa_physics(90, 5.0, 6.3, 3.0, 15)
    assert lo > hi  # less negative = flatter


def test_physics_more_carry_and_velo_is_flatter():
    assert ft.vaa_physics(90, 5.8, 6.3, 3.0, 18) > ft.vaa_physics(90, 5.8, 6.3, 3.0, 10)
    assert ft.vaa_physics(95, 5.8, 6.3, 3.0, 15) > ft.vaa_physics(82, 5.8, 6.3, 3.0, 15)


def test_ordinal():
    assert [ft.ordinal(n) for n in (1, 2, 3, 11, 12, 13, 21, 42, 100)] == \
        ["1st", "2nd", "3rd", "11th", "12th", "13th", "21st", "42nd", "100th"]


def test_height_parse():
    assert abs(ft.height_ft('6\'1"') - (6 + 1 / 12)) < 1e-9
    assert ft.height_ft(None) is None


def _pitcher(ext, rel, ivb, n=40, velo=84.0, plate=3.0):
    rng = np.random.default_rng(0)
    vaa = ft.vaa_physics(velo, rel, ext, plate, ivb)
    return pd.DataFrame({
        "shape_family": "Four-seam", "RelSpeed": velo + rng.normal(0, 1, n), "RelHeight": rel,
        "Extension": ext, "PlateLocHeight": plate, "InducedVertBreak": ivb, "VertApprAngle": vaa,
    })


def test_traits_flags_extension_and_flat_angle():
    model = {"vaa_ae_pct": np.linspace(-1.2, 1.2, 99), "ext_pct": np.linspace(5.5, 7.4, 99)}
    long_low = ft.traits(_pitcher(7.2, 4.6, 16), model, (0.0, 0.0))
    assert any("Extension is a weapon" in s for s in long_low["strengths"])
    assert any("Flat approach angle" in s for s in long_low["strengths"])
    short = ft.traits(_pitcher(5.6, 6.3, 12), model, (0.0, 0.0), height=6.25)
    assert any("Extension is costing velocity" in s for s in short["limits"])
    assert short["fixes"]
