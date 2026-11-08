"""Checks for the fastball reference and shape features."""
import numpy as np
import pandas as pd
import pytest

from prism_model.shapes import BUCKET, FAMILIES, MIN_PITCHES, add_fastball_reference, shape_features, subtype_for


def session(pitcher="1", throws="R", fb_velo=84.0, n_fb=20, n_sl=10, n_ch=6):
    """A synthetic outing: fastballs, sliders, and changeups with fixed shapes."""
    rows = []
    for _ in range(n_fb):
        rows.append(dict(RelSpeed=fb_velo, InducedVertBreak=16.0, hb_arm=8.0, SpinAxis=210.0, SpinRate=2200.0))
    for _ in range(n_sl):
        rows.append(dict(RelSpeed=fb_velo - 10, InducedVertBreak=1.0, hb_arm=-6.0, SpinAxis=60.0, SpinRate=2400.0))
    for _ in range(n_ch):
        rows.append(dict(RelSpeed=fb_velo - 8, InducedVertBreak=6.0, hb_arm=14.0, SpinAxis=240.0, SpinRate=1650.0))
    df = pd.DataFrame(rows)
    df["PitcherId"], df["Season"], df["PitcherThrows"] = pitcher, 2026, throws
    return df


def test_fastball_reference_uses_hardest_pitches_only():
    df = add_fastball_reference(session())
    assert df["fb_velo"].iloc[0] == pytest.approx(84.0)
    assert df["fb_ivb"].iloc[0] == pytest.approx(16.0)
    assert df["fb_hb"].iloc[0] == pytest.approx(8.0)


def test_features_are_relative_to_the_fastball():
    feats = shape_features(add_fastball_reference(session()))
    slider = feats.iloc[25]
    assert slider["velo_diff"] == pytest.approx(-10.0)
    assert slider["ivb_diff"] == pytest.approx(-15.0)
    assert slider["hb_diff"] == pytest.approx(-14.0)
    assert feats.iloc[30]["spin_ratio"] == pytest.approx(1650 / 2200)


def test_same_shape_at_different_speeds_gives_same_features():
    """A D3 arm at 82 and an MLB arm at 95 with the same shapes look identical, by design."""
    slow = shape_features(add_fastball_reference(session(fb_velo=82.0)))
    fast = shape_features(add_fastball_reference(session(fb_velo=95.0)))
    pd.testing.assert_frame_equal(slow, fast)


def test_lefty_spin_axis_is_mirrored():
    righty = session(throws="R")
    lefty = session(throws="L")
    lefty["SpinAxis"] = 360 - lefty["SpinAxis"]
    a = shape_features(add_fastball_reference(righty))
    b = shape_features(add_fastball_reference(lefty))
    np.testing.assert_allclose(a["axis_sin"], b["axis_sin"], atol=1e-9)
    np.testing.assert_allclose(a["axis_cos"], b["axis_cos"], atol=1e-9)


def test_too_few_pitches_gets_no_reference():
    small = session(n_fb=MIN_PITCHES - 10, n_sl=2, n_ch=0)
    assert add_fastball_reference(small)["fb_velo"].isna().all()


def test_every_pitch_type_has_a_bucket_and_cutters_are_fastballs():
    assert set(FAMILIES) | {"Knuckleball"} == set(BUCKET)
    assert BUCKET["Cutter"] == "Fastball"
    assert BUCKET["Splitter"] == "Offspeed"


@pytest.mark.parametrize("pitch_type, shape, expected", [
    ("Four-seam", dict(ivb=19, glove=-8, hb_diff=0, spin_ratio=1), "Ride"),
    ("Four-seam", dict(ivb=15, glove=-8, hb_diff=0, spin_ratio=1), "Ride"),
    ("Four-seam", dict(ivb=12, glove=-8, hb_diff=0, spin_ratio=1), "Flat"),
    ("Sinker", dict(ivb=6, glove=-16, hb_diff=8, spin_ratio=1), None),
    ("Cutter", dict(ivb=9, glove=2, hb_diff=-10, spin_ratio=1), None),
    ("Slider", dict(ivb=1, glove=14, hb_diff=-20, spin_ratio=1.1), "Sweeper"),
    ("Slider", dict(ivb=2, glove=3, hb_diff=-10, spin_ratio=1.1), "Gyro"),
    ("Slider", dict(ivb=0, glove=7, hb_diff=-14, spin_ratio=1.1), "Traditional"),
    ("Curveball", dict(ivb=-14, glove=2, hb_diff=-10, spin_ratio=1.1), "12-6"),
    ("Curveball", dict(ivb=-8, glove=12, hb_diff=-20, spin_ratio=1.1), "Slurve"),
    ("Curveball", dict(ivb=-14, glove=7, hb_diff=-16, spin_ratio=1.1), "Standard"),
    ("Changeup", dict(ivb=6, glove=-15, hb_diff=6, spin_ratio=0.8), "Fade"),
    ("Changeup", dict(ivb=3, glove=-10, hb_diff=1, spin_ratio=0.6), "Drop"),
    ("Splitter", dict(ivb=3, glove=-10, hb_diff=1, spin_ratio=0.5), None),
])
def test_subtype_rules(pitch_type, shape, expected):
    assert subtype_for(pitch_type, shape) == expected
