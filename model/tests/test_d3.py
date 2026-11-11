import numpy as np
import pandas as pd

from prism_model import d3


def fake_ref():
    return {"d3_scale_mph": np.linspace(74, 91, 99), "mlb_perceived_mph": np.linspace(88, 100, 99),
            "velo_weight": 2.0, "rates": {"Fastball": {"strike": 0.62, "whiff": 0.17}}}


def test_translate_scales_speed_spin_and_separation_together():
    rows = pd.DataFrame({"PitcherId": ["A", "A", "B"], "shape_family": ["Four-seam", "Slider", "Sinker"],
                         "n": [50, 20, 40], "perceived_velo": [82.5, 74.0, 87.0], "spin": [2050.0, 2200.0, 2100.0],
                         "velo_diff": [None, -8.5, None], "ivb": [16.0, 1.0, 8.0]})
    out = d3.translate(rows, fake_ref())
    assert (out.perceived_velo == rows.perceived_velo).all()  # displayed values stay as thrown
    a = out[out.PitcherId == "A"]
    assert a.velo_ratio.nunique() == 1 and a.velo_ratio.iloc[0] > 1 and a.velo_shift.iloc[0] > 5
    m = d3.model_inputs(out)
    ma = m[m.PitcherId == "A"]
    k = a.velo_ratio.iloc[0]
    assert abs(ma.perceived_velo.iloc[0] - 82.5 * k) < 1e-9
    assert abs(ma.spin.iloc[0] / ma.perceived_velo.iloc[0] - 2050 / 82.5) < 1e-9  # spin per mph unchanged
    assert abs(ma.perceived_velo.iloc[1] / ma.perceived_velo.iloc[0] - 74 / 82.5) < 1e-9  # same % off the fastball
    assert abs(ma.velo_diff.iloc[1] - (-8.5 * k)) < 1e-9
    assert (m.ivb == rows.ivb).all()  # movement as thrown
    b = out[out.PitcherId == "B"].iloc[0]
    assert b.fb_pct_d3 > a.fb_pct_d3.iloc[0]


def test_d3_rate_maps_pitch_types():
    assert d3.d3_rate("Sinker", "whiff", fake_ref()) == 0.17
    assert np.isnan(d3.d3_rate("Slider", "whiff", fake_ref()))
