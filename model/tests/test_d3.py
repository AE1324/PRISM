import numpy as np
import pandas as pd

from prism_model import d3


def fake_ref():
    return {"d3_scale_mph": np.linspace(74, 91, 99), "mlb_perceived_mph": np.linspace(88, 100, 99),
            "velo_weight": 2.0, "rates": {"Fastball": {"strike": 0.62, "whiff": 0.17}}}


def test_translate_moves_all_pitches_together():
    rows = pd.DataFrame({"PitcherId": ["A", "A", "B"], "shape_family": ["Four-seam", "Slider", "Sinker"],
                         "n": [50, 20, 40], "perceived_velo": [82.5, 74.0, 87.0]})
    out = d3.translate(rows, fake_ref())
    a = out[out.PitcherId == "A"]
    assert a.velo_shift.nunique() == 1 and a.velo_shift.iloc[0] > 5
    assert abs((a.perceived_velo.iloc[0] - a.perceived_velo.iloc[1]) - 8.5) < 1e-9  # gap preserved
    b = out[out.PitcherId == "B"].iloc[0]
    assert b.fb_pct_d3 > a.fb_pct_d3.iloc[0] and b.perceived_velo > a.perceived_velo.iloc[0]


def test_d3_rate_maps_pitch_types():
    assert d3.d3_rate("Sinker", "whiff", fake_ref()) == 0.17
    assert np.isnan(d3.d3_rate("Slider", "whiff", fake_ref()))
