import numpy as np
import pandas as pd

from prism_data.trackman import read_trackman


def test_low_confidence_blanks_movement_but_keeps_the_pitch(tmp_path):
    rows = pd.DataFrame({
        "PitcherId": ["1", "1", "1"], "Pitcher": ["A"] * 3, "PitcherThrows": ["Right"] * 3, "Date": ["2026-06-16"] * 3,
        "RelSpeed": [90.0, 89.0, 80.0], "SpinRate": [2300, 2280, 2400], "InducedVertBreak": [18.0, 50.0, 2.0],
        "HorzBreak": [10.0, 9.0, -6.0], "RelSide": [1.5, 1.5, 1.6], "RelHeight": [6.0, 6.0, 6.0],
        "PlateLocSide": [0.1, -0.2, 0.5], "PlateLocHeight": [2.5, 2.4, 1.9], "PitchCall": ["InPlay", "StrikeCalled", "BallCalled"],
        "PitchMovementConfidence": ["High", "Low", "High"],
    })
    f = tmp_path / "game.xlsx"
    rows.to_excel(f, index=False)
    out = read_trackman(f)
    assert len(out) == 3                                  # nothing dropped
    assert np.isnan(out.loc[1, "InducedVertBreak"])       # low-confidence movement blanked
    assert out.loc[1, "PitchCall"] == "StrikeCalled"      # result kept
    assert out.loc[0, "hb_arm"] > 0                       # RHP arm-side run positive (confirmed convention)
