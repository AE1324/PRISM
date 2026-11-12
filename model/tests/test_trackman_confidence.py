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


def test_ids_match_across_number_and_text_exports():
    from prism_data.trackman import clean_id
    assert clean_id(pd.Series([1000139061.0, "1000139061", " 1000139061.0 "])).nunique() == 1


def test_wrong_side_pitch_names_the_pitcher_whose_release_matches():
    from prism_data.clean import attribute_wrong_pitcher
    rows = [dict(GameID="g", PitcherId="R1", Pitcher="Righty", PitcherThrows="R", RelSide=1.5, RelHeight=6.0, issues=[])
            for _ in range(6)]
    rows += [dict(GameID="g", PitcherId="L1", Pitcher="Lefty", PitcherThrows="L", RelSide=-1.3, RelHeight=5.4, issues=[])
             for _ in range(6)]
    rows.append(dict(GameID="g", PitcherId="R1", Pitcher="Righty", PitcherThrows="R", RelSide=-1.25, RelHeight=5.45,
                     issues=["released from the wrong side for a RHP (-1.2 ft)"]))
    out = attribute_wrong_pitcher(pd.DataFrame(rows))
    assert any("likely thrown by Lefty" in x for x in out.iloc[-1]["issues"])


def test_practice_export_approach_angles_are_rescaled(tmp_path):
    import pandas as pd
    from prism_data.trackman import read_trackman
    p = tmp_path / "practice.csv"
    pd.DataFrame({"PitchNo": [1, 2], "Date": "2026-07-31", "Pitcher": "Black, Grayson", "PitcherId": 117036,
                  "PitcherThrows": "Right", "RelSpeed": [88.6, 77.9], "VertApprAngle": [-198.29, -313.6],
                  "HorzApprAngle": [19.13, -223.26], "HorzBreak": [16.2, -17.7], "RelSide": [1.5, 2.2],
                  "InducedVertBreak": [20.9, 2.8]}).to_csv(p, index=False)
    df = read_trackman(p)
    assert abs(df["VertApprAngle"].iloc[0] + 5.04) < 0.01 and abs(df["HorzApprAngle"].iloc[1] + 5.67) < 0.01
