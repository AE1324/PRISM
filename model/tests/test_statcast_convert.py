"""Checks the Statcast conversion on hand-built pitches with known answers."""
import math

import pandas as pd
import pytest

from prism_data.schema import ALL_COLUMNS
from prism_data.statcast import convert


def pitch(**overrides):
    base = {
        "game_type": "R", "game_pk": 1, "game_date": "2025-06-01", "game_year": 2025,
        "at_bat_number": 3, "pitch_number": 2, "player_name": "Test, Pitcher",
        "pitcher": 10, "batter": 20, "p_throws": "R", "stand": "R", "balls": 1, "strikes": 2,
        "pitch_type": "FF", "release_speed": 94.0, "release_spin_rate": 2300, "spin_axis": 210,
        "release_pos_z": 6.0, "release_pos_x": -1.8, "release_extension": 6.5,
        "pfx_z": 1.3, "pfx_x": -0.7, "plate_z": 2.8, "plate_x": 0.2,
        # A typical four-seamer's release velocity and acceleration (ft/s, ft/s^2).
        "vx0": 7.0, "vy0": -137.0, "vz0": -5.0, "ax": -12.0, "ay": 30.0, "az": -15.0,
        "description": "swinging_strike", "events": "strikeout",
        "launch_speed": None, "launch_angle": None,
        "delta_run_exp": -0.2, "estimated_woba_using_speedangle": None,
    }
    base.update(overrides)
    return base


def test_columns_match_schema():
    out = convert(pd.DataFrame([pitch()]))
    assert list(out.columns) == ALL_COLUMNS


def test_units_and_outcomes():
    row = convert(pd.DataFrame([pitch()])).iloc[0]
    assert row["InducedVertBreak"] == pytest.approx(15.6)   # 1.3 ft -> inches
    assert row["PitchCall"] == "StrikeSwinging"
    assert row["KorBB"] == "Strikeout"
    assert row["PitchNo"] == 302


def test_arm_side_is_positive_for_both_hands():
    """A righty's run (toward 3B) and a lefty's mirror image (toward 1B) both read as arm side."""
    righty = convert(pd.DataFrame([pitch(p_throws="R", pfx_x=-0.7, release_pos_x=-1.8)])).iloc[0]
    lefty = convert(pd.DataFrame([pitch(p_throws="L", pfx_x=0.7, release_pos_x=1.8)])).iloc[0]
    assert righty["hb_arm"] == pytest.approx(8.4)
    assert lefty["hb_arm"] == pytest.approx(8.4)
    assert righty["rel_side_arm"] == pytest.approx(1.8)
    assert lefty["rel_side_arm"] == pytest.approx(1.8)


def test_vertical_approach_angle_is_downward_and_plausible():
    row = convert(pd.DataFrame([pitch()])).iloc[0]
    assert -9 < row["VertApprAngle"] < -3
    assert not math.isnan(row["HorzApprAngle"])


def test_in_play_result_and_non_regular_season_dropped():
    df = pd.DataFrame([
        pitch(description="hit_into_play", events="double", launch_speed=101.0, launch_angle=18.0,
              estimated_woba_using_speedangle=0.9),
        pitch(game_type="S"),  # spring training is dropped
    ])
    out = convert(df)
    assert len(out) == 1
    assert out.iloc[0]["PlayResult"] == "Double"
    assert out.iloc[0]["ExitSpeed"] == 101.0
