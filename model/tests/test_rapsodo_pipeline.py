"""Checks for the Rapsodo importer, the AWRE matcher, and the cleaning flags."""
import numpy as np
import pandas as pd
import pytest

from prism_data.clean import apply_decisions, flag_basic, flag_shapes
from prism_data.match import align, match, merge
from prism_data.rapsodo import clock_to_axis, correct_release, fill_movement, prepare, read_rapsodo

HEADER = ('"No","Date","Pitch ID","Pitch Type","Is Strike","Strike Zone Side","Strike Zone Height","Velocity",'
          '"Total Spin","True Spin (release)","Spin Efficiency (release)","Spin Direction","Spin Confidence",'
          '"VB (trajectory)","HB (trajectory)","SSW VB","SSW HB","VB (spin)","HB (spin)","Horizontal Angle",'
          '"Release Angle","Release Height","Release Side","Gyro Degree (deg)","Unique ID","Device Serial Number",'
          '"Horizontal Approach Angle","Vertical Approach Angle","Session Name","Intent Type","Release Extension (ft)"')


def row(no, when, velo, vb, hb, side=1.8, spin=2200, uid=None, vbs=None, hbs=None, tag="Fastball"):
    traj_vb = "-" if vb is None else vb
    traj_hb = "-" if hb is None else hb
    return (f'{no},"{when}",{no},"{tag}","Y",0,30,{velo},{spin},2000,90,"01:30",1,{traj_vb},{traj_hb},"-","-",'
            f'{vbs if vbs is not None else (vb or 10) - 4},{hbs if hbs is not None else (hb or 10)},0,0,5.8,{side},10,'
            f'"1@{uid or no}","X",-1,-6,"untitled","-",6.3')


def write(tmp_path, rows, name="Test Pitcher"):
    path = tmp_path / f"{name.replace(' ', '_')}.csv"
    path.write_text(f'\n"Player ID:",1\n"Player Name:",{name}\n\n{HEADER}\n' + "\n".join(rows) + "\n")
    return path


def test_clock_to_axis_matches_trackman():
    assert clock_to_axis("12:00") == 180
    assert clock_to_axis("01:00") == 210
    assert clock_to_axis("11:00") == 150
    assert np.isnan(clock_to_axis(None))


def test_reads_export_and_converts_units(tmp_path):
    df = read_rapsodo(write(tmp_path, [row(1, "Thu Sep 17 2026 7:11:37 PM", 84.2, 15.0, 12.0)]))
    r = df.iloc[0]
    assert r["Pitcher"] == "Test Pitcher" and r["RelSpeed"] == 84.2
    assert r["PlateLocHeight"] == pytest.approx(2.5)        # 30 in
    assert str(r["PitchTime"]) == "2026-09-17 19:11:37"
    assert r["SpinAxis"] == 225


def test_spin_only_movement_uses_the_pitchers_own_gap(tmp_path):
    rows = [row(i, f"Thu Sep 17 2026 7:{10 + i:02d}:00 PM", 84, 16.0, 12.0, vbs=11.0, hbs=12.0) for i in range(6)]
    rows.append(row(9, "Thu Sep 17 2026 7:30:00 PM", 84, None, None, vbs=10.0, hbs=13.0))
    df = prepare(read_rapsodo(write(tmp_path, rows)))
    last = df.sort_values("PitchTime").iloc[-1]
    assert last["movement_source"] == "spin-adjusted"
    assert last["InducedVertBreak"] == pytest.approx(15.0)     # spin 10 + his usual +5 gap


def test_moving_the_unit_between_sessions_is_undone():
    df = pd.DataFrame({
        "PitcherId": ["A"] * 6, "stint": [1, 1, 1, 2, 2, 2],
        "RelSide": [1.5, 1.6, 1.4, 3.5, 3.6, 3.4], "RelHeight": [5.8] * 6,
    })
    out = correct_release(df)
    assert out.groupby("stint")["RelSide"].median().round(2).tolist() == [2.5, 2.5]
    assert out["RelSide"].diff().iloc[1] == pytest.approx(0.1)  # pitch-to-pitch variation kept


def test_align_respects_order_velocity_and_skips_extras():
    a_t = np.array([100, 130, 160.0])
    a_v = np.array([85, 72, np.nan])
    r_t = np.array([95, 110, 125, 152.0])       # 110 is a warm-up Rapsodo saw but AWRE didn't log
    r_v = np.array([85.2, 60.0, 72.4, 84.0])
    assert align(a_t, a_v, r_t, r_v, typical=5) == [(0, 0), (1, 2), (2, 3)]


def _pitch_frame(n=30, **over):
    base = dict(PitcherId="A", PitcherThrows="R", RelSpeed=84.0, SpinRate=2200.0, InducedVertBreak=16.0,
                HorzBreak=10.0, hb_arm=10.0, RelHeight=5.8, RelSide=1.8, rel_side_arm=1.8, Extension=6.3,
                PitchUID=None, movement_source="trajectory", vb_spin=12.0, hb_spin=10.0, stint=1)
    df = pd.DataFrame([dict(base, PitchUID=str(i)) for i in range(n)])
    for k, v in over.items():
        idx, val = v
        df.loc[idx, k] = val
    return df


def test_basic_flags():
    df = _pitch_frame(RelSpeed=(0, np.nan), RelSide=(1, -2.0), rel_side_arm=(1, -2.0), RelHeight=(2, 7.4),
                      SpinRate=(3, 300.0))
    out = flag_basic(df)
    assert "no velocity read" in out.at[0, "issues"]
    assert any("wrong side" in s for s in out.at[1, "issues"])
    assert any("release point" in s for s in out.at[2, "issues"])
    assert any("impossible spin" in s for s in out.at[3, "issues"])
    assert out.loc[4:, "issues"].map(len).sum() == 0


def test_shape_outlier_and_decisions():
    df = flag_basic(_pitch_frame(InducedVertBreak=(5, 2.0)))
    df["shape_family"] = "Four-seam"
    out = flag_shapes(df)
    assert any("unlike his other four-seams" in s for s in out.at[5, "issues"])
    decided = apply_decisions(out, {"5": "keep", "7": "drop"})
    assert not decided.at[5, "exclude"] and decided.at[7, "exclude"]
    assert decided["exclude"].sum() == 1


def test_match_and_merge_attach_results():
    t0 = pd.Timestamp("2026-09-17 19:00:00")
    rap = pd.DataFrame({"PitcherId": ["Colin D'Onofrio"] * 3, "Date": [t0.date()] * 3,
                        "PitchTime": [t0, t0 + pd.Timedelta(seconds=40), t0 + pd.Timedelta(seconds=80)],
                        "RelSpeed": [82.4, 60.0, 73.1]})
    awre = pd.DataFrame({"PitcherId": ["Colin D’Onofrio"] * 2, "Date": [t0.date()] * 2,
                         "PitchTime": [t0 - pd.Timedelta(seconds=6), t0 + pd.Timedelta(seconds=73)],
                         "RelSpeed": [82.0, 73.0], "Balls": [0, 0], "Strikes": [0, 1],
                         "PitchCall": ["StrikeCalled", "InPlay"], "PlayResult": ["Undefined", "Single"],
                         "KorBB": ["Undefined"] * 2, "TaggedPitchType": ["Fastball", "Breaking"],
                         "Batter": ["X", "Y"], "BatterId": ["X", "Y"], "BatterSide": [None, None],
                         "GameID": ["g", "g"], "PitchNo": [1, 2]})
    pairs = match(awre, rap)
    assert set(pairs["confidence"]) == {"strong", "rapsodo-only"}
    out = merge(awre, rap, pairs)
    assert out["PitchCall"].tolist()[0] == "StrikeCalled" and pd.isna(out["PitchCall"].iloc[1])
    assert out["PlayResult"].iloc[2] == "Single"


def test_one_off_shapes_join_a_close_pitch_or_get_dropped():
    df = flag_basic(_pitch_frame(n=40))
    df["shape_family"], df["subtype"], df["bucket"] = "Four-seam", "Ride", "Fastball"
    df.loc[30:39, ["shape_family", "subtype", "bucket"]] = ["Slider", "Gyro", "Breaking"]
    df.loc[30:39, ["RelSpeed", "InducedVertBreak", "hb_arm"]] = [76.0, 1.0, -4.0]
    df.loc[0, ["shape_family", "RelSpeed", "InducedVertBreak", "hb_arm"]] = ["Cutter", 83.0, 14.0, 8.0]   # basically his four-seam
    df.loc[1, ["shape_family", "RelSpeed", "InducedVertBreak", "hb_arm"]] = ["Changeup", 70.0, 25.0, 25.0]  # matches nothing
    out = flag_shapes(df)
    assert out.at[0, "shape_family"] == "Four-seam" and out.at[0, "issues"] == []
    assert any("moved to his four-seam" in s for s in out.at[0, "notes"])
    assert any("matches none of his pitches" in s for s in out.at[1, "issues"])
