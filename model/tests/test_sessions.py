import pandas as pd
import pytest

from prism_data.sessions import graded, load_sessions, tag_sessions


def _pitches():
    return pd.DataFrame({
        "Source": ["rapsodo"] * 4 + ["trackman"],
        "Date": ["2026-09-11", "2026-09-11", "2026-09-12", "2026-09-11", "2026-09-11"],
        "PitcherId": ["Grayson Black", "Shido Doe", "Grayson Black", "Grayson Black", "1000"],
        "Pitcher": ["Grayson Black", "Shido Doe", "Grayson Black", "Grayson Black", "Gearhart, Warner"],
    })


def _file(tmp_path, rows):
    p = tmp_path / "sessions.csv"
    pd.DataFrame(rows, columns=["Date", "Device", "Setting", "Pitcher"]).to_csv(p, index=False)
    return load_sessions(p)


def test_untagged_rapsodo_is_bullpen_and_trackman_is_game():
    out = tag_sessions(_pitches())
    assert list(out["Setting"]) == ["bullpen"] * 4 + ["game"]
    assert list(graded(out)["Device"]) == ["trackman"]


def test_day_tag_covers_everyone_and_pitcher_row_wins(tmp_path):
    s = _file(tmp_path, [["2026-09-11", "Rapsodo", "Live", ""],
                         ["9/11/2026", "rapsodo", "bullpen", "Black, Grayson"],
                         ["2026-09-11", "trackman", "live", ""]])
    out = tag_sessions(_pitches(), s)
    assert list(out["Setting"]) == ["bullpen", "live", "bullpen", "bullpen", "live"]


def test_tag_alone_decides_not_the_data(tmp_path):
    p = _pitches().assign(RelSpeed=[60.0] * 5)  # slow bullpen-looking pitches still count if tagged live
    out = tag_sessions(p, _file(tmp_path, [["2026-09-12", "rapsodo", "live", ""]]))
    assert out.loc[2, "Setting"] == "live" and len(graded(out)) == 2


def test_bad_rows_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="setting one of"):
        _file(tmp_path, [["2026-09-11", "rapsodo", "scrimmage", ""]])


def test_missing_file_means_no_tags(tmp_path):
    assert load_sessions(tmp_path / "nope.csv").empty
