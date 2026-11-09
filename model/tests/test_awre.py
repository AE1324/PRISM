"""Checks for the AWRE importer, on a small hand-built export."""
import pandas as pd
import pytest

from prism_data.awre import read_awre
from prism_data.schema import ALL_COLUMNS


def test_reads_awre_export(tmp_path):
    pytest.importorskip("openpyxl")
    raw = pd.DataFrame({
        "PitchNo": [1, 2, 3, 4], "Date": ["9/27/26"] * 4, "Time": ["1:09:42 PM", "1:09:51 PM", "1:10:20 PM", "1:10:32 PM"],
        "PAofInning": [1, 1, 1, 2], "PAofInning.1": [1, 2, 3, 1], "Pitcher": ["Shane McKenna"] * 4,
        "PitcherTeam": ["Oberlin College"] * 4, "Batter": ["A", "A", "A", "B"], "BatterTeam": ["Oberlin College"] * 4,
        "Inning": [1] * 4, "Top/Bottom": ["Top"] * 4, "Outs": [0] * 4, "Balls": [0, 1, 1, 0], "Strikes": [0, 0, 1, 0],
        "PitchType": ["Fast Ball", "Not Tracked", "Breaking Ball", "Fast Ball"],
        "PitchResult": ["Ball", "Strike Looking", "Strike In Play", "Ball"],
        "AtBatResult": [None, None, "1B", "HBP"], "PitchVelo": [85, 0, 71, 84], "PitchId": [None] * 4,
    })
    raw = raw.rename(columns={"PAofInning.1": "PAofInning"})
    path = tmp_path / "PitchData-927.xlsx"
    raw.to_excel(path, index=False)
    df = read_awre(path)
    assert list(df.columns) == ALL_COLUMNS
    assert df["PitchCall"].tolist() == ["BallCalled", "StrikeCalled", "InPlay", "HitByPitch"]
    assert pd.isna(df["RelSpeed"].iloc[1]) and df["RelSpeed"].iloc[0] == 85
    assert df["PlayResult"].tolist() == ["Undefined", "Undefined", "Single", "Undefined"]
    assert pd.isna(df["TaggedPitchType"].iloc[1])
    assert str(df["PitchTime"].iloc[0]) == "2026-09-27 13:09:42"
