"""Checks for the TrackMan importer, using a small hand-built export."""
import pandas as pd
import pytest

from prism_data.schema import ALL_COLUMNS
from prism_data.trackman import read_trackman


def write_export(tmp_path, flip=False):
    """Two pitchers: a righty and a lefty whose hand is missing, fastballs and sliders."""
    rows = []
    for pid, hand, side, run in [("10", "Right", 1.5, 9.0), ("20", "Undefined", -2.2, -11.0)]:
        for i in range(10):
            rows.append(dict(PitchNo=i, Date="2/8/2023", PitcherId=pid, PitcherThrows=hand, BatterSide="Left",
                             Balls=0, Strikes=1, TaggedPitchType="Fastball", PitchCall="BallinDirt" if i == 0 else "StrikeCalled",
                             RelSpeed=90.0, SpinRate=2200, SpinAxis=212, InducedVertBreak=15.0,
                             HorzBreak=-run if flip else run, RelSide=-side if flip else side,
                             RelHeight=5.7, Extension=5.5, PlateLocSide=0.1, PlateLocHeight=2.5,
                             VertApprAngle=-5.3, HorzApprAngle=0.5, PitchMovementConfidence="High"))
        for i in range(5):
            rows.append(dict(rows[-1], PitchNo=20 + i, TaggedPitchType="Slider", RelSpeed=80.0,
                             HorzBreak=(run * -0.9) * (-1 if flip else 1), InducedVertBreak=1.0))
    rows.append(dict(rows[0], Date="1/4/1900", PitchMovementConfidence="Low"))
    path = tmp_path / "game.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_reads_into_prism_format(tmp_path):
    df = read_trackman(write_export(tmp_path))
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 31                      # the low-confidence pitch is kept...
    assert df["InducedVertBreak"].isna().sum() == 1   # ...with its movement blanked
    assert set(df["PitchCall"]) == {"BallCalled", "StrikeCalled"}
    assert df["Season"].dropna().unique().tolist() == [2023]


def test_missing_hand_is_inferred_and_arm_side_is_positive(tmp_path):
    df = read_trackman(write_export(tmp_path))
    lefty = df[df["PitcherId"] == "20"]
    assert (lefty["PitcherThrows"] == "L").all()
    fastballs = df[df["RelSpeed"] == 90.0]
    assert fastballs.groupby("PitcherId")["hb_arm"].median().tolist() == pytest.approx([9.0, 11.0])


def test_mirrored_file_is_corrected(tmp_path):
    with pytest.warns(UserWarning, match="mirrored"):
        df = read_trackman(write_export(tmp_path, flip=True))
    fastballs = df[(df["RelSpeed"] == 90.0) & (df["PitcherThrows"] == "R")]
    assert fastballs["hb_arm"].median() == pytest.approx(9.0)
