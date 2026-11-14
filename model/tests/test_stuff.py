"""Checks for the Stuff+ target, aggregation, velocity rule, and scaling."""
import numpy as np
import pandas as pd
import pytest

from prism_model.stuff import add_xrv, aggregate, make_model, scale_within, to_plus


def pitches(n=120, family="Slider", velo=84.0, rv=0.0, call="BallCalled", pitcher="1"):
    return pd.DataFrame({
        "Season": 2025, "PitcherId": pitcher, "Pitcher": "Test", "PitcherThrows": "R",
        "shape_family": family, "subtype": "Gyro", "RelSpeed": velo, "Extension": 6.3,
        "InducedVertBreak": 2.0, "hb_arm": -5.0, "SpinRate": 2400.0, "RelHeight": 5.8,
        "rel_side_arm": 1.9, "fb_velo": 93.0, "fb_ivb": 16.0, "fb_hb": 8.0,
        "Balls": 1, "Strikes": 1, "PitchCall": call, "RunValue": rv, "xwOBA": np.nan,
    }, index=range(n))


def test_non_contact_pitches_keep_actual_run_value():
    df = add_xrv(pitches(rv=0.05))
    assert (df["xRV"] == 0.05).all()


def test_balls_in_play_use_expected_value_not_the_actual_result():
    # Same expected wOBA, different luck: one fell for a hit, one was caught.
    hit = pitches(n=1, call="InPlay", rv=0.4).assign(xwOBA=0.3)
    out = pitches(n=1, call="InPlay", rv=-0.2).assign(xwOBA=0.3)
    df = add_xrv(pd.concat([hit, out], ignore_index=True))
    assert df["xRV"].iloc[0] == pytest.approx(0.1)
    assert df["xRV"].iloc[1] == pytest.approx(0.1)


def test_aggregate_needs_minimum_pitches_and_reports_per_100():
    df = add_xrv(pd.concat([pitches(n=120, rv=-0.01), pitches(n=50, pitcher="2")], ignore_index=True))
    rows = aggregate(df)
    assert list(rows["PitcherId"]) == ["1"]
    assert rows["xrv100"].iloc[0] == pytest.approx(-1.0)
    assert rows["velo_diff"].iloc[0] == pytest.approx(-9.0)
    assert rows["group"].iloc[0] == "Breaking"


def test_harder_never_grades_worse():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"perceived_velo": rng.uniform(80, 98, n), "ivb": rng.normal(15, 3, n)})
    # Noisy target where velocity helps on average but not everywhere.
    y = -0.1 * (X["perceived_velo"] - 90) + rng.normal(0, 2, n) + 0.3 * np.sin(X["perceived_velo"])
    model = make_model(["perceived_velo", "ivb"]).fit(X, y)
    grid = pd.DataFrame({"perceived_velo": np.linspace(80, 98, 50), "ivb": 15.0})
    preds = model.predict(grid)
    assert np.all(np.diff(preds) <= 1e-9)


def test_plus_scale_direction_and_spread():
    xrv = pd.Series([-1.0, 0.0, 1.0])
    plus = to_plus(xrv, ref_mean=0.0, ref_sd=1.0)
    assert list(plus) == [110.0, 100.0, 90.0]


def test_scaling_is_within_each_pitch_type():
    rows = pd.DataFrame({"shape_family": ["Slider", "Slider", "Changeup", "Changeup"],
                         "stuff_xrv": [-1.0, 1.0, 5.0, 7.0]})
    plus = scale_within(rows, rows, "stuff_xrv")
    # Each pitch type is centered on its own average, whatever its raw level.
    assert plus.iloc[0] > 100 > plus.iloc[1]
    assert plus.iloc[2] > 100 > plus.iloc[3]
    assert plus.iloc[0] == pytest.approx(plus.iloc[2])


def test_smoothing_cloud_is_fixed_and_centered():
    from prism_model import stuff
    assert all(abs(off.mean()) < 1e-12 and len(off) == stuff.SMOOTH_DRAWS for off in stuff._OFFSETS.values())

    class Linear:  # a model that is already smooth: averaging over a centered cloud must change nothing
        def predict(self, X):
            return (2 * X["perceived_velo"] - X["ivb"] + 0.5 * X["velo_diff"]).to_numpy()

    X = pd.DataFrame({"perceived_velo": [90.0, 95.0], "ivb": [15.0, 10.0], "velo_diff": [-8.0, -10.0]})
    feats = ["perceived_velo", "ivb", "velo_diff"]
    assert np.allclose(stuff.smooth_predict(Linear(), X, feats), Linear().predict(X))
    assert np.allclose(stuff.smooth_predict(Linear(), X, feats), stuff.smooth_predict(Linear(), X.iloc[::-1], feats)[::-1])


def test_smoothing_softens_a_step():
    from prism_model import stuff

    class Step:  # a staircase like a tree split: 1 run better at 92 mph and above
        def predict(self, X):
            return np.where(X["perceived_velo"] >= 92, -1.0, 0.0)

    X = pd.DataFrame({"perceived_velo": [91.9, 92.1, 80.0, 100.0]})
    out = stuff.smooth_predict(Step(), X, ["perceived_velo"])
    assert abs(out[1] - out[0]) < 0.5 and out[1] < out[0]      # a 0.2 mph nudge no longer flips a whole run
    assert out[2] == 0.0 and out[3] == -1.0                    # far from the step nothing changes
