import numpy as np
import pandas as pd

from prism_data.clean import apply_game_movement, check_game_movement, load_decisions
from prism_model.team_data import TYPE_SCALE, type_without_movement


def _games(shift_game=None, shift=0.0, missing_game=None):
    """Six pitchers x four games of fastballs and sliders; one game can read low or lose movement."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(4):
        game = f"G{g}"
        for p in range(6):
            pid = f"P{p}" if p < 3 else f"P{p}_{g}"  # three regulars, three one-game opponents
            for i in range(12):
                fb = i < 8
                ivb = (18 + p % 3 if fb else 1) + rng.normal(0, 0.5)
                if game == shift_game:
                    ivb += shift
                rows.append({"GameID": game, "PitcherId": pid, "RelSpeed": (90 if fb else 82) + rng.normal(0, 0.5),
                             "SpinRate": (2300 if fb else 2500) + rng.normal(0, 30),
                             "InducedVertBreak": np.nan if game == missing_game else ivb,
                             "hb_arm": (12 if fb else -6) + rng.normal(0, 0.5), "HorzBreak": 1.0,
                             "issues": [], "notes": []})
    return pd.DataFrame(rows)


def test_normal_games_are_trusted():
    g = check_game_movement(_games())
    assert g["trusted"].all()


def test_game_reading_low_for_everyone_is_flagged():
    g = check_game_movement(_games(shift_game="G2", shift=-5)).set_index("GameID")
    assert not g.at["G2", "trusted"] and "ride -5" in g.at["G2", "reason"]
    assert g.drop(index="G2")["trusted"].all()


def test_game_missing_movement_is_flagged_and_blanked():
    df = _games(missing_game="G1")
    games = check_game_movement(df)
    assert not games.set_index("GameID").at["G1", "trusted"]
    out = apply_game_movement(df.assign(InducedVertBreak=df["InducedVertBreak"].fillna(10)), games)
    g1 = out[out["GameID"] == "G1"]
    assert g1["InducedVertBreak"].isna().all() and g1["hb_arm"].isna().all()
    assert all("movement not used" in n[-1] for n in g1["notes"])
    assert out[out["GameID"] != "G1"]["InducedVertBreak"].notna().all()


def test_review_call_overrules_the_game_check():
    df = _games(shift_game="G2", shift=-5)
    out = apply_game_movement(df, check_game_movement(df), {"game:G2": "trust"})
    assert out["movement_trusted"].all() and out["InducedVertBreak"].notna().all()


def test_game_calls_load_from_csv(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"PitchUID": ["a", "game:G2"], "decision": ["drop", "Distrust"]}).to_csv(p, index=False)
    assert load_decisions(p) == {"a": "drop", "game:G2": "distrust"}


def test_pitches_without_movement_typed_from_speed_and_spin():
    known = pd.DataFrame({"PitcherId": "P", "RelSpeed": [90.0] * 6 + [82.0] * 6, "SpinRate": [2300.0] * 6 + [2500.0] * 6,
                          "shape_family": ["Four-seam"] * 6 + ["Slider"] * 6,
                          "subtype": ["Ride"] * 6 + ["Sweeper"] * 6, "bucket": ["Fastball"] * 6 + ["Breaking"] * 6})
    new = pd.DataFrame({"PitcherId": "P", "RelSpeed": [89.5, 82.4, 70.0], "SpinRate": [2280.0, 2530.0, 1500.0],
                        "shape_family": None, "subtype": None, "bucket": None})
    df = pd.concat([known, new], ignore_index=True)
    df["issues"] = [[] for _ in range(len(df))]
    df["notes"] = [[] for _ in range(len(df))]
    out = type_without_movement(df).tail(3)
    assert list(out["shape_family"].iloc[:2]) == ["Four-seam", "Slider"]
    assert out["subtype"].iloc[1] == "Sweeper" and "speed and spin" in out["notes"].iloc[0][0]
    assert pd.isna(out["shape_family"].iloc[2])  # too far from anything he throws
    assert set(TYPE_SCALE) == {"RelSpeed", "SpinRate"}


def test_lone_velocity_misread_is_flagged_but_a_hard_day_is_not():
    from prism_data.clean import flag_shapes
    speeds = [79.0, 79.5, 80.0, 79.2, 80.4, 79.8, 81.0, 78.6, 85.2]          # one lone 85.2
    hard = [79.0, 79.5, 80.0, 79.2, 80.4, 79.8, 83.6, 84.1, 84.4, 84.0]     # a whole hard day, not alone
    for v, expect in ((speeds, 1), (hard, 0)):
        df = pd.DataFrame({"PitcherId": "P", "RelSpeed": v, "InducedVertBreak": 20.0, "hb_arm": 12.0,
                           "shape_family": "Four-seam"})
        df["issues"] = [[] for _ in range(len(df))]
        df["notes"] = [[] for _ in range(len(df))]
        out = flag_shapes(df)
        hits = out["issues"].map(lambda f: any("velocity far off" in x for x in f))
        assert hits.sum() == expect
