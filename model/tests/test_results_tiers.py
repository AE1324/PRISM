import numpy as np
import pandas as pd

from prism_model import results_tiers as RT

TIERS = {
    "mlb_ev_quantiles": list(np.linspace(60, 115, 101)),
    "contact_cutoffs": {"GroundBall": (82.0, 97.0), "LineDrive": (90.0, 101.0), "FlyBall": (87.0, 97.0), "Popup": (79.0, 88.0)},
    "cells": {(t, c): v for t, base in (("GroundBall", -0.1), ("LineDrive", 0.2), ("FlyBall", 0.0), ("Popup", -0.27))
              for c, v in (("weak", base - 0.05), ("average", base), ("hard", base + 0.3))},
}
TABLE = {"by_call": {}, "call_only": {"BallCalled": 0.05, "StrikeCalled": -0.05}, "bip_count": {"0-0": 0.01},
         "bip_all": 0.0, "evla": {}, "play": {"Single": 0.45, "Out": -0.25}}


def _bip(**cols):
    n = len(next(iter(cols.values())))
    base = {"PitchCall": ["InPlay"] * n, "Balls": [0] * n, "Strikes": [0] * n, "ExitSpeed": [np.nan] * n,
            "Angle": [np.nan] * n, "PlayResult": ["Out"] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_level_mapping_makes_hard_relative_to_the_level():
    level = pd.Series(np.linspace(50, 100, 200))       # a level that hits ~15 mph softer than MLB
    out = RT.level_ev_to_mlb(pd.Series([95.0, 75.0]), level, TIERS)
    assert out[0] > 104 and 85 < out[1] < 90             # 95 is top 10% here, 75 is the middle
    assert RT.level_ev_to_mlb(pd.Series([95.0]), level[:10], TIERS).isna().all()   # too few to place


def test_tiers_are_assigned_by_what_was_recorded():
    df = _bip(Angle=[15.0, np.nan, np.nan], ExitSpeed_mlb=[105.0, np.nan, np.nan],
              HitType=[None, "line drive", None], Contact=[None, "Hard", None], PlayResult=["Single", "Single", "Single"])
    out = RT.classify_contact(df, TIERS)
    assert list(out["result_tier"]) == ["measured", "charted", "outcome"]
    assert list(out["bip_type"][:2]) == ["LineDrive", "LineDrive"] and list(out["contact"][:2]) == ["hard", "hard"]
    v = RT.tiered_values(out, TABLE, TIERS)
    assert np.isclose(v[0], 0.01 + 0.5) and np.isclose(v[1], 0.51) and np.isclose(v[2], 0.01 + 0.45)


def test_charted_inputs_accept_common_spellings():
    assert list(RT.normalize_hit_type(pd.Series(["GB", "Fly Ball", "popup", "LD", "xx"]))) == \
        ["GroundBall", "FlyBall", "Popup", "LineDrive", None]
    assert list(RT.normalize_contact(pd.Series(["Soft", "medium", "HARD", None]))) == ["weak", "average", "hard", None]


def test_weight_blends_by_tier_mix():
    df = pd.DataFrame({"PitcherId": "P", "shape_family": "Slider",
                       "result_tier": ["measured"] * 3 + ["outcome"]})
    mix = RT.tier_mix(df, ["PitcherId", "shape_family"]).iloc[0]
    assert mix["results_tier"] == "measured"
    assert np.isclose(mix["results_k"], (3 * RT.TIER_K["measured"] + RT.TIER_K["outcome"]) / 4)


def test_lower_tiers_never_count_more():
    k = RT.TIER_K
    assert k["measured"] < k["charted"] < k["outcome"]
