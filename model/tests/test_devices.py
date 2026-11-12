import numpy as np
import pandas as pd

from prism_data.devices import PROFILES, apply_offsets, describe, device_of, profile
from prism_data.sessions import tag_sessions


def test_profiles_carry_the_device_differences():
    tm, rap = profile("TrackMan"), profile("rapsodo")
    assert tm.location == "measured" and not tm.calibrate_location and tm.best_results_tier == "measured"
    assert rap.location == "extrapolated" and rap.calibrate_location and rap.best_results_tier == "charted"
    assert profile("yakkertech").key == "unknown" and profile("yakkertech").calibrate_location
    assert describe("trackman")["label"] == "TrackMan"


def test_session_defaults_come_from_the_profile():
    df = pd.DataFrame({"Source": ["trackman", "rapsodo", "mystery"], "Date": "2026-06-11", "PitcherId": "P"})
    assert list(tag_sessions(df)["Setting"]) == ["game", "bullpen", "bullpen"]


def test_provisional_offsets_change_nothing():
    assert all(not o.applied for p in PROFILES.values() for o in p.offsets)
    df = pd.DataFrame({"Source": ["rapsodo", "trackman"], "RelHeight": [5.8, 6.0], "RelSpeed": [84.0, 83.0]})
    assert apply_offsets(df).equals(df)


def test_device_of_prefers_the_session_tag():
    df = pd.DataFrame({"Source": ["rapsodo"], "Device": ["TrackMan"]})
    assert device_of(df).iloc[0] == "trackman"


def test_measured_location_uses_its_own_level_not_strike_rate():
    from prism_model.arsenal import regress_location
    ref = {"scale": {"Four-seam": {"location_mean": 40.0}}}
    d3ref = {"location": {"Four-seam": {"d3_location_mean": 43.5, "slope": 0.8, "d3_strike": 62.0}}}
    rows = pd.DataFrame({"PitcherId": ["A", "B"], "shape_family": "Four-seam", "location": [36.0, 42.0],
                         "n_located": [400, 400]})
    strikes = pd.DataFrame({"device": ["trackman"], "shape_family": ["Four-seam"], "strike": [0.62], "n": [800]})
    # TrackMan with a measured level center of 37 (staff averages 39): staff sits at its measured level vs that center
    tm = regress_location(rows.assign(device="trackman", level_location=37.0), ref, d3ref, strikes)
    # Rapsodo: the staff level comes from strike rate (exactly D3 average here), whatever the raw locations say
    rap = regress_location(rows.assign(device="rapsodo"), ref, d3ref, strikes.assign(device="rapsodo"))
    assert np.isclose(rap["location"].mean(), 40.0, atol=0.5)            # centered: strike rate = D3 average
    assert tm["location"].mean() > 40.0                                   # staff is above the level's measured center
    assert set(tm["location_level"]) == {"measured"} and set(rap["location_level"]) == {"strike rate"}
