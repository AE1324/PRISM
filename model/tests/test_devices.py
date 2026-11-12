import numpy as np
import pandas as pd

from prism_data.devices import PROFILES, apply_offsets, describe, device_of, profile
from prism_data.sessions import tag_sessions


def test_profiles_carry_the_device_differences():
    tm, rap = profile("TrackMan"), profile("rapsodo")
    assert tm.location == "measured" and tm.best_results_tier == "measured"
    assert rap.location == "projected" and rap.best_results_tier == "charted"
    assert profile("yakkertech").key == "unknown" and profile("yakkertech").best_results_tier == "charted"
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
