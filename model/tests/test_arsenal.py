"""Checks for the full-arsenal engine, using a synthetic staff and stand-in models (no MLB data needed)."""
import numpy as np
import pandas as pd
import pytest

from prism_data.schema import ALL_COLUMNS
from prism_model.arsenal import analyze, clearly, pitch_plus, staff_rescale, tilt, to_markdown
from prism_model.references import add_flags, result_values


# --------------------------------------------------------------------------
# Unit pieces
# --------------------------------------------------------------------------
def test_pitch_plus_matches_garnet_sweeper_example():
    # Case study: Stuff 99, Location 95 -> projected 97; Results 88 over 41 game pitches -> 90.
    assert round(pitch_plus(99, 95, 88, 41, k=15)) == 90  # Garnet's case-study example
    assert round(pitch_plus(99, 95, 88, 41)) == 97  # PRISM's evidence-based weight


def test_pitch_plus_without_games_is_the_projection():
    assert pitch_plus(110, 90, None, 0) == pytest.approx(100)
    assert pitch_plus(110, None, None, 0) == pytest.approx(110)
    assert pitch_plus(None, None, 90, 30) is None


def test_clearly_needs_sample_and_a_real_gap():
    assert clearly(0.80, 0.60, 40, 15) == 1
    assert clearly(0.40, 0.60, 40, 15) == -1
    assert clearly(0.80, 0.60, 10, 15) == 0     # too few
    assert clearly(0.62, 0.60, 400, 15) == 0    # under 3 points


def test_staff_rescale_widens_when_a_type_is_rare():
    rows = pd.DataFrame({
        "shape_family": ["Slider"] * 4 + ["Curveball", "Four-seam"],
        "bucket": ["Breaking"] * 5 + ["Fastball"],
        "x": [90.0, 100.0, 110.0, 100.0, 120.0, 95.0],
    })
    out = staff_rescale(rows, "x")
    assert list(out["basis"]) == ["type"] * 4 + ["group", "all"]
    assert out["plus"].iloc[:4].mean() == pytest.approx(100)


def test_tilt_clock():
    assert tilt(180) == "12:00"
    assert tilt(210) == "1:00"
    assert tilt(135) == "10:30"


def test_flags_zone_chase_and_whiff():
    df = pd.DataFrame({
        "PitchCall": ["StrikeSwinging", "BallCalled", "FoulBall", "InPlay", None],
        "PlateLocSide": [0.0, 1.5, 1.5, 0.0, 0.0], "PlateLocHeight": [2.5, 2.5, 2.5, 2.5, 2.5],
        "Balls": [0, 1, 1, 1, pd.NA], "Strikes": [0, 0, 1, 2, pd.NA], "KorBB": [None] * 5,
        "ExitSpeed": [None, None, None, 101.0, None], "Angle": [None, None, None, 5.0, None],
    })
    f = add_flags(df)
    assert list(f["in_zone"]) == [1, 0, 0, 1, 1]
    assert f["chase"].tolist()[1:3] == [0, 1]            # take vs swing outside the zone
    assert f["whiff"].dropna().tolist() == [1, 0, 0]     # one whiff in three swings
    assert f["hard"].dropna().tolist() == [1] and f["gb"].dropna().tolist() == [1]
    assert np.isnan(f["strike"].iloc[4])                 # bullpen pitch has no strike flag


def test_result_values_use_count_and_contact_quality():
    table = {"by_call": {("0-0", "BallCalled"): 0.03, ("0-0", "StrikeCalled"): -0.04},
             "call_only": {"BallCalled": 0.05, "StrikeCalled": -0.05},
             "bip_count": {"0-0": 0.0}, "bip_all": 0.0, "evla": {(20, 1): 0.5}, "play": {"Out": -0.25}}
    df = pd.DataFrame({"PitchCall": ["BallCalled", "StrikeCalled", "InPlay", "InPlay", None],
                       "Balls": [0, 2, 0, 0, pd.NA], "Strikes": [0, 2, 0, 0, pd.NA],
                       "ExitSpeed": [None, None, 102.0, None, None], "Angle": [None, None, 15.0, None, None],
                       "PlayResult": [None, None, "Single", "Out", None]})
    rv = result_values(df, table).tolist()
    assert rv[:4] == pytest.approx([0.03, -0.05, 0.5, -0.25])
    assert np.isnan(rv[4])


# --------------------------------------------------------------------------
# Whole engine on a synthetic staff
# --------------------------------------------------------------------------
class FakeModel:
    """Lower runs for harder, more-moving pitches; enough to exercise the engine."""

    def __init__(self, features):
        self.features = features

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.features)
        out = np.zeros(len(X))
        if "perceived_velo" in X:
            out -= 0.15 * (X["perceived_velo"] - 85)
        out -= 0.05 * X["ivb"].abs() + 0.05 * X["hb"].abs()
        return out.to_numpy() if hasattr(out, "to_numpy") else out


def fake_models():
    from prism_model.stuff import GROUPS

    models = {}
    for g, spec in GROUPS.items():
        shape = [f for f in spec["features"] if f != "perceived_velo"]
        models[g] = {"features": spec["features"], "stuff": FakeModel(spec["features"]),
                     "shape_features": shape, "shape": FakeModel(shape)}
    return models


def fake_ref():
    kinds = ["Four-seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]
    scale = {k: {"stuff_xrv_mean": -1.0, "stuff_xrv_std": 0.5, "shape_xrv_mean": -1.0, "shape_xrv_std": 0.5,
                 "location_mean": 20.0, "location_std": 8.0, "results_rv100_mean": 0.0, "results_rv100_std": 1.5}
             for k in kinds}
    feats = {k: {"velo": 85, "perceived_velo": 85, "ivb": 5, "hb": 0, "spin": 2300, "rel_height": 5.8,
                 "rel_side": 1.8, "velo_diff": -8, "ivb_diff": -10, "hb_diff": -10} for k in kinds}
    rates = {k: {"strike": 0.63, "called_strike": 0.17, "zone": 0.48, "whiff": 0.25, "chase": 0.30,
                 "hard": 0.38, "gb": 0.43, "putaway": 0.20} for k in kinds}
    results = {"by_call": {}, "call_only": {"BallCalled": 0.06, "StrikeCalled": -0.06, "StrikeSwinging": -0.07,
                                           "FoulBall": -0.04, "HitByPitch": 0.2},
               "bip_count": {}, "bip_all": 0.0, "evla": {}, "play": {"Out": -0.25, "Single": 0.45, "Double": 0.75}}
    return {"scale": scale, "features": feats, "rates": rates, "results": results,
            "first_pitch": {0.25: 0.59, 0.5: 0.62, 0.75: 0.64}, "hard_hit_mph": 95.0}


def fake_mlb_rows():
    rng = np.random.default_rng(1)
    n = 300
    return pd.DataFrame({"shape_family": rng.choice(["Four-seam", "Slider", "Changeup"], n),
                         "velo": rng.uniform(80, 98, n), "ivb": rng.uniform(-5, 20, n), "hb": rng.uniform(-15, 15, n),
                         "xrv100": rng.normal(0, 1, n), "n": rng.integers(100, 1500, n)})


def staff(seed=3):
    """Five right-handers with fastball/slider/changeup; pitcher 'A' throws few strikes with his slider."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, pid in enumerate("ABCDE"):
        for date in ("2026-03-01", "2026-03-08"):
            for i in range(40):
                kind = ["Four-seam", "Four-seam", "Slider", "Changeup"][i % 4]
                velo = {"Four-seam": 88 + k, "Slider": 79 + k, "Changeup": 80 + k}[kind]
                ivb, hb = {"Four-seam": (16, 9), "Slider": (2, -6), "Changeup": (8, 13)}[kind]
                wild = pid == "A" and kind == "Slider"
                call = rng.choice(["BallCalled", "StrikeCalled", "StrikeSwinging", "FoulBall", "InPlay"],
                                  p=[0.8, 0.05, 0.05, 0.05, 0.05] if wild else [0.35, 0.2, 0.15, 0.15, 0.15])
                rows.append({
                    "Source": "trackman", "Date": date, "PitchNo": i + 1, "Pitcher": f"Pitcher, {pid}", "PitcherId": pid,
                    "PitcherThrows": "R", "PitcherTeam": "OBE", "BatterSide": "L" if i % 3 else "R",
                    "Balls": [0, 1, 2, 1][i % 4], "Strikes": [0, 1, 1, 2][i % 4], "PitchCall": call,
                    "PlayResult": rng.choice(["Out", "Single"]) if call == "InPlay" else "Undefined",
                    "KorBB": "Undefined", "ExitSpeed": rng.uniform(70, 100) if call == "InPlay" else np.nan,
                    "Angle": rng.uniform(-10, 30) if call == "InPlay" else np.nan,
                    "RelSpeed": velo + rng.normal(0, 0.7), "SpinRate": 2300.0, "SpinAxis": 210.0, "Extension": 6.0,
                    "InducedVertBreak": ivb + rng.normal(0, 1), "hb_arm": hb + rng.normal(0, 1), "RelHeight": 5.8,
                    "rel_side_arm": 1.8, "PlateLocSide": rng.normal(0, 0.9), "PlateLocHeight": rng.normal(2.5, 0.8),
                    "VertApprAngle": -5.0,
                })
    df = pd.DataFrame(rows)
    for c in ALL_COLUMNS:
        if c not in df:
            df[c] = pd.NA
    df["shape_family"] = df["RelSpeed"].map(lambda v: None)
    fam = np.where(df["InducedVertBreak"] > 12, "Four-seam", np.where(df["hb_arm"] > 0, "Changeup", "Slider"))
    df["shape_family"] = fam
    df["bucket"] = df["shape_family"].map({"Four-seam": "Fastball", "Slider": "Breaking", "Changeup": "Offspeed"})
    df["subtype"] = df["shape_family"].map({"Four-seam": "Ride", "Slider": "Traditional", "Changeup": "Fade"})
    fb = df[df["shape_family"] == "Four-seam"].groupby("PitcherId")[["RelSpeed", "InducedVertBreak", "hb_arm"]].median()
    df["fb_velo"] = df["PitcherId"].map(fb["RelSpeed"])
    df["fb_ivb"] = df["PitcherId"].map(fb["InducedVertBreak"])
    df["fb_hb"] = df["PitcherId"].map(fb["hb_arm"])
    return df


@pytest.fixture(scope="module")
def reports():
    return {r["pitcher_id"]: r for r in analyze(staff(), fake_ref(), fake_mlb_rows(), fake_models(), scale="group")}


def test_every_pitcher_gets_a_full_report(reports):
    assert set(reports) == set("ABCDE")
    r = reports["B"]
    assert r["pitcher"] == "B Pitcher"
    for key in ("summary", "development", "pitch_by_pitch", "situations", "movement_profile", "progress", "fastball_role"):
        assert r[key]
    kinds = [p["pitch"] for p in r["pitch_by_pitch"]]
    assert kinds[0] == "Four-seam" and set(kinds) == {"Four-seam", "Slider", "Changeup"}
    fb = r["pitch_by_pitch"][0]
    assert isinstance(fb["grades"]["pitch_plus"], int)
    assert fb["grades"]["compared_to"] == "other four-seams in this file"
    assert "Velo" in fb["stuff_explained"] and "Separation" not in fb["stuff_explained"]


def test_wild_slider_is_struggling_and_drives_priorities(reports):
    sl = next(p for p in reports["A"]["pitch_by_pitch"] if p["pitch"] == "Slider")
    assert sl["verdict"] == "Struggling"
    assert sl["deal_breakers"][0]["issue"] == "Not landing strikes"
    assert "zone" in sl["so"]
    assert reports["A"]["development"]["priority_order"][0].startswith("Fix the slider")


def test_situations_and_progress(reports):
    s = reports["C"]["situations"]
    assert s["first_pitch_strike"]["pro_range"] == "59–64%"
    assert {row["count"].split(" ")[0] for row in s["by_count"]} >= {"First", "Ahead", "Behind"}
    assert set(s["splits"]) == {"vs LHH", "vs RHH"}
    prog = reports["C"]["progress"]
    assert {x["outing"] for x in prog} == {"2026-03-01", "2026-03-08"}
    assert all(isinstance(x["stuff_plus"], (int, float)) for x in prog)


def test_progress_is_na_under_five_pitches():
    df = staff()
    # Pitcher B threw only three changeups on a third date.
    extra = df[(df["PitcherId"] == "B") & (df["shape_family"] == "Changeup")].head(3).assign(Date="2026-03-15")
    out = analyze(pd.concat([df, extra], ignore_index=True), fake_ref(), fake_mlb_rows(), fake_models(), pitchers=["B"])
    late = [x for x in out[0]["progress"] if x["outing"] == "2026-03-15"]
    assert late and late[0]["stuff_plus"] == "N/A"


def test_bullpen_only_pitches_get_shape_grades_but_no_verdict_on_results():
    df = staff()
    for c in ("PitchCall", "Balls", "Strikes", "ExitSpeed", "Angle", "PlayResult", "KorBB"):
        df[c] = pd.NA
    out = analyze(df, fake_ref(), fake_mlb_rows(), fake_models(), pitchers=["B"])[0]
    fb = out["pitch_by_pitch"][0]
    assert fb["verdict"] == "Bullpen only"
    assert fb["grades"]["results_plus"] == "N/A" and isinstance(fb["grades"]["stuff_plus"], int)


def test_markdown_renders(reports):
    md = to_markdown(reports["A"])
    for heading in ("## Development", "## Pitch grades", "## Pitch by pitch", "## Situations", "## Movement profile"):
        assert heading in md
    assert "**So:**" in md


def _loc_model():
    return {"d3_center": {"Fastball": 0.008, "Slider": 0.009, "Changeup": 0.010, "Curveball": 0.0075, "Cutter": 0.009},
            "scale_sd": 0.0044}


def test_location_plus_is_absolute_not_staff_relative():
    """A pitcher's Location+ depends on his locations vs D3 average, not on how well his teammates locate."""
    import pandas as pd
    from prism_model import location

    def staff(teammate_value):
        df = pd.DataFrame({"PitcherId": ["me"] * 300 + ["mate"] * 300, "shape_family": "Four-seam",
                           "loc_value": [0.004] * 300 + [teammate_value] * 300})
        rows = pd.DataFrame({"PitcherId": ["me", "mate"], "shape_family": ["Four-seam"] * 2})
        out = location.grade(rows, df, ["PitcherId", "shape_family"], _loc_model())
        return out.set_index("PitcherId")["location_plus"]

    good_staff, bad_staff = staff(0.006), staff(0.030)
    assert abs(good_staff["me"] - bad_staff["me"]) < 1e-9      # teammates don't move him
    assert good_staff["me"] > 100 > bad_staff["mate"]            # better than D3 average spots grade above 100


def test_location_plus_regresses_small_samples_toward_average():
    import pandas as pd
    from prism_model import location

    def plus(n):
        df = pd.DataFrame({"PitcherId": "me", "shape_family": "Four-seam", "loc_value": [0.0] * n})
        rows = pd.DataFrame({"PitcherId": ["me"], "shape_family": ["Four-seam"]})
        return float(location.grade(rows, df, ["PitcherId", "shape_family"], _loc_model())["location_plus"].iloc[0])

    assert 100 < plus(20) < plus(200) < plus(2000)              # same spots: more pitches, more trust


def test_pitch_plus_needs_15_pitches_vs_hitters():
    from prism_model.arsenal import MIN_GAME_FOR_PITCH_PLUS, pitch_plus
    assert MIN_GAME_FOR_PITCH_PLUS == 15
    # pitch_plus itself still computes; the gate lives in grade_rows / the report entry
    assert pitch_plus(110, 100, 90, 14) is not None


def test_similar_pool_compares_changeups_by_separation_off_a_similar_fastball():
    import pandas as pd
    from prism_model.arsenal import similar_pool
    rng = np.random.default_rng(0)
    n = 600
    mlb = pd.DataFrame({"shape_family": "Changeup", "velo_diff": rng.normal(-9, 2, n), "ivb_diff": rng.normal(-8, 3, n),
                        "hb_diff": rng.normal(4, 3, n), "spin": rng.normal(1700, 200, n)})
    mlb["ivb"] = rng.normal(16, 2, n) + mlb["ivb_diff"]
    fb_hb = np.where(np.arange(n) < 300, 7.0, 15.0)          # half off four-seams, half off high-run sinkers
    mlb["hb"] = fb_hb + mlb["hb_diff"]
    me = pd.Series({"shape_family": "Changeup", "bucket": "Offspeed", "velo_diff": -7.6, "ivb_diff": -8.1, "hb_diff": 3.9,
                    "ivb": 6.2, "hb": 18.3, "spin": 1636})
    pool = similar_pool(me, mlb)
    assert 60 <= len(pool) <= 150
    assert (pool["hb"] - pool["hb_diff"]).mean() > 12          # mostly changeups off high-run fastballs


def test_rare_movement_gets_no_target():
    from prism_model.arsenal import SUPPORT_Q
    assert SUPPORT_Q == 0.03


def test_rapsodo_location_is_graded_through_its_blur():
    """The same measured spots grade higher from a blurred device: its average includes the blur's cost."""
    import pandas as pd
    from prism_model import location

    model = dict(_loc_model(), fams=["Four-seam"], noise_penalty={0: {"Four-seam": 0.0}, 6: {"Four-seam": 0.009}})
    df = pd.DataFrame({"PitcherId": "me", "shape_family": "Four-seam", "loc_value": [0.012] * 200})
    rows = pd.DataFrame({"PitcherId": ["me"], "shape_family": ["Four-seam"]})
    tm = location.grade(rows.assign(device="trackman"), df, ["PitcherId", "shape_family"], model)["location_plus"].iloc[0]
    rap = location.grade(rows.assign(device="rapsodo"), df, ["PitcherId", "shape_family"], model)["location_plus"].iloc[0]
    assert tm < 100 < rap
