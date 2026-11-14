"""
MLB reference tables the arsenal reports compare against.

Built once from the classified Statcast seasons and cached in data/models/:

  mlb_rows.parquet        one row per MLB pitcher-season pitch type (100+ pitches)
                          with shape, Stuff/Shape predictions, Location and
                          Results. Used for MLB-scale grades, grade explanations
                          and comparable pitches.
  mlb_reference.joblib    everything else:
      results       run value of each pitch outcome by count, plus how exit
                    velocity and launch angle move the value of a ball in play
      scale         mean and SD of each component per pitch type (the MLB 100 scale)
      features      MLB average shape per pitch type (for "Velo +9" style explanations)
      rates         MLB pooled rates per pitch type (strike, whiff, chase, ...)
      first_pitch   pitcher-season first-pitch strike rate range
      hard_hit_mph  exit velocity counted as hard contact

Usage (from the model/ folder):
    python -m prism_model.references build
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR
from prism_model.stuff import LOAD_COLUMNS, MIN_PITCHES, add_xrv, aggregate, predict

ARTIFACT_DIR = DATA_DIR / "models"
ROWS_PATH = ARTIFACT_DIR / "mlb_rows.parquet"
REFERENCE_PATH = ARTIFACT_DIR / "mlb_reference.joblib"
MLB_YEARS = [2022, 2023, 2024, 2025]

HARD_HIT_MPH = 95.0
EV_BIN = 5.0     # mph
LA_BIN = 10.0    # degrees

# Strike zone in feet (catcher's view, plate center = 0). The competitive zone
# is about 133% of the zone, the middle about 67%, following common public
# definitions (e.g. Baseball Prospectus's competitive zone).
ZONE_SIDE, ZONE_LOW, ZONE_HIGH = 0.83, 1.5, 3.5
COMPETITIVE_SIDE, COMPETITIVE_LOW, COMPETITIVE_HIGH = 1.11, 1.17, 3.83
MIDDLE_SIDE, MIDDLE_LOW, MIDDLE_HIGH = 0.56, 1.83, 3.17

STRIKE_CALLS = {"StrikeCalled", "StrikeSwinging", "FoulBall", "InPlay"}
SWING_CALLS = {"StrikeSwinging", "FoulBall", "InPlay"}
BEHIND_COUNTS = {"1-0", "2-0", "2-1", "3-0", "3-1"}
AHEAD_COUNTS = {"0-1", "0-2", "1-2"}

RATE_COLUMNS = ["strike", "called_strike", "zone", "whiff", "chase", "hard", "gb", "putaway"]


def num(s: pd.Series) -> pd.Series:
    """Numeric version of a column that may hold blanks or pd.NA."""
    return pd.to_numeric(s, errors="coerce").astype(float)


# --------------------------------------------------------------------------
# Pitch-level flags (shared with the arsenal engine)
# --------------------------------------------------------------------------
def add_flags(df: pd.DataFrame, hard_hit_mph: float = HARD_HIT_MPH) -> pd.DataFrame:
    """
    Per-pitch indicators. Rates are left as NaN where a pitch doesn't count
    toward them (whiff only on swings, chase only outside the zone, and so on),
    so a plain mean gives the right rate.
    """
    df = df.copy()
    call = df["PitchCall"].astype(object)
    game = call.notna()
    side, height = num(df["PlateLocSide"]), num(df["PlateLocHeight"])
    located = side.notna() & height.notna()

    def box(s, lo, hi):
        return (side.abs() <= s) & height.between(lo, hi)

    df["count"] = np.where(df["Balls"].notna() & df["Strikes"].notna(),
                           df["Balls"].astype("string") + "-" + df["Strikes"].astype("string"), None)
    df["is_game"] = game
    df["located"] = located
    df["in_zone"] = box(ZONE_SIDE, ZONE_LOW, ZONE_HIGH).where(located)
    df["competitive"] = box(COMPETITIVE_SIDE, COMPETITIVE_LOW, COMPETITIVE_HIGH).where(located)
    df["middle"] = box(MIDDLE_SIDE, MIDDLE_LOW, MIDDLE_HIGH).where(located)

    swing = call.isin(SWING_CALLS)
    in_play = call == "InPlay"
    # hard contact is judged on the MLB-equivalent exit velocity when the level was translated
    ev = num(df["ExitSpeed_mlb"]) if "ExitSpeed_mlb" in df else num(df["ExitSpeed"])
    la = num(df["Angle"])
    df["strike"] = call.isin(STRIKE_CALLS).where(game)
    df["called_strike"] = (call == "StrikeCalled").where(game)
    df["zone"] = df["in_zone"].where(game)
    df["swing"] = swing.where(game)
    df["whiff"] = (call == "StrikeSwinging").where(swing)
    df["chase"] = swing.where(game & (df["in_zone"] == False))  # noqa: E712
    df["in_play"] = in_play.where(game)
    df["hard"] = (ev >= hard_hit_mph).where(in_play & ev.notna())
    df["gb"] = (la < 10).where(in_play & la.notna())
    two_strike = game & (df["Strikes"] == 2)
    df["putaway"] = (df["KorBB"].astype(object) == "Strikeout").where(two_strike)
    df["behind"] = df["count"].isin(BEHIND_COUNTS).where(game & df["count"].notna())
    df["ahead"] = df["count"].isin(AHEAD_COUNTS).where(game & df["count"].notna())
    for col in ["in_zone", "competitive", "middle", "strike", "called_strike", "zone", "swing", "whiff",
                "chase", "in_play", "hard", "gb", "putaway", "behind", "ahead"]:
        df[col] = df[col].astype(float)
    return df


# --------------------------------------------------------------------------
# Results: run value of what actually happened
# --------------------------------------------------------------------------
def build_results_table(df: pd.DataFrame) -> dict:
    """
    Run value of each pitch result, learned from MLB.

    Balls, strikes and fouls: the average run value of that call in that count.
    Balls in play: the count's average for balls in play, moved up or down by
    how hard and at what angle the ball was hit. If exit velocity or launch
    angle is missing (common on weak college contact), the recorded play
    result is used, and failing that the count average.
    """
    d = df[df["PitchCall"].notna() & df["RunValue"].notna()].copy()
    d["count"] = d["Balls"].astype(str) + "-" + d["Strikes"].astype(str)
    bip = d["PitchCall"] == "InPlay"

    by_call = d[~bip].groupby(["count", "PitchCall"])["RunValue"].mean()
    call_only = d[~bip].groupby("PitchCall")["RunValue"].mean()
    bip_count = d[bip].groupby("count")["RunValue"].mean()
    b = d[bip].copy()
    b["resid"] = b["RunValue"] - b["count"].map(bip_count)
    b = b.dropna(subset=["ExitSpeed", "Angle"])
    b["ev_bin"] = (b["ExitSpeed"].clip(40, 120) // EV_BIN).astype(int)
    b["la_bin"] = (b["Angle"].clip(-60, 70) // LA_BIN).astype(int)
    evla = b.groupby(["ev_bin", "la_bin"])["resid"].agg(["mean", "size"])
    evla = evla[evla["size"] >= 30]["mean"]
    play = d[bip].assign(resid=d[bip]["RunValue"] - d[bip]["count"].map(bip_count)).groupby("PlayResult")["resid"].mean()
    return {
        "by_call": by_call.to_dict(),
        "call_only": call_only.to_dict(),
        "bip_count": bip_count.to_dict(),
        "bip_all": float(d.loc[bip, "RunValue"].mean()),
        "evla": evla.to_dict(),
        "play": play.to_dict(),
    }


def result_values(df: pd.DataFrame, table: dict) -> pd.Series:
    """Run value of each game pitch's result (positive = good for the hitter). NaN for bullpen pitches."""
    call = df["PitchCall"].astype(object)
    count = (df["Balls"].astype("string") + "-" + df["Strikes"].astype("string")).astype(object)
    out = pd.Series(np.nan, index=df.index)

    keys = list(zip(count, call))
    by_call = pd.Series([table["by_call"].get(k, np.nan) for k in keys], index=df.index)
    by_call = by_call.fillna(call.map(table["call_only"]))
    nonbip = call.notna() & (call != "InPlay")
    out[nonbip] = by_call[nonbip]

    bip = call == "InPlay"
    if bip.any():
        base = count[bip].map(table["bip_count"]).fillna(table["bip_all"]).astype(float)
        ev, la = num(df.loc[bip, "ExitSpeed"]), num(df.loc[bip, "Angle"])
        ev_bin = (ev.clip(40, 120) // EV_BIN)
        la_bin = (la.clip(-60, 70) // LA_BIN)
        evla = pd.Series([table["evla"].get((int(e), int(a)), np.nan) if pd.notna(e) and pd.notna(a) else np.nan
                          for e, a in zip(ev_bin, la_bin)], index=base.index)
        play = num(df.loc[bip, "PlayResult"].astype(object).map(table["play"]))
        out[bip] = base + evla.fillna(play).fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Per pitcher-type components
# --------------------------------------------------------------------------
def location_score(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Location per pitch type: share in the competitive zone minus share in the middle (higher = better)."""
    loc = df[df["located"]].groupby(by).agg(
        n_located=("competitive", "size"), competitive=("competitive", "mean"), middle=("middle", "mean"))
    loc["location"] = 100 * (loc["competitive"] - loc["middle"])
    return loc.reset_index()


def results_score(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Results per pitch type: run value per 100 game pitches (lower = better for the pitcher)."""
    g = df[df["result_rv"].notna()].groupby(by).agg(n_game=("result_rv", "size"), results_rv100=("result_rv", "mean"))
    g["results_rv100"] *= 100
    return g.reset_index()


# --------------------------------------------------------------------------
# Build and load
# --------------------------------------------------------------------------
def build(years: list[int] = MLB_YEARS) -> dict:
    from joblib import dump

    extra = ["PlateLocSide", "PlateLocHeight", "ExitSpeed", "Angle", "PlayResult", "KorBB", "bucket", "SpinAxis",
             "VertApprAngle"]
    cols = list(dict.fromkeys(LOAD_COLUMNS + extra))
    frames = []
    for y in years:
        f = pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols)
        frames.append(f)
        print(f"  loaded {y}: {len(f):,} pitches")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["shape_family"].notna()]

    print("Results table…")
    table = build_results_table(df)
    df["result_rv"] = result_values(df, table)
    df = add_flags(df)

    print("Pitcher-season rows and Stuff predictions…")
    by = ["PitcherId", "Season", "shape_family"]
    rows = predict(aggregate(add_xrv(df[df["RunValue"].notna()]), by=by, min_pitches=MIN_PITCHES))
    rows = rows.merge(location_score(df, by), on=by, how="left").merge(results_score(df, by), on=by, how="left")
    rows.to_parquet(ROWS_PATH, index=False)

    scale = rows.groupby("shape_family")[["stuff_xrv", "shape_xrv", "location", "results_rv100"]].agg(["mean", "std"])
    feature_cols = ["velo", "perceived_velo", "ivb", "hb", "spin", "rel_height", "rel_side", "velo_diff", "ivb_diff",
                    "hb_diff"]
    features = rows.groupby("shape_family")[feature_cols].mean()
    rates = df.groupby("shape_family")[RATE_COLUMNS].mean()

    first = df[(df["count"] == "0-0") & df["is_game"]].groupby(["PitcherId", "Season"])["strike"].agg(["mean", "size"])
    first = first[first["size"] >= 100]["mean"]

    ref = {
        "years": years,
        "results": table,
        "scale": {k: {f"{a}_{b}": float(v) for (a, b), v in row.items()} for k, row in scale.iterrows()},
        "features": features.to_dict("index"),
        "rates": rates.to_dict("index"),
        "first_pitch": {q: float(first.quantile(q)) for q in (0.25, 0.5, 0.75)},
        "hard_hit_mph": HARD_HIT_MPH,
    }
    dump(ref, REFERENCE_PATH)
    return ref


def refresh_stuff() -> dict:
    """
    Re-predict Stuff and Shape for the saved MLB pitcher-season rows and reset their scale (mean and SD by
    pitch type), leaving everything else in the reference as built. For when the Stuff prediction changes
    but the Statcast seasons behind the rows aren't on this machine.
    """
    from joblib import dump, load as jl

    ref, rows = jl(REFERENCE_PATH), pd.read_parquet(ROWS_PATH)
    rows = predict(rows.drop(columns=["stuff_xrv", "shape_xrv"]))[list(rows.columns)]
    rows.to_parquet(ROWS_PATH, index=False)
    scale = rows.groupby("shape_family")[["stuff_xrv", "shape_xrv"]].agg(["mean", "std"])
    for kind, row in scale.iterrows():
        ref["scale"].setdefault(kind, {}).update({f"{a}_{b}": float(v) for (a, b), v in row.items()})
    dump(ref, REFERENCE_PATH)
    _CACHE.clear()
    return ref


_CACHE: dict = {}


def load() -> tuple[dict, pd.DataFrame]:
    """The cached MLB reference and MLB pitcher-season rows (building them on first use)."""
    if "ref" not in _CACHE:
        from joblib import load as jl

        if not REFERENCE_PATH.exists() or not ROWS_PATH.exists():
            print("Building the MLB reference tables (one time)…")
            build()
        _CACHE["ref"] = jl(REFERENCE_PATH)
        _CACHE["rows"] = pd.read_parquet(ROWS_PATH)
    return _CACHE["ref"], _CACHE["rows"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--years", type=int, nargs="+", default=MLB_YEARS)
    parser.add_argument("--refresh-stuff", action="store_true", help="only re-predict Stuff/Shape on the saved rows")
    args = parser.parse_args()
    ref = refresh_stuff() if args.refresh_stuff else build(args.years)
    fp = ref["first_pitch"]
    print(f"\nFirst-pitch strike, middle half of MLB pitchers: {fp[0.25]:.0%}–{fp[0.75]:.0%}")
    print("Scale per pitch type (mean / SD):")
    for kind, s in ref["scale"].items():
        print(f"  {kind:<10} stuff {s['stuff_xrv_mean']:+.2f}/{s['stuff_xrv_std']:.2f}  "
              f"location {s['location_mean']:.1f}/{s['location_std']:.1f}  "
              f"results {s['results_rv100_mean']:+.2f}/{s['results_rv100_std']:.2f}")
    print(f"\nSaved to {REFERENCE_PATH} and {ROWS_PATH}")


if __name__ == "__main__":
    main()
