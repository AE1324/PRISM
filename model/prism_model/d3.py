"""
D3 context for grading: velocity translation, a D3 velocity weight, and D3
rate references.

1. Velocity translation. A pitcher's main fastball (perceived velocity, so
   extension counts) is placed on the D3 pitcher scale and moved to the MLB
   perceived velocity at the same percentile. Every pitch he throws moves by
   the same amount, so the gaps between his pitches (which the MLB model also
   uses) are unchanged.
2. D3 velocity weight. Synergy velocity bands show whiffs per swing rising about
   twice as fast with velocity percentile in D3 as in MLB
   (``python -m prism_model.velo_whiff_check``). The velocity part of a
   fastball's Stuff (Stuff minus Shape) is multiplied by this weight.
3. D3 rates. Strike% and whiff-per-swing by pitch type from the Synergy team
   pulls, re-weighted so team strength matches all of D3.

All inputs are licensed/derived data in the gitignored data/ folder.

    python -m prism_model.d3 build     # writes data/models/d3_reference.joblib
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

PATH = DATA_DIR / "models" / "d3_reference.joblib"
FASTBALLS = ["Four-seam", "Sinker"]
MAX_VELO_WEIGHT = 2.0
MPH_PER_FT = 1.6
MLB_AVG_EXTENSION_FT = 6.3

# PRISM shape family -> Synergy pitch type.
SYNERGY_TYPE = {"Four-seam": "Fastball", "Sinker": "Fastball", "Cutter": "Cutter", "Slider": "Slider",
                "Sweeper": "Slider", "Curveball": "Curveball", "Changeup": "Changeup", "Splitter": "Changeup"}


def _mlb_perceived_scale(years=(2024, 2025)) -> np.ndarray:
    cols = ["PitcherId", "Season", "shape_family", "RelSpeed", "Extension"]
    m = pd.concat([pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols) for y in years])
    fb = m[m["shape_family"].isin(FASTBALLS)].copy()
    fb["pv"] = fb["RelSpeed"].astype(float) + MPH_PER_FT * (fb["Extension"].astype(float) - MLB_AVG_EXTENSION_FT)
    fb["pv"] = fb["pv"].fillna(fb["RelSpeed"].astype(float))
    per = fb.groupby(["PitcherId", "Season"])["pv"].agg(["size", "mean"])
    return np.percentile(per[per["size"] >= 50]["mean"], np.arange(1, 100))


def _d3_rates() -> dict:
    r = pd.read_csv(DATA_DIR / "synergy" / "team_results_2026.csv")
    s = pd.read_csv(DATA_DIR / "synergy_team_strength_2026.csv")
    st = dict(zip(s["team_name"], s["strength_pct"]))
    r["strength"] = r["team"].map(st)
    r = r[r["strength"].notna() & (r["pitch_type"] != "Overall")]
    bins = np.linspace(0, 100.01, 6)
    share = pd.Series(np.digitize(s["strength_pct"], bins)).value_counts(normalize=True)
    teams = r.drop_duplicates("team").assign(bin=lambda d: np.digitize(d["strength"], bins))
    tw = teams.set_index("team")["bin"].map(share / teams["bin"].value_counts(normalize=True))
    r["w"] = r["team"].map(tw)
    out = {}
    for kind, g in r.groupby("pitch_type"):
        if g["total"].sum() < 500:
            continue
        w = g["w"]
        out[kind] = {
            "strike": float(np.average(g["strike"] / g["total"], weights=w * g["total"])),
            "whiff": float(np.average(g["miss"] / g["swing"].clip(lower=1), weights=w * g["swing"])),
            "n_teams": int(g["team"].nunique()), "n_pitches": int(g["total"].sum()),
        }
    return out


def _location_reference(years=(2024, 2025), min_pitches: int = 100) -> dict:
    """
    D3 centering for Location+. There's no D3 plate-location data, but there are D3 strike rates. On MLB
    pitcher-season pitch types, location score rises with strike rate; that slope turns the D3-vs-MLB
    strike gap into an expected D3 location score, and later turns a staff's own strike rate into where
    the staff's command sits (strike calls are outcomes, so they don't depend on the tracking device).
    """
    from prism_model import references as R

    strike_calls = set(R.STRIKE_CALLS)
    parts = []
    for yr in years:
        f = PROCESSED_DIR / f"statcast_{yr}_shapes.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f, columns=["PitcherId", "shape_family", "PitchCall", "PlateLocSide", "PlateLocHeight"])
        s = s.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        box = lambda w, lo, hi: (s["PlateLocSide"].abs() <= w) & s["PlateLocHeight"].between(lo, hi)  # noqa: E731
        s["comp"] = box(R.COMPETITIVE_SIDE, R.COMPETITIVE_LOW, R.COMPETITIVE_HIGH)
        s["mid"] = box(R.MIDDLE_SIDE, R.MIDDLE_LOW, R.MIDDLE_HIGH)
        s["strike"] = s["PitchCall"].isin(strike_calls)
        g = s.groupby(["PitcherId", "shape_family"]).agg(n=("strike", "size"), strike=("strike", "mean"),
                                                         comp=("comp", "mean"), mid=("mid", "mean"))
        parts.append(g[g["n"] >= min_pitches].reset_index())
    if not parts:
        return {}
    g = pd.concat(parts)
    g["loc"] = 100 * (g["comp"] - g["mid"])
    g["strike"] *= 100
    rates = _d3_rates()
    mlb_scale = R.load()[0]["scale"]
    out = {}
    for kind, x in g.groupby("shape_family"):
        syn = SYNERGY_TYPE.get(kind)
        if syn not in rates or kind not in mlb_scale or len(x) < 30:
            continue
        slope = float(np.polyfit(x["strike"], x["loc"], 1, w=np.sqrt(x["n"]))[0])
        mlb_strike = float(np.average(x["strike"], weights=x["n"]))
        d3_strike = 100 * rates[syn]["strike"]
        out[kind] = {
            "slope": slope, "mlb_strike": mlb_strike, "d3_strike": d3_strike,
            "d3_location_mean": float(mlb_scale[kind]["location_mean"] + slope * (d3_strike - mlb_strike)),
        }
    return out


def add_location_reference() -> dict:
    """Add (or refresh) the Location+ D3 centering in the saved reference without rebuilding the rest."""
    ref = load() or {}
    ref["location"] = _location_reference()
    joblib.dump(ref, PATH)
    return ref


def build() -> dict:
    from prism_model.velo_whiff_check import _logit_slope, d3_curve, mlb_curve

    m, d = mlb_curve(), d3_curve()
    ratio = (_logit_slope(d.pct, d.whiff_per_swing, d.swings) /
             _logit_slope(m.pct, m.whiff_per_swing, m.swings))
    scale = pd.read_csv(DATA_DIR / "models" / "d3_fb_velo_scale.csv")
    ref = {
        "d3_scale_mph": scale["mph"].to_numpy(float),
        "mlb_perceived_mph": _mlb_perceived_scale(),
        "velo_weight_measured": float(ratio),
        "velo_weight": float(min(max(ratio, 1.0), MAX_VELO_WEIGHT)),
        "rates": _d3_rates(),
        "location": _location_reference(),
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, PATH)
    return ref


def load() -> dict | None:
    return joblib.load(PATH) if PATH.exists() else None


def d3_percentile(mph, ref: dict):
    return np.interp(mph, ref["d3_scale_mph"], np.arange(1, 100), left=0.5, right=99.5)


def mlb_equivalent(pct, ref: dict):
    return np.interp(pct, np.arange(1, 100), ref["mlb_perceived_mph"])


def translate(rows: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """
    Work out each pitcher's D3-to-MLB translation; the pitch features themselves are left as thrown.

    His main fastball's perceived velocity -> its D3 percentile -> the MLB perceived velocity at that
    percentile. ``velo_ratio`` = MLB-equivalent / his fastball, applied by ``model_inputs`` to every pitch
    of his. Adds ``velo_ratio``, ``velo_shift`` (fastball mph added, for display) and ``fb_pct_d3``.
    """
    rows = rows.copy()
    rows["perceived_velo_d3"] = rows["perceived_velo"]
    rows["velo_ratio"] = 1.0
    rows["velo_shift"] = 0.0
    rows["fb_pct_d3"] = np.nan
    fb = rows[rows["shape_family"].isin(FASTBALLS)].sort_values("n", ascending=False).drop_duplicates("PitcherId")
    for _, r in fb.iterrows():
        if not (r["perceived_velo"] > 0):
            continue
        pct = float(d3_percentile(r["perceived_velo"], ref))
        target = float(mlb_equivalent(pct, ref))
        mask = rows["PitcherId"] == r["PitcherId"]
        rows.loc[mask, "velo_ratio"] = target / r["perceived_velo"]
        rows.loc[mask, "velo_shift"] = target - r["perceived_velo"]
        rows.loc[mask, "fb_pct_d3"] = pct
    return rows


# Features that scale with arm speed when a pitch is put on its MLB-equivalent footing.
SCALED_WITH_VELO = ("perceived_velo", "spin", "velo_diff")


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Stuff model's view of a D3 pitch: the same pitch thrown at his MLB-equivalent arm speed.

    Everything that scales with arm speed is multiplied by the pitcher's ``velo_ratio``:
    - velocity (so his fastball sits at its D3 percentile on the MLB scale),
    - spin rate (spin rises with arm speed for the same grip and release; spin per mph, the usual
      velocity-free measure, is unchanged, and Oberlin's matches MLB's at about 24-25 rpm/mph on four-seams),
    - velocity separation from his fastball (the same percentage off, which keeps the hitter's relative
      timing gap the same).
    Movement is left as thrown: with spin scaled with speed, Magnus force grows with speed squared while
    flight time shrinks with it, so break stays about the same. Release point is unaffected by speed.
    """
    if "velo_ratio" not in df:
        return df
    out = df.copy()
    k = pd.to_numeric(out["velo_ratio"], errors="coerce").fillna(1.0)
    for c in SCALED_WITH_VELO:
        if c in out:
            out[c] = out[c] * k
    return out


def d3_rate(kind: str, col: str, ref: dict) -> float:
    t = SYNERGY_TYPE.get(kind)
    return ref["rates"].get(t, {}).get(col, np.nan) if t else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.parse_args()
    ref = build()
    print(f"D3 velocity weight: measured {ref['velo_weight_measured']:.2f}, used {ref['velo_weight']:.2f}")
    for p in (10, 50, 90):
        print(f"  D3 {p}th pct {ref['d3_scale_mph'][p - 1]:.1f} mph -> MLB perceived {ref['mlb_perceived_mph'][p - 1]:.1f}")
    for k, v in ref["rates"].items():
        print(f"  {k}: strike {v['strike']:.1%}, whiff/swing {v['whiff']:.1%} ({v['n_teams']} teams)")


if __name__ == "__main__":
    main()
