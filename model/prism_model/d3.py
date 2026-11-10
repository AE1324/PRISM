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
    Move every pitch's perceived velocity by the gap between the pitcher's main
    fastball and the MLB fastball at the same D3 percentile. Adds ``velo_shift``,
    ``fb_pct_d3`` and keeps the original as ``perceived_velo_d3``.
    """
    rows = rows.copy()
    rows["perceived_velo_d3"] = rows["perceived_velo"]
    rows["velo_shift"] = 0.0
    rows["fb_pct_d3"] = np.nan
    fb = rows[rows["shape_family"].isin(FASTBALLS)].sort_values("n", ascending=False).drop_duplicates("PitcherId")
    for _, r in fb.iterrows():
        pct = float(d3_percentile(r["perceived_velo"], ref))
        shift = float(mlb_equivalent(pct, ref)) - r["perceived_velo"]
        mask = rows["PitcherId"] == r["PitcherId"]
        rows.loc[mask, "velo_shift"] = shift
        rows.loc[mask, "fb_pct_d3"] = pct
    rows["perceived_velo"] = rows["perceived_velo"] + rows["velo_shift"]
    return rows


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
