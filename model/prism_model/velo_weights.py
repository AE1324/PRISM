"""
How much more (or less) velocity matters in D3 than in MLB, by pitch type.

For each pitch type, compares what one more mph is worth in D3 and in MLB, on the same footing:

- D3: Synergy velocity bands (data/synergy/velo/d3_velo_bands_2026.csv), one set per staff. Each band's
  speed is put on its MLB-equivalent scale the same way grades are (the fastball that goes with it ->
  D3 percentile -> MLB perceived velocity at that percentile; a secondary keeps its typical share of the
  fastball's speed). Slopes are fit within staff, so a good staff's results don't pass for velocity.
- MLB: Statcast pitches, slope per mph.

Two outcomes: whiffs per swing (logit) and wOBA per plate appearance ending on the pitch. The weight is
the D3 slope over the MLB slope, averaged across the two outcomes; where MLB whiffs barely move with
velocity (changeups) the whiff ratio is unstable and only wOBA is used. Weights are rounded to 0.05 and
kept between 1 and MAX_VELO_WEIGHT.

    python -m prism_model.velo_weights            # measure and print
    python -m prism_model.velo_weights --boot 400  # with 90% ranges (resampling staffs and MLB pitchers)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

BANDS_PATH = DATA_DIR / "synergy" / "velo" / "d3_velo_bands_2026.csv"
TEAM_RESULTS_PATH = DATA_DIR / "synergy" / "team_results_2026.csv"
TYPES = ["Fastball", "Slider", "Curveball", "Changeup"]
MLB_TYPE = {"Four-seam": "Fastball", "Sinker": "Fastball", "Slider": "Slider", "Curveball": "Curveball",
            "Changeup": "Changeup"}
WOBA = {"bb": 0.69, "hbp": 0.72, "b1": 0.89, "b2": 1.27, "b3": 1.62, "hr": 2.1}
SWINGS = ["StrikeSwinging", "FoulBall", "InPlay"]
MAX_VELO_WEIGHT = 2.0
MIN_MLB_WHIFF_SLOPE = 0.02  # logit per mph; below this the whiff ratio divides by almost nothing
MIN_BANDS = 3


def used_weight(measured: float) -> float:
    return float(min(max(round(measured * 20) / 20, 1.0), MAX_VELO_WEIGHT))


def _fit(y, x, groups=None, weights=None, binomial=False) -> float:
    import statsmodels.api as sm

    x = np.asarray(x, float)
    X = pd.get_dummies(groups, dtype=float) if groups is not None else pd.DataFrame({"const": np.ones(len(x))})
    X = X.reset_index(drop=True)
    X["x"] = x - x.mean()
    if binomial:
        return float(sm.GLM(np.asarray(y, float), X, family=sm.families.Binomial()).fit().params["x"])
    return float(sm.WLS(np.asarray(y, float), X, weights=weights).fit().params["x"])


def load_bands() -> pd.DataFrame:
    b = pd.read_csv(BANDS_PATH)
    b["grp"] = b["team"] + "|" + b["matchup"]
    return b


def type_ratios() -> dict:
    """Typical D3 speed of each pitch type as a share of the same staff's fastball."""
    t = pd.read_csv(TEAM_RESULTS_PATH).pivot_table(index="team", columns="pitch_type", values="avg_velo", aggfunc="first")
    return {"Fastball": 1.0, **{k: float((t[k] / t["Fastball"]).median()) for k in TYPES[1:]}}


def mlb_equiv(velo, kind: str, ref: dict, ratios: dict):
    from prism_model import d3

    r = ratios[kind]
    return r * d3.mlb_equivalent(d3.d3_percentile(np.asarray(velo, float) / r, ref), ref)


def d3_slopes(bands: pd.DataFrame, kind: str, ref: dict, ratios: dict) -> tuple[float, float]:
    """(logit whiff/swing, wOBA) per MLB-equivalent mph, within staff."""
    g = bands[bands["pitch_type"] == kind]
    g = g.groupby("grp").filter(lambda x: len(x) >= MIN_BANDS)
    x = mlb_equiv(g["avg_velo"], kind, ref, ratios)
    whiff = _fit(np.c_[g["miss"], g["swing"] - g["miss"]], x, g["grp"], binomial=True)
    woba = sum(g[k] * v for k, v in WOBA.items()) / g["pa"]
    return whiff, _fit(woba, x, g["grp"], weights=g["pa"])


def load_mlb(years=(2024, 2025)) -> pd.DataFrame:
    cols = ["PitcherId", "RelSpeed", "PitchCall", "PlayResult", "KorBB", "shape_family"]
    m = pd.concat([pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols) for y in years])
    m = m[m["RelSpeed"].notna()].copy()
    m["RelSpeed"] = m["RelSpeed"].astype(float)
    m["type"] = m["shape_family"].map(MLB_TYPE)
    m = m[m["type"].notna()]
    m["swing"] = m["PitchCall"].isin(SWINGS)
    m["whiff"] = m["PitchCall"].eq("StrikeSwinging")
    m["pa_end"] = m["PitchCall"].isin(["InPlay", "HitByPitch"]) | m["KorBB"].isin(["Strikeout", "Walk"])
    m["woba"] = m["PlayResult"].map({"Single": 0.89, "Double": 1.27, "Triple": 1.62, "HomeRun": 2.1}).fillna(0.0)
    m.loc[m["KorBB"].eq("Walk"), "woba"] = WOBA["bb"]
    m.loc[m["PitchCall"].eq("HitByPitch"), "woba"] = WOBA["hbp"]
    return m


def mlb_slopes(m: pd.DataFrame) -> tuple[float, float]:
    s, p = m[m["swing"]], m[m["pa_end"]]
    return (_fit(s["whiff"].astype(float), s["RelSpeed"], binomial=True),
            _fit(p["woba"], p["RelSpeed"], weights=np.ones(len(p))))


def _combine(d3s, mlbs) -> tuple[float, float, float]:
    whiff = d3s[0] / mlbs[0] if abs(mlbs[0]) >= MIN_MLB_WHIFF_SLOPE else np.nan
    woba = d3s[1] / mlbs[1]
    return whiff, woba, float(np.nanmean([whiff, woba]))


def measure(ref: dict, boot: int = 0, seed: int = 1) -> dict:
    """Per pitch type: whiff and wOBA ratios, their average (``measured``) and the weight ``used``."""
    bands, ratios, mlb = load_bands(), type_ratios(), load_mlb()
    rng = np.random.default_rng(seed)
    out = {}
    for kind in TYPES:
        mm = mlb[mlb["type"] == kind]
        whiff, woba, measured = _combine(d3_slopes(bands, kind, ref, ratios), mlb_slopes(mm))
        g = bands[bands["pitch_type"] == kind]
        staffs = [k for k, x in g.groupby("grp") if len(x) >= MIN_BANDS]
        row = {"whiff_ratio": whiff, "woba_ratio": woba, "measured": measured, "used": used_weight(measured),
               "staffs": len(staffs), "pitches": int(g[g["grp"].isin(staffs)]["pitches"].sum())}
        if boot:
            idx, pids = mm.groupby("PitcherId").indices, mm["PitcherId"].unique()
            mb = [mlb_slopes(mm.iloc[np.concatenate([idx[p] for p in rng.choice(pids, len(pids))])])
                  for _ in range(max(boot // 10, 20))]
            draws = []
            for _ in range(boot):
                pick = pd.concat([g[g["grp"] == k].assign(grp=f"{k}#{i}")
                                  for i, k in enumerate(rng.choice(staffs, len(staffs)))])
                draws.append(_combine(d3_slopes(pick, kind, ref, ratios), mb[rng.integers(len(mb))])[2])
            row["lo"], row["hi"] = (float(v) for v in measured + np.percentile(np.array(draws) - np.mean(draws), [5, 95]))
        out[kind] = row
    return out


def main() -> None:
    from prism_model import d3

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    for kind, r in measure(d3.load(), a.boot).items():
        rng = f"  90% range {r['lo']:.2f} to {r['hi']:.2f}" if "lo" in r else ""
        print(f"{kind:10s} whiff {r['whiff_ratio']:.2f}  wOBA {r['woba_ratio']:.2f}  average {r['measured']:.2f}"
              f"  -> used {r['used']:.2f}  ({r['staffs']} staffs, {r['pitches']:,} pitches){rng}")


if __name__ == "__main__":
    main()
