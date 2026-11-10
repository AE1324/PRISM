"""
Does fastball velocity matter as much, by percentile, in D3 as in MLB?

Compares whiff per swing by velocity percentile:
  - MLB: Statcast fastballs (data/processed/statcast_<year>_shapes.parquet),
    percentile on the MLB pitcher-average fastball scale.
  - D3: Synergy fastball velocity bands (data/synergy/teams_2026.csv),
    percentile on the D3 pitcher scale (data/models/d3_fb_velo_scale.csv).

    python -m prism_model.velo_whiff_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

SWINGS = ["StrikeSwinging", "FoulBall", "InPlay"]


def _logit_slope(x, rate, n) -> float:
    y = np.log(rate / (1 - rate))
    X = np.c_[np.ones(len(x)), x]
    w = np.sqrt(np.asarray(n, float))
    return float(np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0][1] * 10)


def mlb_curve(years=(2024, 2025)) -> pd.DataFrame:
    cols = ["PitcherId", "Season", "RelSpeed", "PitchCall", "shape_family"]
    m = pd.concat([pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols) for y in years])
    fb = m[m["shape_family"].isin(["Four-seam", "Sinker"]) & m["RelSpeed"].notna()].copy()
    fb["RelSpeed"] = fb["RelSpeed"].astype(float)
    per = fb.groupby(["PitcherId", "Season"])["RelSpeed"].agg(["size", "mean"])
    scale = np.sort(per[per["size"] >= 50]["mean"].to_numpy(float))
    fb["pct"] = np.searchsorted(scale, fb["RelSpeed"].to_numpy(float)) / len(scale) * 100
    fb["swing"] = fb["PitchCall"].isin(SWINGS)
    fb["whiff"] = fb["PitchCall"].eq("StrikeSwinging")
    fb["bin"] = pd.cut(fb["pct"], [0, 10, 25, 40, 55, 70, 85, 95, 100.1], right=False)
    out = fb.groupby("bin", observed=True).agg(mph=("RelSpeed", "mean"), pct=("pct", "mean"),
                                               swings=("swing", "sum"), whiffs=("whiff", "sum"))
    out["whiff_per_swing"] = out["whiffs"] / out["swings"]
    return out.reset_index(drop=True)


def d3_curve() -> pd.DataFrame:
    scale = pd.read_csv(DATA_DIR / "models" / "d3_fb_velo_scale.csv")
    t = pd.read_csv(DATA_DIR / "synergy" / "teams_2026.csv")
    b = t[t["cut"].str.startswith("FB")].copy()
    b["pct"] = np.interp(b["avg_velo"], scale["mph"], scale["percentile"], left=0.5, right=99.5)
    b["whiff_per_swing"] = b["whiff"] / b["swing"]
    return b.rename(columns={"avg_velo": "mph", "swing": "swings", "whiff": "whiffs"})


def main() -> None:
    m, d = mlb_curve(), d3_curve()
    print("MLB\n", m.round(3).to_string(index=False))
    print("\nD3\n", d[["team", "cut", "mph", "pct", "swings", "whiff_per_swing"]].round(3).to_string(index=False))
    print(f"\nlogit(whiff/swing) slope per 10 percentile points: MLB {_logit_slope(m.pct, m.whiff_per_swing, m.swings):.3f}, "
          f"D3 pooled {_logit_slope(d.pct, d.whiff_per_swing, d.swings):.3f}")
    for team, g in d.groupby("team"):
        if len(g) >= 3:
            print(f"  {team}: {_logit_slope(g.pct, g.whiff_per_swing, g.swings):.3f}")


if __name__ == "__main__":
    main()
