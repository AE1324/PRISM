"""
Build the D3 fastball velocity scale (pitcher-level percentiles) from the
Synergy team pulls in data/synergy/velo_teams_2026.csv and team strength in
data/synergy_team_strength_2026.csv. Both inputs are licensed/derived data and
stay in the gitignored data/ folder; this script only holds the method.

Method (agreed Nov 2026):
  1. Team fastball average per team; drop broken team readings (avg < 76).
  2. Shrink each team toward its strength-predicted average with weight n/(n+200).
  3. Re-weight teams so strength quintiles match all of D3.
  4. Pitcher average = team average - 0.4 (pitcher-count vs pitch-count)
     + N(0, 2.9) within-staff spread (calibrated to observed team max velo).
Output: data/models/d3_fb_velo_scale.csv (percentile, mph).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR

WITHIN_STAFF_SD = 2.9
PITCHER_SHIFT = -0.4
SHRINK_K = 200
MIN_TEAM_AVG = 76.0


def build(seed: int = 0, n: int = 400_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = pd.read_csv(DATA_DIR / "synergy" / "velo_teams_2026.csv")
    s = pd.read_csv(DATA_DIR / "synergy_team_strength_2026.csv")
    fb = v[(v.pitch_type == "Fastball") & v.strength_pct.notna() & (v.n_velo >= 50) & (v.avg >= MIN_TEAM_AVG)].copy()
    w = np.c_[np.ones(len(fb)), fb.strength_pct / 100]
    wt = fb.n_velo / (fb.n_velo + 100)
    beta = np.linalg.lstsq(w * np.sqrt(wt.values)[:, None], fb.avg * np.sqrt(wt), rcond=None)[0]
    pred = beta[0] + beta[1] * fb.strength_pct / 100
    fb["team_avg"] = (fb.n_velo * fb.avg + SHRINK_K * pred) / (fb.n_velo + SHRINK_K)
    bins = np.linspace(0, 100.01, 6)
    fb["bin"] = np.digitize(fb.strength_pct, bins)
    share = pd.Series(np.digitize(s.strength_pct, bins)).value_counts(normalize=True)
    fb["w"] = fb.bin.map(share / fb.bin.value_counts(normalize=True))
    idx = rng.choice(len(fb), n, p=fb.w / fb.w.sum())
    p = fb.team_avg.values[idx] + PITCHER_SHIFT + rng.normal(0, WITHIN_STAFF_SD, n)
    pct = np.arange(1, 100)
    return pd.DataFrame({"percentile": pct, "mph": np.percentile(p, pct).round(2)})


def main() -> None:
    out = build()
    path = DATA_DIR / "models" / "d3_fb_velo_scale.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out[out.percentile.isin([5, 10, 25, 50, 75, 90, 95, 99])].to_string(index=False))


if __name__ == "__main__":
    main()
