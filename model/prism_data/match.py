"""
Match AWRE results to Rapsodo pitches.

Both logs come from the same pitches, but each keeps its own record: Rapsodo
records the pitch when it's thrown; AWRE stores the time the result was
entered, a few seconds later, and its velocity is Rapsodo's, rounded. Rapsodo
also records pitches AWRE doesn't (warm-ups between innings), and now and then
AWRE logs a pitch Rapsodo missed.

For each pitcher and date the two lists are aligned in time order, the way
sequences are aligned: a pair scores well when the entry delay looks normal
and the velocities agree, and either list may skip pitches. Order is never
crossed, so one bad reading can't scramble the rest.

Each match gets a confidence:
  strong     velocities agree (within 1 mph) and the delay is normal
  time-only  AWRE has no velocity; matched on timing and order
  check      matched, but velocity or timing disagrees; listed for review
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prism_data.rapsodo import clean_name

MIN_LAG_S, MAX_LAG_S = -75, 60       # AWRE time minus Rapsodo time (the clocks differ by a few seconds)
TYPICAL_LAG_S = -8
VELO_AGREE_MPH = 1.0
RESULT_COLUMNS = ["Balls", "Strikes", "PitchCall", "PlayResult", "KorBB", "Batter", "BatterId", "BatterSide",
                  "GameID", "PitchNo"]


def _pair_score(lag, va, vr, typical):
    if lag < MIN_LAG_S or lag > MAX_LAG_S:
        return -np.inf
    s = 2.0 - abs(lag - typical) / 20.0
    if not np.isnan(va) and not np.isnan(vr):
        d = abs(va - vr)
        s += 2.0 if d <= VELO_AGREE_MPH else 0.5 if d <= 2.0 else -3.0
    return s


def align(a_times, a_velo, r_times, r_velo, typical=TYPICAL_LAG_S):
    """Order-preserving best alignment. Returns a list of (i, j) index pairs."""
    n, m = len(a_times), len(r_times)
    S = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0 diag, 1 skip AWRE, 2 skip Rapsodo
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = _pair_score(a_times[i - 1] - r_times[j - 1], a_velo[i - 1], r_velo[j - 1], typical)
            best, mv = S[i - 1, j], 1
            if S[i, j - 1] > best:
                best, mv = S[i, j - 1], 2
            if sc > 0 and S[i - 1, j - 1] + sc > best:
                best, mv = S[i - 1, j - 1] + sc, 0
            S[i, j], move[i, j] = best, mv
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if move[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def match(awre: pd.DataFrame, rap: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pair, with AWRE and Rapsodo indexes, delay, velocities and confidence.
    Unmatched pitches on either side appear with the other index missing.
    """
    awre = awre.assign(_pid=awre["PitcherId"].map(clean_name))
    rap = rap.assign(_pid=rap["PitcherId"].map(clean_name))
    rows = []
    for (pid, day), ga in awre.groupby(["_pid", "Date"]):
        gr = rap[(rap["_pid"] == pid) & (rap["Date"] == day)]
        ga = ga.sort_values("PitchTime")
        gr = gr.sort_values("PitchTime")
        at = ga["PitchTime"].astype("int64").to_numpy() / 1e9
        rt = gr["PitchTime"].astype("int64").to_numpy() / 1e9
        av = ga["RelSpeed"].astype(float).to_numpy()
        rv = gr["RelSpeed"].astype(float).to_numpy()
        pairs = align(at, av, rt, rv)
        if len(pairs) >= 5:                       # second pass with this session's own typical delay
            typical = float(np.median([at[i] - rt[j] for i, j in pairs]))
            pairs = align(at, av, rt, rv, typical)
        else:
            typical = TYPICAL_LAG_S
        hit_a = {i for i, _ in pairs}
        hit_r = {j for _, j in pairs}
        for i, j in pairs:
            lag = at[i] - rt[j]
            d = abs(av[i] - rv[j]) if not np.isnan(av[i]) and not np.isnan(rv[j]) else np.nan
            if np.isnan(d):
                conf = "time-only" if abs(lag - typical) <= 30 else "check"
            else:
                conf = "strong" if d <= VELO_AGREE_MPH and abs(lag - typical) <= 30 else "check"
            rows.append({"pitcher": pid, "date": day, "awre_idx": ga.index[i], "rap_idx": gr.index[j],
                         "lag_s": lag, "awre_velo": av[i], "rap_velo": rv[j], "confidence": conf})
        for i in range(len(ga)):
            if i not in hit_a:
                rows.append({"pitcher": pid, "date": day, "awre_idx": ga.index[i], "rap_idx": None,
                             "awre_velo": av[i], "confidence": "awre-only"})
    out = pd.DataFrame(rows)
    matched_r = set(out["rap_idx"].dropna().astype(int))
    extra = rap[~rap.index.isin(matched_r)]
    out = pd.concat([out, pd.DataFrame({"pitcher": extra["_pid"], "date": extra["Date"], "awre_idx": None,
                                        "rap_idx": extra.index, "rap_velo": extra["RelSpeed"],
                                        "confidence": "rapsodo-only"})], ignore_index=True)
    return out


def merge(awre: pd.DataFrame, rap: pd.DataFrame, pairs: pd.DataFrame, include_unmatched_rapsodo: bool = True,
          drop_check: bool = False) -> pd.DataFrame:
    """
    Rapsodo pitches with AWRE's count and result attached where matched.
    Velocity stays Rapsodo's (it's the source of AWRE's). Unmatched Rapsodo
    pitches (warm-ups, untracked) keep their shape but no result.
    """
    m = pairs[pairs["awre_idx"].notna() & pairs["rap_idx"].notna()]
    if drop_check:
        m = m[m["confidence"] != "check"]
    out = rap.copy()
    out["match_confidence"] = "rapsodo-only"
    ai, ri = m["awre_idx"].astype(int).to_numpy(), m["rap_idx"].astype(int).to_numpy()
    for col in RESULT_COLUMNS:
        if col in awre:
            out.loc[ri, col] = awre.loc[ai, col].to_numpy()
    out.loc[ri, "match_confidence"] = m["confidence"].to_numpy()
    out.loc[ri, "awre_pitch_type"] = awre.loc[ai, "TaggedPitchType"].to_numpy()
    if not include_unmatched_rapsodo:
        out = out[out["match_confidence"] != "rapsodo-only"]
    return out
