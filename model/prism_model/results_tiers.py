"""
Results+ tiers: how much is known about each ball in play, and how much the results should count.

Every ball in play is valued with the same 12 categories: batted-ball type (ground ball, line drive,
fly ball, popup) x contact (weak, average, hard), each worth the average MLB run value of that
category. What changes by tier is how a ball gets placed in a category:

  measured   TrackMan exit velocity and launch angle. Type from launch angle (Statcast's bands); contact
             from where the exit velocity ranks among that level's batted balls (thirds within the type,
             after mapping the level's exit velocities onto MLB's by percentile), so a hard-hit ball means
             hard for that level, not 95+ mph.
  charted    a person's call of batted-ball type and weak/average/hard. Trusted as entered: the categories
             are relative, so "hard" sorts itself out by level.
  outcome    only the result (single, out, ...), valued by its MLB average.

Why categories even for measured data: on MLB pitcher seasons, valuing balls by these 12 categories
predicts a pitcher's runs allowed on his other pitches better than exact exit-velocity/launch-angle cells
(5 mph x 10 deg). For pitchers, the fine cells mostly add luck (a 101 mph liner at the shortstop vs in
the gap). Coarser was better every time we tried (3 thirds > 5 > 10 bands; 5x10 cells worst).

How much each tier counts (Pitch+ moves toward Results by n / (n + k), n = game pitches): k is the
number of game pitches at which a pitch type's results predict its runs on the other half of his
pitches as well as they vary (MLB 2024-25 pitcher-season pitch types with 100+ game pitches, random
half splits). Charted is measured with 20% of calls one category off, the rate TrackMan operators'
hand-tagged hit types disagreed with launch angle in the NECBL files.

    python -m prism_model.results_tiers build      # rebuilds data/models/results_tiers.joblib
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

PATH = DATA_DIR / "models" / "results_tiers.joblib"
YEARS = (2024, 2025)

HIT_TYPES = ("GroundBall", "LineDrive", "FlyBall", "Popup")
LA_EDGES = [-np.inf, 10, 25, 50, np.inf]     # Statcast batted-ball type bands (degrees)
CONTACT = ("weak", "average", "hard")
TIERS = ("measured", "charted", "outcome")
# Measured on MLB 2024-25 (see measure_k), averaged over four random splits (measured 399-410, charted
# 434-449, outcome 955-1077) and rounded; a rebuild prints fresh values.
TIER_K = {"measured": 400, "charted": 440, "outcome": 1000}
TIER_LABEL = {"measured": "measured contact (exit velocity + launch angle)",
              "charted": "charted contact (batted-ball type + weak/average/hard)",
              "outcome": "outcomes only (no contact data)"}
CHART_ERROR = 0.20
MIN_LEVEL_BIP = 50       # batted balls needed to place exit velocities within a level

_HIT_ALIASES = {
    "groundball": "GroundBall", "ground": "GroundBall", "gb": "GroundBall", "grounder": "GroundBall",
    "groundout": "GroundBall",
    "linedrive": "LineDrive", "line": "LineDrive", "ld": "LineDrive", "liner": "LineDrive", "lineout": "LineDrive",
    "flyball": "FlyBall", "fly": "FlyBall", "fb": "FlyBall", "flyout": "FlyBall",
    "popup": "Popup", "pop": "Popup", "pu": "Popup", "popfly": "Popup", "popout": "Popup",
}
_CONTACT_ALIASES = {"weak": "weak", "soft": "weak", "w": "weak", "s": "weak", "poor": "weak",
                    "average": "average", "avg": "average", "medium": "average", "med": "average", "m": "average",
                    "a": "average", "solid": "average",
                    "hard": "hard", "h": "hard", "strong": "hard", "barrel": "hard", "barreled": "hard"}


def _key(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalpha())


def normalize_hit_type(s: pd.Series) -> pd.Series:
    return s.map(lambda v: _HIT_ALIASES.get(_key(v)) if pd.notna(v) else None)


def normalize_contact(s: pd.Series) -> pd.Series:
    return s.map(lambda v: _CONTACT_ALIASES.get(_key(v)) if pd.notna(v) else None)


def hit_type_from_angle(la: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(la, errors="coerce"), LA_EDGES, right=False, labels=list(HIT_TYPES)).astype(object)


# --------------------------------------------------------------------------
# Level: exit velocity -> MLB-equivalent by percentile
# --------------------------------------------------------------------------
def level_ev_to_mlb(ev: pd.Series, level_ev: pd.Series, tiers: dict) -> pd.Series:
    """
    Map exit velocities onto MLB's by percentile within the level's own batted balls (``level_ev``: every
    measured ball in play at that level, both teams). Returns NaN when the level sample is too small.
    """
    ev = pd.to_numeric(ev, errors="coerce")
    ref = np.sort(pd.to_numeric(level_ev, errors="coerce").dropna().to_numpy())
    if len(ref) < MIN_LEVEL_BIP:
        return pd.Series(np.nan, index=ev.index)
    pct = (np.searchsorted(ref, ev.to_numpy(), side="left") + np.searchsorted(ref, ev.to_numpy(), side="right")) / (2 * len(ref))
    mlb_q = np.asarray(tiers["mlb_ev_quantiles"])                     # MLB exit velocity at 0..100th percentile
    out = np.interp(pct * 100, np.arange(len(mlb_q)), mlb_q)
    return pd.Series(np.where(ev.notna(), out, np.nan), index=ev.index)


def contact_from_ev(hit_type: pd.Series, ev_mlb: pd.Series, tiers: dict) -> pd.Series:
    cuts = tiers["contact_cutoffs"]                                   # type -> (weak/average, average/hard) mph

    def one(t, e):
        if t not in cuts or pd.isna(e):
            return None
        lo, hi = cuts[t]
        return "weak" if e < lo else "hard" if e >= hi else "average"
    return pd.Series([one(t, e) for t, e in zip(hit_type, ev_mlb)], index=hit_type.index, dtype=object)


# --------------------------------------------------------------------------
# Per pitch: tier and value
# --------------------------------------------------------------------------
def classify_contact(df: pd.DataFrame, tiers: dict) -> pd.DataFrame:
    """
    For each ball in play: ``bip_type``, ``contact`` and ``result_tier`` (measured / charted / outcome).
    Measured needs launch angle plus an MLB-equivalent exit velocity (``ExitSpeed_mlb``, set when the
    data is built from a level's files). Charted needs ``HitType`` and ``Contact`` columns.
    """
    df = df.copy()
    bip = df["PitchCall"].astype(object) == "InPlay"
    la = pd.to_numeric(df["Angle"], errors="coerce") if "Angle" in df else pd.Series(np.nan, index=df.index)
    ev_mlb = pd.to_numeric(df["ExitSpeed_mlb"], errors="coerce") if "ExitSpeed_mlb" in df else pd.Series(np.nan, index=df.index)
    if "Device" in df or "Source" in df:   # only devices that measure contact can be measured (devices.py)
        from prism_data.devices import device_of, profile
        ev_mlb = ev_mlb.where(device_of(df).map(lambda k: profile(k).contact == "measured"))
    m_type = hit_type_from_angle(la)
    m_contact = contact_from_ev(m_type, ev_mlb, tiers)
    measured = bip & m_type.notna() & m_contact.notna()

    c_type = normalize_hit_type(df["HitType"]) if "HitType" in df else pd.Series(None, index=df.index, dtype=object)
    c_contact = normalize_contact(df["Contact"]) if "Contact" in df else pd.Series(None, index=df.index, dtype=object)
    charted = bip & ~measured & c_type.notna() & c_contact.notna()

    df["bip_type"] = np.where(measured, m_type, np.where(charted, c_type, None))
    df["contact"] = np.where(measured, m_contact, np.where(charted, c_contact, None))
    df["result_tier"] = np.where(measured, "measured", np.where(charted, "charted", np.where(bip, "outcome", None)))
    return df


def tiered_values(df: pd.DataFrame, table: dict, tiers: dict) -> pd.Series:
    """
    Run value of each game pitch's result (positive = good for the hitter), using ``classify_contact``'s
    tiers for balls in play. Balls, strikes and fouls are valued as before.
    """
    from prism_model.references import result_values

    out = result_values(df, table)                                    # non-BIP values (+ old BIP values)
    bip = df["result_tier"].notna() if "result_tier" in df else pd.Series(False, index=df.index)
    if not bip.any():
        return out
    count = (df["Balls"].astype("string") + "-" + df["Strikes"].astype("string")).astype(object)
    base = count[bip].map(table["bip_count"]).fillna(table["bip_all"]).astype(float)
    cell = pd.Series([tiers["cells"].get((t, c), np.nan) for t, c in zip(df.loc[bip, "bip_type"], df.loc[bip, "contact"])],
                     index=base.index)
    play = pd.to_numeric(df.loc[bip, "PlayResult"].astype(object).map(table["play"]), errors="coerce")
    out[bip] = base + cell.fillna(play).fillna(0.0)
    return out


def tier_mix(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: balls in play by tier, the weight constant k (blended by those shares) and the main tier."""
    b = df[df["result_tier"].notna()]
    counts = b.groupby(by + ["result_tier"]).size().unstack("result_tier", fill_value=0)
    counts = counts.reindex(columns=list(TIERS), fill_value=0)
    n = counts.sum(axis=1)
    k = sum(counts[t] * TIER_K[t] for t in TIERS) / n
    main = counts.idxmax(axis=1)
    return pd.DataFrame({"results_k": k, "results_tier": main, "bip": n}).reset_index()


# --------------------------------------------------------------------------
# Build from MLB
# --------------------------------------------------------------------------
def _load_mlb() -> pd.DataFrame:
    cols = ["PitcherId", "Season", "shape_family", "PitchCall", "Balls", "Strikes", "ExitSpeed", "Angle",
            "PlayResult", "RunValue"]
    m = pd.concat([pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols) for y in YEARS],
                  ignore_index=True)
    return m[m["shape_family"].notna() & m["PitchCall"].notna()].reset_index(drop=True)


def build() -> dict:
    from joblib import dump

    from prism_model import references

    ref, _ = references.load()
    table = ref["results"]
    m = _load_mlb()
    bip = (m["PitchCall"] == "InPlay") & m["RunValue"].notna()
    count = (m["Balls"].astype("string") + "-" + m["Strikes"].astype("string")).astype(object)
    b = m[bip].dropna(subset=["ExitSpeed", "Angle"]).copy()
    b["resid"] = b["RunValue"] - count[b.index].map(table["bip_count"]).fillna(table["bip_all"]).astype(float)
    b["bip_type"] = hit_type_from_angle(b["Angle"])
    cut = b.groupby("bip_type")["ExitSpeed"].quantile([1 / 3, 2 / 3]).unstack()
    tiers = {"mlb_ev_quantiles": b["ExitSpeed"].quantile(np.linspace(0, 1, 101)).round(2).tolist(),
             "contact_cutoffs": {t: (round(float(cut.at[t, 1 / 3]), 1), round(float(cut.at[t, 2 / 3]), 1)) for t in HIT_TYPES}}
    b["contact"] = contact_from_ev(b["bip_type"], b["ExitSpeed"], tiers)
    tiers["cells"] = {k: round(float(v), 4) for k, v in b.groupby(["bip_type", "contact"])["resid"].mean().items()}
    tiers["k"] = measure_k(m, table, tiers)
    tiers["years"] = list(YEARS)
    dump(tiers, PATH)
    return tiers


def measure_k(m: pd.DataFrame, table: dict, tiers: dict, seed: int = 1) -> dict:
    """Weight constant per tier: per-pitch noise / covariance with runs on the other half of his pitches."""
    rng = np.random.default_rng(seed)
    m = m.copy()
    m["ExitSpeed_mlb"] = m["ExitSpeed"]
    meas = classify_contact(m, tiers)
    m["v_measured"] = tiered_values(meas, table, tiers)
    # charted: the same categories with CHART_ERROR of calls one step off
    ch = meas.copy()
    bipm = ch["result_tier"] == "measured"
    for col, order in (("bip_type", HIT_TYPES), ("contact", CONTACT)):
        idx = ch.loc[bipm, col].map({v: i for i, v in enumerate(order)}).to_numpy()
        flip = rng.random(len(idx)) < CHART_ERROR
        step = np.where(rng.random(len(idx)) < 0.5, -1, 1)
        new = idx + np.where(flip, step, 0)
        new = np.where((new < 0) | (new >= len(order)), idx - step * flip, new)
        ch.loc[bipm, col] = [order[i] for i in new]
    m["v_charted"] = tiered_values(ch, table, tiers)
    oc = meas.assign(bip_type=None, contact=None)
    m["v_outcome"] = tiered_values(oc, table, tiers)

    key = ["PitcherId", "Season", "shape_family"]
    m["half"] = rng.integers(0, 2, len(m))
    n = m.groupby(key).size()
    keep = n[n >= 100].index
    halves = m.groupby(key + ["half"])[["v_measured", "v_charted", "v_outcome"]].mean().unstack("half").reindex(keep)
    w = n.reindex(keep).astype(float).to_numpy()

    def wcov(a, b):
        a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
        return float(np.average(a * b, weights=w))

    out = {}
    for t in TIERS:
        col = f"v_{t}"
        noise = float(m.groupby(key)[col].var().reindex(keep).mean())
        cov = (wcov(halves[(col, 0)].to_numpy(), halves[("v_outcome", 1)].to_numpy())
               + wcov(halves[(col, 1)].to_numpy(), halves[("v_outcome", 0)].to_numpy())) / 2
        out[t] = round(noise / cov)
    return out


_CACHE: dict = {}


def load() -> dict | None:
    """The saved tier tables, or None when they haven't been built (grading then uses the old values)."""
    if "t" not in _CACHE:
        from joblib import load as jl

        _CACHE["t"] = jl(PATH) if PATH.exists() else None
    return _CACHE["t"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build"])
    ap.parse_args()
    t = build()
    print("contact cutoffs (MLB mph, weak|average|hard):", t["contact_cutoffs"])
    print("measured weight constants:", t["k"], " (in use:", TIER_K, ")")
    print(f"saved {PATH}")


if __name__ == "__main__":
    main()
