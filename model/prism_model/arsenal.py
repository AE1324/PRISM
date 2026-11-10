"""
Full-arsenal reports: work through each pitcher's whole arsenal the way the
Garnet Pitch Lab case study does.

For every pitcher in a file this produces:

  Grades       Pitch+ for each pitch, built from Stuff (shape and speed),
               Location (competitive zone minus middle) and Results (run value
               of what actually happened), with plain-language explanations
               ("Velo +9, Movement +3") and comparable MLB pitches.
  Jobs         what each pitch does clearly better than its peers (strike
               pitch, whiff pitch, weak contact, ...) and the two deal-breakers
               (not landing strikes, getting hit hard).
  Verdicts     working / struggling / no clear job yet, with a one-line "So".
               Outcomes come first: a pitch that is working is never told to
               change; shape ideas are offered only as add-ons.
  Development  stuff, how the pitches work together, command, what makes him,
               what's holding him back, and a priority order.
  Situations   first-pitch strikes, results by count, lefties vs righties,
               and whether he fades after pitch 25.
  Progress     each pitch's shape and Stuff+ outing by outing (N/A under 5 pitches).

Scales: 100 = average for that pitch type among the pitchers in the file (the
staff, when the file is a team's own data). When too few pitchers in the file
throw a pitch type, the comparison widens to the pitch group, then to everyone,
then to MLB; the report says which.

Usage (from the model/ folder):
    python -m prism_model.arsenal path/to/trackman.csv [more.csv ...]
    python -m prism_model.arsenal path.csv --team OBE_YEO --scale mlb
Reports are written to data/reports/<file name>/ as JSON and Markdown.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR
from prism_model import d3, fastball_traits, references
from prism_model.references import RATE_COLUMNS, add_flags, num, location_score, result_values, results_score
from prism_model.shapes import BUCKET
from prism_model.stuff import GROUPS, aggregate, predict, to_plus

REPORTS_DIR = DATA_DIR / "reports"

MIN_TYPE_PITCHES = 10       # a pitch type needs this many to be graded
MIN_LOCATED = 10            # located pitches needed for a Location grade
MIN_SESSION_PITCHES = 5     # per outing, for Progress
MIN_PEERS = 4               # pitch types needed to set a 100 scale
RESULTS_K = 15              # Pitch+ trusts Results n / (n + 15)
FADE_AFTER = 25
MIN_GAME_FOR_VERDICT = 15
CLIP = (40.0, 160.0)
D3_RATES = ["strike", "whiff"]  # rates with a D3 reference from Synergy
OPTIONAL_COLUMNS = ["ExitSpeed", "Angle", "PitcherTeam", "TaggedHitType", "PlayResult", "KorBB"]

COMPONENTS = {
    "Velo": ["perceived_velo"],
    "Movement": ["ivb", "hb"],
    "Spin": ["spin"],
    "Release": ["rel_height", "rel_side"],
    "Separation": ["velo_diff", "ivb_diff", "hb_diff"],
}

# Minimum denominators before a rate can be called clearly above or below.
RATE_MIN_N = {"strike": 15, "called_strike": 15, "zone": 15, "whiff": 8, "chase": 15, "hard": 8, "gb": 8, "putaway": 8}
RATE_LABEL = {
    "strike": "strike rate", "called_strike": "called-strike rate", "zone": "zone rate", "whiff": "whiff rate",
    "chase": "chase rate", "hard": "hard-hit rate", "gb": "ground-ball rate", "putaway": "putaway rate",
}
# (rate, direction that is good, job name, what it does)
JOBS = [
    ("strike", +1, "Strike pitch", "lands strikes"),
    ("whiff", +1, "Whiff pitch", "gets swings and misses"),
    ("called_strike", +1, "Called-strike pitch", "steals called strikes"),
    ("chase", +1, "Chase pitch", "gets chases out of the zone"),
    ("hard", -1, "Weak-contact pitch", "limits hard contact"),
    ("gb", +1, "Ground-ball pitch", "keeps the ball on the ground"),
    ("putaway", +1, "Putaway pitch", "finishes hitters with two strikes"),
]
DEAL_BREAKERS = {"strike": (-1, "Not landing strikes"), "hard": (+1, "Getting hit hard")}

COUNT_GROUPS = [
    ("First pitch (0-0)", {"0-0"}),
    ("Ahead (0-1, 0-2, 1-2)", {"0-1", "0-2", "1-2"}),
    ("Even (1-1, 2-2)", {"1-1", "2-2"}),
    ("Behind (1-0, 2-0, 2-1, 3-0, 3-1)", {"1-0", "2-0", "2-1", "3-0", "3-1"}),
    ("Two strikes", {"0-2", "1-2", "2-2", "3-2"}),
    ("Full (3-2)", {"3-2"}),
]
HITS = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _num(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return None
    return int(round(float(x))) if digits == 0 else round(float(x), digits)


def _pct(x):
    return None if x is None or pd.isna(x) else round(100 * float(x), 1)


def display_name(name) -> str:
    """'Last, First' -> 'First Last'."""
    if not isinstance(name, str) or name.strip().isdigit():
        return f"Pitcher {name}"
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{first} {last}".strip()
    return name


def pitch_plus(stuff, location, results, n_game: int, k: float = RESULTS_K):
    """
    Garnet's Pitch+: the projection (mean of Stuff and Location) moved toward
    Results by n / (n + k), where n is game pitches.
    Example from the case study: Stuff 99, Location 95, Results 88, n 41 -> 90.
    """
    parts = [v for v in (stuff, location) if v is not None and not pd.isna(v)]
    if not parts:
        return None
    proj = float(np.mean(parts))
    if results is None or pd.isna(results) or not n_game:
        return proj
    w = n_game / (n_game + k)
    return proj + w * (float(results) - proj)


def clearly(p, p0, n, min_n) -> int:
    """+1 / -1 if rate ``p`` is clearly above / below reference ``p0``; else 0."""
    if p is None or p0 is None or pd.isna(p) or pd.isna(p0) or n < min_n:
        return 0
    if abs(p - p0) < 0.03:
        return 0
    se = np.sqrt(max(p0 * (1 - p0), 0.01) / n)
    z = (p - p0) / se
    return 1 if z >= 1.28 else -1 if z <= -1.28 else 0


def staff_rescale(rows: pd.DataFrame, col: str, min_peers: int = MIN_PEERS) -> pd.DataFrame:
    """
    Re-center an MLB-scale grade on the pitchers in this file.

    Peers are the same pitch type if enough pitchers throw it, otherwise the
    same group (fastball/breaking/offspeed), otherwise every graded pitch.
    With too few of any, the MLB scale is kept. Returns plus, basis, and the
    peer mean and SD (so outing grades can be put on the same scale).
    """
    out = pd.DataFrame({"plus": np.nan, "basis": None, "mean": np.nan, "sd": np.nan}, index=rows.index)
    valid = rows[col].notna()
    for idx, r in rows[valid].iterrows():
        for basis, mask in (("type", rows["shape_family"] == r["shape_family"]),
                            ("group", rows["bucket"] == r["bucket"]),
                            ("all", pd.Series(True, index=rows.index))):
            peers = rows.loc[valid & mask, col]
            if len(peers) >= min_peers and peers.std() > 0:
                m, s = peers.mean(), peers.std()
                out.loc[idx, ["plus", "basis", "mean", "sd"]] = [100 + 10 * (r[col] - m) / s, basis, m, s]
                break
        else:
            out.loc[idx, ["plus", "basis", "mean", "sd"]] = [r[col], "mlb", 100.0, 10.0]
    out["plus"] = out["plus"].astype(float).clip(*CLIP)
    return out


# --------------------------------------------------------------------------
# Preparing pitches
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Add result values, per-pitch flags, outing keys and pitch number within the outing."""
    df = df.copy()
    for col in OPTIONAL_COLUMNS:  # e.g. Rapsodo + AWRE data has no exit velocity
        if col not in df:
            df[col] = np.nan
    for col in ("Balls", "Strikes"):  # counts stored as floats would read "0.0-0.0"
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    df["result_rv"] = result_values(df, ref["results"])
    df = add_flags(df, ref.get("hard_hit_mph", references.HARD_HIT_MPH))
    df["outing"] = df["Date"].astype("string").fillna("unknown")
    order = num(df["PitchNo"]) if df["PitchNo"].notna().any() else pd.Series(np.arange(len(df)), index=df.index)
    df["_order"] = order.fillna(pd.Series(np.arange(len(df)), index=df.index).astype(float))
    df["pitch_in_outing"] = df.groupby(["PitcherId", "outing"])["_order"].rank(method="first")
    return df.drop(columns="_order")


# --------------------------------------------------------------------------
# Grades
# --------------------------------------------------------------------------
def d3_stuff(rows: pd.DataFrame, models: dict | None, d3ref: dict | None,
             shifts: dict | None = None) -> pd.DataFrame:
    """
    Predict Stuff/Shape xRV. With a D3 reference, velocities are first moved to their MLB
    equivalent by D3 percentile (``d3.translate``; or by the given per-pitcher ``shifts``),
    and the velocity part of fastball Stuff is weighted by the D3 velocity weight.
    """
    if d3ref is not None:
        if shifts is None:
            rows = d3.translate(rows, d3ref)
        else:
            rows = rows.copy()
            rows["perceived_velo_d3"] = rows["perceived_velo"]
            rows["velo_shift"] = rows["PitcherId"].map(shifts).fillna(0.0)
            rows["perceived_velo"] = rows["perceived_velo"] + rows["velo_shift"]
    rows = predict(rows, models)
    if d3ref is not None:
        fb = rows["group"] == "Fastball"
        w = d3ref["velo_weight"]
        rows.loc[fb, "stuff_xrv"] = rows.loc[fb, "shape_xrv"] + w * (rows.loc[fb, "stuff_xrv"] - rows.loc[fb, "shape_xrv"])
    return rows


def grade_rows(df: pd.DataFrame, ref: dict, models: dict | None, scale: str = "d3",
               d3ref: dict | None = None) -> pd.DataFrame:
    """One row per pitcher pitch type with Stuff, Shape, Location, Results and Pitch+."""
    by = ["PitcherId", "shape_family"]
    rows = aggregate(df, by=by, min_pitches=1)
    rows["bucket"] = rows["shape_family"].map(BUCKET)
    rows = d3_stuff(rows, models, d3ref if scale == "d3" else None)
    rows = rows.merge(location_score(df, by), on=by, how="left").merge(results_score(df, by), on=by, how="left")
    rows["n_located"] = rows["n_located"].fillna(0).astype(int)
    rows["n_game"] = rows["n_game"].fillna(0).astype(int)
    rows["usage"] = rows["n"] / rows.groupby("PitcherId")["n"].transform("sum")

    sc = ref["scale"]

    def mlb(col, key, higher_better):
        out = pd.Series(np.nan, index=rows.index)
        for kind, grp in rows.groupby("shape_family"):
            s = sc.get(kind)
            if not s or not s.get(f"{key}_std"):
                continue
            z = (grp[col] - s[f"{key}_mean"]) / s[f"{key}_std"]
            out.loc[grp.index] = 100 + 10 * z if higher_better else 100 - 10 * z
        return out.clip(*CLIP)

    graded = rows["n"] >= MIN_TYPE_PITCHES
    rows["stuff_mlb"] = mlb("stuff_xrv", "stuff_xrv", False).where(graded)
    rows["shape_mlb"] = mlb("shape_xrv", "shape_xrv", False).where(graded)
    rows["location_mlb"] = mlb("location", "location", True).where(graded & (rows["n_located"] >= MIN_LOCATED))
    rows["results_mlb"] = mlb("results_rv100", "results_rv100", False).where(graded & (rows["n_game"] > 0))

    for comp in ["stuff", "shape", "location", "results"]:
        if scale in ("mlb", "d3"):
            rows[comp] = rows[f"{comp}_mlb"]
            rows[f"{comp}_basis"], rows[f"{comp}_mean"], rows[f"{comp}_sd"] = "mlb", 100.0, 10.0
        else:
            r = staff_rescale(rows, f"{comp}_mlb")
            rows[comp], rows[f"{comp}_basis"], rows[f"{comp}_mean"], rows[f"{comp}_sd"] = r["plus"], r["basis"], r["mean"], r["sd"]

    rows["pitch_plus"] = [
        pitch_plus(s, l, r, g) if n >= MIN_TYPE_PITCHES else None
        for s, l, r, g, n in zip(rows["stuff"], rows["location"], rows["results"], rows["n_game"], rows["n"])
    ]
    rows["pitch_plus"] = rows["pitch_plus"].astype(float)

    # Where each pitch ranks on the staff among the same pitch type.
    for comp in ["pitch_plus", "stuff"]:
        rows[f"{comp}_rank"] = rows.groupby("shape_family")[comp].rank(ascending=False, method="min")
        rows[f"{comp}_of"] = rows.groupby("shape_family")[comp].transform("count")
    return rows


def explain_stuff(rows: pd.DataFrame, ref: dict, models: dict | None) -> dict:
    """
    Why each Stuff grade is what it is: swap one part of the pitch for the
    reference average and see how many points the grade moves.
    """
    if models is None:
        from joblib import load
        from prism_model.stuff import MODELS_PATH

        models = load(MODELS_PATH)
    out = {}
    graded = rows[rows["stuff"].notna()]
    for idx, r in graded.iterrows():
        m = models.get(r["group"])
        s = ref["scale"].get(r["shape_family"], {})
        if not m or not s.get("stuff_xrv_std"):
            continue
        feats = m["features"]
        x = r[feats].astype(float).to_frame().T
        if r["stuff_basis"] == "type":
            peers = graded[graded["shape_family"] == r["shape_family"]]
            avg = peers[feats].mean()
        else:
            avg = pd.Series(ref["features"].get(r["shape_family"], {}))
        base = float(m["stuff"].predict(x)[0])
        parts = {}
        for name, cols in COMPONENTS.items():
            cols = [c for c in cols if c in feats and c in avg and pd.notna(avg.get(c))]
            if not cols:
                continue
            x2 = x.copy()
            x2[cols] = avg[cols].to_numpy()
            alt = float(m["stuff"].predict(x2)[0])
            mlb_pts = 10 * (alt - base) / s["stuff_xrv_std"]
            parts[name] = round(mlb_pts * 10 / r["stuff_sd"], 0)
        out[idx] = parts
    return out


def comparable_pitches(r: pd.Series, mlb_rows: pd.DataFrame) -> dict | None:
    """MLB pitcher-seasons with a similar pitch, and how they did versus the average of that pitch type."""
    pool = mlb_rows[mlb_rows["shape_family"] == r["shape_family"]].dropna(subset=["xrv100"])
    if pool.empty or pd.isna(r["velo"]):
        return None
    shift = r.get("velo_shift", 0.0)
    velo = r["velo"] + (shift if pd.notna(shift) else 0.0)  # D3-equivalent MLB speed when translated
    type_avg = np.average(pool["xrv100"], weights=pool["n"])
    for dv, dm in ((1.5, 2.0), (2.5, 3.0), (4.0, 4.0)):
        near = pool[((pool["velo"] - velo).abs() <= dv) & ((pool["ivb"] - r["ivb"]).abs() <= dm)
                    & ((pool["hb"] - r["hb"]).abs() <= dm)]
        if len(near) >= 25:
            break
    if len(near) < 10:
        return {"n": int(len(near)), "text": f"Too few MLB {r['shape_family'].lower()}s with this speed and shape to compare."}
    diff = np.average(near["xrv100"], weights=near["n"]) - type_avg
    word = "fewer" if diff < 0 else "more"
    return {
        "n": int(len(near)),
        "pitches": int(near["n"].sum()),
        "window": f"±{dv:g} mph, ±{dm:g} in",
        "runs_per_100_vs_avg": round(float(diff), 2),
        "mlb_equivalent_velo": round(float(velo), 1),
        "text": (f"{len(near)} comparable MLB {r['shape_family'].lower()}s "
                 + (f"(around {velo:.0f} mph, his D3-equivalent speed, " if abs(velo - r['velo']) >= 0.5 else "(")
                 + f"±{dv:g} mph, ±{dm:g} in of movement) "
                 f"allowed {abs(diff):.2f} {word} runs per 100 pitches than the average MLB {r['shape_family'].lower()}."),
    }


# --------------------------------------------------------------------------
# Rates, jobs and verdicts
# --------------------------------------------------------------------------
def rate_table(df: pd.DataFrame, ref: dict, d3ref: dict | None = None) -> pd.DataFrame:
    """
    Each pitcher pitch type's rates, next to a reference rate. With a D3
    reference, strike% and whiff% compare to D3 averages for the pitch type
    (Synergy); other rates compare to the same pitch type thrown by the other
    pitchers in the file (if at least two others throw it), otherwise MLB.
    """
    by = ["PitcherId", "shape_family"]
    rates = df.groupby(by)[RATE_COLUMNS].agg(["mean", "count"])
    rates.columns = [f"{a}_{'rate' if b == 'mean' else 'n'}" for a, b in rates.columns]
    rates = rates.reset_index()

    sums = df.groupby(by)[RATE_COLUMNS].sum()
    counts = df.groupby(by)[RATE_COLUMNS].count()
    n_by = df.groupby(by).size()
    for c in RATE_COLUMNS:
        rates[f"{c}_ref"] = np.nan
    rates["ref_basis"] = "MLB"
    for i, r in rates.iterrows():
        kind = r["shape_family"]
        others = [k for k in n_by.index if k[1] == kind and k[0] != r["PitcherId"] and n_by[k] >= MIN_TYPE_PITCHES]
        mlb = ref["rates"].get(kind, {})
        if len(others) >= 2:
            s, c = sums.loc[others].sum(), counts.loc[others].sum()
            for col in RATE_COLUMNS:
                rates.loc[i, f"{col}_ref"] = s[col] / c[col] if c[col] >= 20 else mlb.get(col, np.nan)
            rates.loc[i, "ref_basis"] = "staff"
        else:
            for col in RATE_COLUMNS:
                rates.loc[i, f"{col}_ref"] = mlb.get(col, np.nan)
        if d3ref is not None:
            for col in D3_RATES:
                v = d3.d3_rate(kind, col, d3ref)
                if pd.notna(v):
                    rates.loc[i, f"{col}_ref"] = v
                    rates.loc[i, f"{col}_basis"] = "D3"
    return rates


def judge_pitch(row: pd.Series, rates: pd.Series) -> dict:
    """Jobs, deal-breakers, strengths and weaknesses for one pitch, plus its verdict and So line."""
    kind = row["shape_family"].lower()
    ref_word = f"{'staff' if rates['ref_basis'] == 'staff' else 'MLB'} {kind}s"

    def line(col):
        word = f"D3 {kind}s" if rates.get(f"{col}_basis") == "D3" else ref_word
        return f"{RATE_LABEL[col].capitalize()} {_pct(rates[f'{col}_rate']):.0f}% vs {_pct(rates[f'{col}_ref']):.0f}% for {word}"

    calls = {c: clearly(rates[f"{c}_rate"], rates[f"{c}_ref"], rates[f"{c}_n"], RATE_MIN_N[c]) for c in RATE_COLUMNS}
    jobs = [{"job": name, "does": does, "evidence": line(c)} for c, good, name, does in JOBS if calls[c] == good]
    breakers = [{"issue": label, "evidence": line(c)} for c, (bad, label) in DEAL_BREAKERS.items() if calls[c] == bad]
    good_dirs = {c: g for c, g, _, _ in JOBS}
    great = [line(c) for c in RATE_COLUMNS if c in good_dirs and calls[c] == good_dirs[c]]
    weak = [line(c) for c in RATE_COLUMNS if c in good_dirs and calls[c] == -good_dirs[c] and c not in DEAL_BREAKERS]

    n_game = int(row["n_game"])
    pp = row["pitch_plus"]
    if n_game == 0:
        verdict = "Bullpen only"
        so = "Graded on shape and location only; game results will show what job it can do."
    elif n_game < MIN_GAME_FOR_VERDICT:
        verdict = "Too few game pitches"
        so = f"Only {n_game} game pitches so far; keep throwing it before judging results."
    elif breakers:
        verdict = "Struggling"
        b = breakers[0]["issue"]
        if b == "Not landing strikes":
            so = f"Get it in the zone more often: {breakers[0]['evidence'].lower()}."
        else:
            mid = _pct(row.get("middle"))
            so = (f"It's getting hit hard; keep it out of the middle of the zone"
                  + (f" ({mid:.0f}% of these were middle-middle)." if mid is not None else "."))
        if jobs:
            so += f" It still {jobs[0]['does']}, so the pitch has a role once that's fixed."
    elif jobs:
        verdict = "Working"
        so = f"Keep using it as his {jobs[0]['job'].lower()}; nothing about it needs to change."
    else:
        verdict = "No clear job yet"
        if row["stuff"] is not None and not pd.isna(row["stuff"]) and row["stuff"] >= 100:
            so = "The shape grades well; the next step is finding the count or hitter where it does a job."
        else:
            so = "Nothing stands out yet; it needs a defined role, or a shape change to give it one."
    return {"verdict": verdict, "so": so, "jobs": jobs, "deal_breakers": breakers, "great": great,
            "not_strength": weak, "pitch_plus": _num(pp, 0)}


# --------------------------------------------------------------------------
# Situations
# --------------------------------------------------------------------------
def count_table(p: pd.DataFrame) -> list[dict]:
    g = p[p["is_game"]]
    out = []
    for label, counts in COUNT_GROUPS:
        s = g[g["count"].isin(counts)]
        if s.empty:
            continue
        mix = s["shape_family"].value_counts(normalize=True)
        out.append({
            "count": label, "n": int(len(s)), "strike_pct": _pct(s["strike"].mean()), "whiff_pct": _pct(s["whiff"].mean()),
            "hard_pct": _pct(s["hard"].mean()), "most_used": f"{mix.index[0]} ({100 * mix.iloc[0]:.0f}%)",
        })
    return out


def plate_appearances(p: pd.DataFrame) -> pd.DataFrame:
    """The pitch that ends each plate appearance."""
    call, play, kb = p["PitchCall"], p["PlayResult"].fillna("Undefined"), p["KorBB"].fillna("Undefined")
    ends = kb.isin(["Strikeout", "Walk"]) | (call == "HitByPitch") | ((call == "InPlay") & (play != "Undefined"))
    return p[ends]


def slash_line(pa: pd.DataFrame) -> dict:
    if pa.empty:
        return {"pa": 0}
    kb, play, call = pa["KorBB"].fillna("Undefined"), pa["PlayResult"].fillna("Undefined"), pa["PitchCall"]
    bb, hbp, k = (kb == "Walk").sum(), (call == "HitByPitch").sum(), (kb == "Strikeout").sum()
    sac = (play == "Sacrifice").sum()
    hits = play.map(HITS).fillna(0)
    ab = len(pa) - bb - hbp - sac
    h = (hits > 0).sum()
    return {
        "pa": int(len(pa)),
        "avg": round(h / ab, 3) if ab else None,
        "obp": round((h + bb + hbp) / (ab + bb + hbp), 3) if ab + bb + hbp else None,
        "slg": round(hits.sum() / ab, 3) if ab else None,
        "k_pct": _pct(k / len(pa)), "bb_pct": _pct(bb / len(pa)),
    }


def situations(p: pd.DataFrame, ref: dict) -> dict:
    g = p[p["is_game"]]
    out: dict = {"game_pitches": int(len(g))}
    if g.empty:
        return out
    first = g[g["count"] == "0-0"]
    fp = ref["first_pitch"]
    out["first_pitch_strike"] = {
        "pct": _pct(first["strike"].mean()), "n": int(len(first)),
        "pro_range": f"{100 * fp[0.25]:.0f}–{100 * fp[0.75]:.0f}%",
    }
    out["by_count"] = count_table(p)
    splits = {}
    for side, label in (("L", "vs LHH"), ("R", "vs RHH")):
        s = p[p["BatterSide"] == side]
        if s["is_game"].any():
            mix = s.loc[s["is_game"], "shape_family"].value_counts(normalize=True)
            splits[label] = {**slash_line(plate_appearances(s)), "pitches": int(s["is_game"].sum()),
                             "mix": {k: round(100 * v) for k, v in mix.items()}}
    out["splits"] = splits

    early, late = g[g["pitch_in_outing"] <= FADE_AFTER], g[g["pitch_in_outing"] > FADE_AFTER]
    fade = {"early_n": int(len(early)), "late_n": int(len(late))}
    if len(early) >= 15 and len(late) >= 15:
        fade["strike_early"], fade["strike_late"] = _pct(early["strike"].mean()), _pct(late["strike"].mean())
        fb = p[(p["bucket"] == "Fastball")]
        fe, fl = fb[fb["pitch_in_outing"] <= FADE_AFTER], fb[fb["pitch_in_outing"] > FADE_AFTER]
        if len(fe) >= 5 and len(fl) >= 5:
            fade["fb_velo_early"], fade["fb_velo_late"] = _num(fe["RelSpeed"].mean()), _num(fl["RelSpeed"].mean())
    out["fade"] = fade
    return out


# --------------------------------------------------------------------------
# Progress through the season
# --------------------------------------------------------------------------
def progress(p: pd.DataFrame, season: pd.DataFrame, ref: dict, models: dict | None, scale: str,
             d3ref: dict | None = None) -> list[dict]:
    by = ["PitcherId", "outing", "shape_family"]
    sess = aggregate(p, by=by, min_pitches=1)
    if sess.empty:
        return []
    shifts = None
    if "velo_shift" in season and season["velo_shift"].notna().any():
        shifts = {season["PitcherId"].iloc[0]: float(season["velo_shift"].iloc[0])}
    sess = d3_stuff(sess, models, d3ref if scale == "d3" else None, shifts)
    out = []
    for _, r in sess.sort_values(["outing", "n"], ascending=[True, False]).iterrows():
        s = ref["scale"].get(r["shape_family"], {})
        season_row = season[season["shape_family"] == r["shape_family"]]
        stuff = None
        if r["n"] >= MIN_SESSION_PITCHES and s.get("stuff_xrv_std") and pd.notna(r["stuff_xrv"]) and not season_row.empty:
            mlb_plus = float(to_plus(pd.Series([r["stuff_xrv"]]), s["stuff_xrv_mean"], s["stuff_xrv_std"]).iloc[0])
            sr = season_row.iloc[0]
            if pd.notna(sr["stuff_sd"]):
                stuff = round(float(np.clip(100 + 10 * (mlb_plus - sr["stuff_mean"]) / sr["stuff_sd"], *CLIP)))
        out.append({"outing": r["outing"], "pitch": r["shape_family"], "n": int(r["n"]), "velo": _num(r["velo"]),
                    "ivb": _num(r["ivb"]), "hb": _num(r["hb"]), "spin": _num(r["spin"], 0),
                    "stuff_plus": stuff if stuff is not None else "N/A"})
    return out


# --------------------------------------------------------------------------
# Movement profile
# --------------------------------------------------------------------------
def tilt(axis_deg) -> str | None:
    """TrackMan spin axis -> clock tilt (180° = pure backspin = 12:00)."""
    if axis_deg is None or pd.isna(axis_deg):
        return None
    minutes = round(((float(axis_deg) + 180) % 360) / 30 * 60 / 15) * 15
    h, m = divmod(int(minutes), 60)
    return f"{h if h else 12}:{m:02d}"


def circular_mean(deg: pd.Series):
    d = np.deg2rad(num(deg).dropna())
    if d.empty:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean())) % 360)


def movement_profile(p: pd.DataFrame, rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rows.sort_values("n", ascending=False).iterrows():
        s = p[p["shape_family"] == r["shape_family"]]
        axis = circular_mean(s["SpinAxis"])
        out.append({
            "pitch": r["shape_family"], "subtype": r["subtype"] if isinstance(r["subtype"], str) else None,
            "n": int(r["n"]), "usage_pct": _pct(r["usage"]), "velo": _num(r["velo"]), "max_velo": _num(s["RelSpeed"].max()),
            "ivb": _num(r["ivb"]), "hb_arm": _num(r["hb"]), "spin": _num(r["spin"], 0), "tilt": tilt(axis),
            "vaa": _num(num(s["VertApprAngle"]).mean()), "rel_height": _num(r["rel_height"], 2),
            "rel_side_arm": _num(r["rel_side"], 2), "extension": _num(num(s["Extension"]).mean(), 2),
        })
    return out


FB_ROLE = {
    "Ride": "a riding four-seam: it lives at the top of the zone and above it, where the carry beats bats",
    "Flat": "a four-seam without much carry: it plays best located to the edges rather than up",
    "Sinker": "a sinker: it works down and to the arm side for ground balls and early-count contact",
    "Cutter": "a cutter: it moves glove side off the barrel and plays as a contact-management fastball",
}


def fastball_role(rows: pd.DataFrame, traits: dict | None = None) -> dict | None:
    fb = rows[rows["bucket"] == "Fastball"].sort_values("n", ascending=False)
    if fb.empty:
        return None
    r = fb.iloc[0]
    key = r["subtype"] if r["shape_family"] == "Four-seam" and isinstance(r["subtype"], str) else r["shape_family"]
    role = f"His main fastball is {FB_ROLE.get(key, key.lower())}."
    flat = (traits is not None and traits["pitch"] == r["shape_family"]
            and traits["vaa_ae_pct_mlb"] >= fastball_traits.HIGH)
    if flat and key == "Flat":
        role = ("His main fastball is a four-seam without much carry, but his release gives it a flat approach angle, "
                "so it still plays at the top of the zone.")
    return {"pitch": r["shape_family"], "subtype": key, "velo": _num(r["velo"]), "ivb": _num(r["ivb"]),
            "hb_arm": _num(r["hb"]), "role": role}


# --------------------------------------------------------------------------
# Development summary
# --------------------------------------------------------------------------
def stuff_label(x) -> str:
    if x is None or pd.isna(x):
        return "not graded"
    return "above-average" if x >= 105 else "average" if x >= 95 else "below-average"


def development(p: pd.DataFrame, rows: pd.DataFrame, judged: dict, sit: dict, staff: pd.DataFrame,
                traits: dict | None = None) -> dict:
    g = rows[rows["stuff"].notna()]
    w_stuff = float(np.average(g["stuff"], weights=g["n"])) if not g.empty else None
    loc = rows[rows["location"].notna()]
    w_loc = float(np.average(loc["location"], weights=loc["n"])) if not loc.empty else None

    # How the pitches work together.
    together = []
    fb = rows[rows["bucket"] == "Fastball"].sort_values("n", ascending=False)
    sec = g[g["bucket"] != "Fastball"]
    if not fb.empty and not sec.empty:
        f = fb.iloc[0]
        gap = f["velo"] - sec["velo"].min()
        together.append(f"{gap:.0f} mph between his fastball and his slowest pitch")
        arm = sec[sec["hb"] >= 5]["shape_family"].tolist()
        glove = sec[sec["hb"] <= -3]["shape_family"].tolist()
        if arm and glove:
            together.append(f"pitches that move both ways ({', '.join(arm)} arm side; {', '.join(glove)} glove side)")
        elif glove:
            together.append(f"everything off the fastball moves glove side ({', '.join(glove)}); no arm-side option")
        elif arm:
            together.append(f"everything off the fastball moves arm side ({', '.join(arm)}); no glove-side option")
        drop = f["ivb"] - sec["ivb"].min()
        together.append(f"{drop:.0f} in of vertical separation off the fastball")

    # Command.
    gp = p[p["is_game"]]
    strike = gp["strike"].mean() if not gp.empty else np.nan
    staff_game = staff[staff["is_game"] & (staff["PitcherId"] != p["PitcherId"].iloc[0])]
    staff_strike = staff_game["strike"].mean() if len(staff_game) >= 100 else np.nan
    pa = plate_appearances(p)
    bb = (pa["KorBB"] == "Walk").mean() if len(pa) else np.nan
    command = {"strike_pct": _pct(strike), "staff_strike_pct": _pct(staff_strike), "bb_pct": _pct(bb),
               "pa": int(len(pa)), "location_plus": _num(w_loc, 0)}
    if pd.notna(strike) and pd.notna(staff_strike) and len(gp) >= 30:
        diff = strike - staff_strike
        command["label"] = "throws strikes" if diff >= 0.03 else "strike-throwing is holding him back" if diff <= -0.03 else "average strike-throwing"
    elif w_loc is not None:
        command["label"] = "above-average location" if w_loc >= 105 else "below-average location" if w_loc < 95 else "average location"

    makes = [f"{kind}: {j['job'].lower()} ({j['evidence']})" for kind, v in judged.items() for j in v["jobs"]]

    holding = []
    priorities = []
    for kind, v in judged.items():
        for b in v["deal_breakers"]:
            holding.append(f"{kind} — {b['issue'].lower()} ({b['evidence']})")
            priorities.append((0 if b["issue"] == "Not landing strikes" else 1, f"Fix the {kind.lower()}: {b['issue'].lower()}."))

    behind = gp[gp["count"].isin(COUNT_GROUPS[3][1])]
    if len(behind) >= 10:
        fb_share = (behind["bucket"] == "Fastball").mean()
        if fb_share >= 0.75:
            cand = rows[(rows["bucket"] != "Fastball") & (rows["n_game"] >= 10)]
            rates = gp.groupby("shape_family")["strike"].mean()
            best = None
            if not cand.empty:
                cr = rates.reindex(cand["shape_family"]).dropna()
                if not cr.empty:
                    best = (cr.idxmax(), cr.max())
            hard = behind[behind["bucket"] == "Fastball"]["hard"].mean()
            text = f"Predictable when behind: {100 * fb_share:.0f}% fastballs in hitter's counts ({len(behind)} pitches)"
            if pd.notna(hard):
                text += f", {100 * hard:.0f}% of those put in play hit hard"
            holding.append(text)
            tip = f"Land a secondary when behind so hitters can't sit on the fastball"
            if best:
                tip += f" — his {best[0].lower()} goes for strikes {100 * best[1]:.0f}% of the time"
            priorities.append((2, tip + "."))

    if traits:
        makes += traits["strengths"]
        holding += traits["limits"]
        priorities += [(2.5, f) for f in traits["fixes"]]

    no_job = [k for k, v in judged.items() if v["verdict"] == "No clear job yet"]
    if no_job:
        holding.append(f"Pitches not doing a job yet: {', '.join(no_job)}")
        priorities.append((4, f"Give the {', '.join(n.lower() for n in no_job)} a defined job."))

    fade = sit.get("fade", {})
    if fade.get("strike_late") is not None and fade["strike_early"] - fade["strike_late"] >= 5:
        txt = f"Fades after pitch {FADE_AFTER}: strike rate {fade['strike_early']:.0f}% → {fade['strike_late']:.0f}%"
        if fade.get("fb_velo_late") is not None and fade["fb_velo_early"] - fade["fb_velo_late"] >= 1:
            txt += f", fastball {fade['fb_velo_early']:.1f} → {fade['fb_velo_late']:.1f} mph"
        holding.append(txt)
        priorities.append((3, f"Build the stamina to hold his stuff and strikes past pitch {FADE_AFTER}."))

    priorities.sort(key=lambda t: t[0])
    return {
        "stuff": {"stuff_plus": _num(w_stuff, 0), "label": stuff_label(w_stuff)},
        "pitches_together": together,
        "command": command,
        "what_makes_him": makes,
        "holding_back": holding,
        "priority_order": [t for _, t in priorities],
    }


def summary_sentence(name: str, dev: dict, judged: dict, rows: pd.DataFrame) -> str:
    graded = rows[rows["pitch_plus"].notna()].sort_values("pitch_plus", ascending=False)
    if graded.empty:
        return f"{name} doesn't have enough pitches of any one type to grade yet."
    parts = [f"{name} has {dev['stuff']['label']} stuff"]
    if dev["command"].get("label"):
        parts[0] += f" and {dev['command']['label']}"
    working = [k for k, v in judged.items() if v["verdict"] == "Working"]
    if working:
        k = working[0]
        parts.append(f"his {k.lower()} is doing its job as a {judged[k]['jobs'][0]['job'].lower()}")
    elif not graded.empty:
        parts.append(f"his best-graded pitch is the {graded.iloc[0]['shape_family'].lower()} "
                     f"({graded.iloc[0]['pitch_plus']:.0f} Pitch+)")
    if dev["priority_order"]:
        step = dev["priority_order"][0].split(" (")[0].split(":")[0].split(";")[0].rstrip(".")
        parts.append(f"the next step is: {step[0].lower() + step[1:]}")
    return "; ".join(parts) + "."


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------
def _fb_setup(df: pd.DataFrame) -> tuple[dict | None, tuple[float, float] | None, pd.DataFrame | None]:
    """Fastball-trait model, this data's VAA calibration, and the D3 velocity scale (each optional)."""
    model = fastball_traits.load()
    if model is None:
        return None, None, None
    fb = fastball_traits._fastballs(df)
    offset = fastball_traits.calibrate(fb) if len(fb) >= 200 else model["offset_statcast"]
    scale_path = DATA_DIR / "models" / "d3_fb_velo_scale.csv"
    d3 = pd.read_csv(scale_path) if scale_path.exists() else None
    return model, offset, d3


def analyze(df: pd.DataFrame, ref: dict, mlb_rows: pd.DataFrame, models: dict | None = None,
            scale: str = "d3", pitchers: list[str] | None = None,
            heights: dict | None = None) -> list[dict]:
    """
    Reports for every pitcher in a classified PRISM-format frame (see ``build_reports`` for raw files).
    ``heights`` maps pitcher name to height in feet (used to judge whether extension is short for his size).
    """
    df = prepare(df[df["shape_family"].notna()], ref)
    fb_model, fb_offset, d3_scale = _fb_setup(df)
    d3ref = d3.load() if scale == "d3" else None
    if scale == "d3" and d3ref is None:
        scale = "mlb"
    rows = grade_rows(df, ref, models, scale, d3ref)
    middle = df[df["located"]].groupby(["PitcherId", "shape_family"])["middle"].mean().rename("middle").reset_index()
    rows = rows.merge(middle, on=["PitcherId", "shape_family"], how="left")
    expl = explain_stuff(rows, ref, models) if rows["stuff"].notna().any() else {}
    rates = rate_table(df, ref, d3ref).set_index(["PitcherId", "shape_family"])

    reports = []
    ids = pitchers or rows["PitcherId"].unique().tolist()
    for pid in ids:
        p = df[df["PitcherId"] == pid]
        prow = rows[rows["PitcherId"] == pid].sort_values("n", ascending=False)
        if p.empty or prow.empty:
            continue
        name = display_name(p["Pitcher"].iloc[0])
        judged, pitches = {}, []
        for idx, r in prow.iterrows():
            kind = r["shape_family"]
            graded = r["n"] >= MIN_TYPE_PITCHES
            entry = {
                "pitch": kind, "subtype": r["subtype"] if isinstance(r["subtype"], str) else None,
                "bucket": r["bucket"], "n": int(r["n"]), "game_pitches": int(r["n_game"]), "usage_pct": _pct(r["usage"]),
                "grades": {
                    "pitch_plus": _num(r["pitch_plus"], 0) if graded else "N/A",
                    "stuff_plus": _num(r["stuff"], 0) if pd.notna(r["stuff"]) else "N/A",
                    "shape_plus": _num(r["shape"], 0) if pd.notna(r["shape"]) else "N/A",
                    "location_plus": _num(r["location"], 0) if pd.notna(r["location"]) else "N/A",
                    "results_plus": _num(r["results"], 0) if pd.notna(r["results"]) else "N/A",
                    "stuff_plus_mlb": _num(r["stuff_mlb"], 0) if pd.notna(r["stuff_mlb"]) else "N/A",
                    "staff_rank": (f"{fastball_traits.ordinal(r['pitch_plus_rank'])} of {int(r['pitch_plus_of'])}"
                                   if graded and pd.notna(r.get("pitch_plus_rank")) else "N/A"),
                    "compared_to": {"type": f"other {kind.lower()}s in this file", "group": f"{r['bucket'].lower()}s in this file",
                                    "all": "all pitches in this file", "mlb": f"MLB {kind.lower()}s"}.get(r["stuff_basis"], "N/A"),
                },
            }
            if not graded:
                entry["verdict"] = "Rarely used"
                entry["so"] = f"Only {int(r['n'])} thrown; not enough to grade."
                pitches.append(entry)
                continue
            parts = expl.get(idx, {})
            if parts:
                entry["stuff_explained"] = ", ".join(f"{k} {v:+.0f}" for k, v in parts.items())
                weakest = min(parts, key=parts.get)
                if parts[weakest] <= -5:
                    entry["add_on_idea"] = (f"Optional add-on: {weakest.lower()} is costing the most Stuff points "
                                            f"({parts[weakest]:+.0f}); worth exploring in bullpens.")
            comps = comparable_pitches(r, mlb_rows)
            if comps:
                entry["comparable_mlb"] = comps
            key = (pid, kind)
            if key in rates.index:
                rr = rates.loc[key]
                j = judge_pitch(r, rr)
                judged[kind] = j
                entry.update({k: j[k] for k in ("verdict", "so", "jobs", "deal_breakers", "great", "not_strength")})
                entry["rates"] = {c: _pct(rr[f"{c}_rate"]) for c in RATE_COLUMNS}
                entry["rates_ref"] = {c: _pct(rr[f"{c}_ref"]) for c in RATE_COLUMNS}
                entry["rates_ref_basis"] = rr["ref_basis"]
            pitches.append(entry)

        sit = situations(p, ref)
        traits = None
        if fb_model is not None:
            h = (heights or {}).get(p["Pitcher"].iloc[0])
            traits = fastball_traits.traits(p, fb_model, fb_offset, h, d3_scale)
        dev = development(p, prow, judged, sit, df, traits)
        report = {
            "pitcher": name, "pitcher_id": str(pid), "throws": p["PitcherThrows"].iloc[0],
            "team": p["PitcherTeam"].dropna().iloc[0] if "PitcherTeam" in p and p["PitcherTeam"].notna().any() else None,
            "pitches": int(len(p)), "game_pitches": int(p["is_game"].sum()), "outings": int(p["outing"].nunique()),
            "scale": {"d3": "D3", "mlb": "MLB"}.get(scale, "pitchers in this file"),
            "summary": summary_sentence(name, dev, judged, prow),
            "development": dev,
            "pitch_by_pitch": pitches,
            "situations": sit,
            "fastball_role": fastball_role(prow, traits),
            "fastball_traits": traits,
            "movement_profile": movement_profile(p, prow),
            "progress": progress(p, prow, ref, models, scale, d3ref),
        }
        reports.append(report)
    return reports


def build_reports(paths: list[str], scale: str = "group", team: str | None = None) -> list[dict]:
    """Read TrackMan files, classify, and build every pitcher's report."""
    from prism_data.trackman import read_trackman
    from prism_model.shapes import classify

    df = pd.concat([read_trackman(p) for p in paths], ignore_index=True)
    df = classify(df, by=["PitcherId"], merge_splits=True)
    ref, mlb_rows = references.load()
    pitchers = None
    if team:
        pitchers = df.loc[df["PitcherTeam"] == team, "PitcherId"].unique().tolist()
    return analyze(df, ref, mlb_rows, None, scale, pitchers)


# --------------------------------------------------------------------------
# Readable report
# --------------------------------------------------------------------------
def _cell(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _avg(v):
    return "–" if v is None else f"{v:.3f}".lstrip("0")


SCALE_NOTE = {
    "D3": ("Grades: 100 = average, 10 = one standard deviation. Velocity is judged against D3 pitchers "
           "(moved to the MLB velocity at the same D3 percentile); pitch shape, location and results are judged "
           "against MLB pitches of the same type, so 100 means an average MLB pitch of that type at his D3-equivalent "
           "velocity. Strike and whiff rates are compared with D3 averages."),
}


def to_markdown(r: dict) -> str:
    L = [f"# {r['pitcher']} ({r['throws']}HP)", "",
         f"{r['pitches']} pitches ({r['game_pitches']} in games) over {r['outings']} outing(s). "
         + SCALE_NOTE.get(r["scale"], f"Grades: 100 = average for {r['scale']}."), "", f"**{r['summary']}**", ""]

    d = r["development"]
    L += ["## Development", "",
          f"- **Stuff:** {d['stuff']['label']}" + (f" ({d['stuff']['stuff_plus']} Stuff+, usage-weighted)" if d['stuff']['stuff_plus'] is not None else "")]
    if d["pitches_together"]:
        L.append(f"- **How the pitches work together:** {'; '.join(d['pitches_together'])}")
    c = d["command"]
    cmd = [f"{c['strike_pct']:.0f}% strikes" if c["strike_pct"] is not None else None,
           f"staff {c['staff_strike_pct']:.0f}%" if c["staff_strike_pct"] is not None else None,
           f"{c['bb_pct']:.0f}% walks in {c['pa']} PA" if c["bb_pct"] is not None else None,
           f"{c['location_plus']} Location+" if c["location_plus"] is not None else None]
    L.append(f"- **Command:** {c.get('label', 'not enough data')} ({', '.join(x for x in cmd if x)})")
    L += ["", "**What makes him**"] + ([f"- {x}" for x in d["what_makes_him"]] or ["- No pitch is clearly beating its peers yet."])
    L += ["", "**What's holding him back**"] + ([f"- {x}" for x in d["holding_back"]] or ["- Nothing clear in this data."])
    if d["priority_order"]:
        L += ["", "**Priority order**"] + [f"{i}. {x}" for i, x in enumerate(d["priority_order"], 1)]

    L += ["", "## Pitch grades", "",
          "| Pitch | Subtype | n | Game | Usage | Pitch+ | Stuff+ | Location+ | Results+ | Shape+ | Staff rank (Pitch+) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in r["pitch_by_pitch"]:
        g = p["grades"]
        L.append(f"| {p['pitch']} | {_cell(p['subtype'])} | {p['n']} | {p['game_pitches']} | {_cell(p['usage_pct'])}% | "
                 f"{g['pitch_plus']} | {g['stuff_plus']} | {g['location_plus']} | {g['results_plus']} | {g['shape_plus']} | "
                 f"{g.get('staff_rank', 'N/A')} |")

    L += ["", "## Pitch by pitch"]
    for p in r["pitch_by_pitch"]:
        L += ["", f"### {p['pitch']}" + (f" ({p['subtype']})" if p["subtype"] else "") + f" — {p.get('verdict', '')}"]
        g = p["grades"]
        L.append(f"Pitch+ {g['pitch_plus']} · Stuff+ {g['stuff_plus']} · Location+ {g['location_plus']} · "
                 f"Results+ {g['results_plus']} (vs {g['compared_to']})")
        if p.get("stuff_explained"):
            L.append(f"- Why the Stuff grade: {p['stuff_explained']}")
        if p.get("comparable_mlb"):
            L.append(f"- {p['comparable_mlb']['text']}")
        for j in p.get("jobs", []):
            L.append(f"- **Job — {j['job']}:** {j['evidence']}")
        for x in p.get("great", []):
            if not any(x == j["evidence"] for j in p.get("jobs", [])):
                L.append(f"- Great: {x}")
        for b in p.get("deal_breakers", []):
            L.append(f"- **Where it's failing — {b['issue']}:** {b['evidence']}")
        for x in p.get("not_strength", []):
            L.append(f"- Not its strength: {x}")
        if p.get("add_on_idea"):
            L.append(f"- {p['add_on_idea']}")
        L.append(f"- **So:** {p.get('so', '')}")

    s = r["situations"]
    L += ["", "## Situations"]
    if s.get("first_pitch_strike"):
        f = s["first_pitch_strike"]
        L.append(f"- First-pitch strikes: {_cell(f['pct'])}% of {f['n']} (MLB middle half: {f['pro_range']})")
    fade = s.get("fade", {})
    if fade.get("strike_late") is not None:
        txt = f"- Pitches 1–{FADE_AFTER}: {fade['strike_early']:.0f}% strikes; after: {fade['strike_late']:.0f}%"
        if fade.get("fb_velo_late") is not None:
            txt += f" (fastball {fade['fb_velo_early']} → {fade['fb_velo_late']} mph)"
        L.append(txt)
    elif s.get("game_pitches"):
        L.append(f"- Fade after pitch {FADE_AFTER}: not enough pitches on both sides yet")
    if s.get("by_count"):
        L += ["", "| Count | n | Strike% | Whiff% | Hard% | Most used |", "|---|---|---|---|---|---|"]
        for row in s["by_count"]:
            L.append(f"| {row['count']} | {row['n']} | {_cell(row['strike_pct'])} | {_cell(row['whiff_pct'])} | "
                     f"{_cell(row['hard_pct'])} | {row['most_used']} |")
    if s.get("splits"):
        L += ["", "| Split | PA | AVG | OBP | SLG | K% | BB% | Mix |", "|---|---|---|---|---|---|---|---|"]
        for label, sp in s["splits"].items():
            mix = ", ".join(f"{k} {v}%" for k, v in sp.get("mix", {}).items())
            L.append(f"| {label} | {sp['pa']} | {_avg(sp.get('avg'))} | {_avg(sp.get('obp'))} | {_avg(sp.get('slg'))} | "
                     f"{_cell(sp.get('k_pct'))} | {_cell(sp.get('bb_pct'))} | {mix} |")

    if r.get("fastball_role"):
        L += ["", "## Fastball", "", r["fastball_role"]["role"]]
    t = r.get("fastball_traits")
    if t:
        line = (f"- **Velocity:** {t['velo']} mph actual, {t['perceived_velo']} perceived "
                f"({t['extension_ft']} ft extension, {t['extension_bonus_mph']:+.1f} mph)")
        if t.get("velo_pct_d3") is not None:
            line += (f" — {fastball_traits.ordinal(t['velo_pct_d3'])} → "
                     f"{fastball_traits.ordinal(t['perceived_pct_d3'])} percentile among D3 pitchers")
        L += ["" if not r.get("fastball_role") else "", line,
              f"- **Approach angle:** {t['vaa']}° at an average plate height of {t['avg_plate_height_ft']} ft; "
              f"{t['vaa_above_expected']:+.2f}° vs expected for his velocity and location "
              f"({fastball_traits.ordinal(t['vaa_ae_pct_mlb'])} percentile vs MLB, higher = flatter)"]
        if t.get("expected_extension_ft"):
            L.append(f"- **Extension vs size:** {t['extension_ft']} ft; about {t['expected_extension_ft']} ft is typical for his height")
    L += ["", "## Movement profile", "",
          "| Pitch | Velo (max) | IVB | HB (arm +) | Spin | Tilt | VAA | Rel H | Rel S | Ext |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in r["movement_profile"]:
        L.append(f"| {m['pitch']} | {_cell(m['velo'])} ({_cell(m['max_velo'])}) | {_cell(m['ivb'])} | {_cell(m['hb_arm'])} | "
                 f"{_cell(m['spin'])} | {_cell(m['tilt'])} | {_cell(m['vaa'])} | {_cell(m['rel_height'])} | "
                 f"{_cell(m['rel_side_arm'])} | {_cell(m['extension'])} |")
    if r["progress"]:
        L += ["", "## Progress by outing", "", "| Outing | Pitch | n | Velo | IVB | HB | Spin | Stuff+ |", "|---|---|---|---|---|---|---|---|"]
        for x in r["progress"]:
            L.append(f"| {x['outing']} | {x['pitch']} | {x['n']} | {_cell(x['velo'])} | {_cell(x['ivb'])} | {_cell(x['hb'])} | "
                     f"{_cell(x['spin'])} | {x['stuff_plus']} |")
    return "\n".join(L) + "\n"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "pitcher"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--scale", choices=["group", "mlb"], default="group")
    parser.add_argument("--team", help="only write reports for this PitcherTeam (everyone still sets the scale)")
    args = parser.parse_args()

    reports = build_reports(args.paths, args.scale, args.team)
    dest = REPORTS_DIR / Path(args.paths[0]).stem
    dest.mkdir(parents=True, exist_ok=True)
    for r in reports:
        stem = _slug(r["pitcher"])
        (dest / f"{stem}.json").write_text(json.dumps(r, indent=2, default=str))
        (dest / f"{stem}.md").write_text(to_markdown(r))
        print(f"- {r['pitcher']}: {r['summary']}")
    print(f"\n{len(reports)} reports saved to {dest}")


if __name__ == "__main__":
    main()
