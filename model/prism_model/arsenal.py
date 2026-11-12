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

from prism_data import sessions
from prism_data.paths import DATA_DIR
from prism_model import d3, fastball_traits, location, references, results_tiers
from prism_model.references import RATE_COLUMNS, add_flags, num, location_score, result_values, results_score
from prism_model.shapes import BUCKET, FB_ZONES, OPEN_TAGS
from prism_model.stuff import GROUPS, aggregate, predict, to_plus

REPORTS_DIR = DATA_DIR / "reports"

MIN_TYPE_PITCHES = 10       # a pitch type needs this many to be graded
MIN_GAME_FOR_PITCH_PLUS = 15  # Pitch+ needs some results: this many pitches vs hitters (game or live sessions)
MIN_LOCATED = 10            # located pitches needed for a Location grade
MIN_SESSION_PITCHES = 5     # per outing, for Progress
MIN_PEERS = 4               # pitch types needed to set a 100 scale
# Pitch+ trusts Results n / (n + k). Garnet used 15, but results are mostly noise at fall sample sizes. k now
# depends on how much is known about the balls in play (results_tiers: measured 400, charted 440, outcome-only
# 1000 game pitches, measured on MLB). RESULTS_K is the fallback when a pitch type has no balls in play yet.
RESULTS_K = results_tiers.TIER_K["outcome"]
# Location+ lives in location.py (MLB location run values, D3 center, sample-size regression).
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
    tiers = results_tiers.load()
    if tiers is not None:
        ev = num(df["ExitSpeed"])
        if "ExitSpeed_mlb" not in df and ev.notna().any():
            # no level reference from the build: place exit velocities within this file's own batted balls
            bip = df["PitchCall"].astype(object) == "InPlay"
            df["ExitSpeed_mlb"] = results_tiers.level_ev_to_mlb(ev, ev[bip], tiers)
        df = results_tiers.classify_contact(df, tiers)
        df["result_rv"] = results_tiers.tiered_values(df, ref["results"], tiers)
    else:
        df["result_rv"] = result_values(df, ref["results"])
        df["result_tier"] = None
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
    Predict Stuff/Shape xRV. With a D3 reference, each pitch is graded as thrown at his MLB-equivalent
    arm speed (``d3.translate`` + ``d3.model_inputs``: velocity, spin and separation scaled together; or by
    the given per-pitcher velocity ``shifts`` ratios), and the velocity part of fastball Stuff is weighted by
    the D3 velocity weight. Displayed values stay as thrown.
    """
    if d3ref is not None:
        if shifts is None:
            rows = d3.translate(rows, d3ref)
        else:
            rows = rows.copy()
            rows["perceived_velo_d3"] = rows["perceived_velo"]
            rows["velo_ratio"] = rows["PitcherId"].map(shifts).fillna(1.0)
    pred = predict(d3.model_inputs(rows), models)
    rows = rows.copy()
    for c in ("group", "stuff_xrv", "shape_xrv"):
        if c in pred:
            rows[c] = pred[c].to_numpy()
    if d3ref is not None:
        fb = rows["group"] == "Fastball"
        w = d3ref["velo_weight"]
        rows.loc[fb, "stuff_xrv"] = rows.loc[fb, "shape_xrv"] + w * (rows.loc[fb, "stuff_xrv"] - rows.loc[fb, "shape_xrv"])
    return rows


def add_results_tier(rows: pd.DataFrame, df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Each pitch type's Results tier and weight constant from its balls in play (results_tiers.tier_mix).
    A pitch type with no balls in play yet takes the pitcher's overall mix, else outcome-only.
    """
    if "result_tier" not in df or df["result_tier"].isna().all():
        return rows.assign(results_k=float(RESULTS_K), results_tier=None)
    mix = results_tiers.tier_mix(df, by)
    rows = rows.merge(mix[by + ["results_k", "results_tier"]], on=by, how="left")
    pitcher = results_tiers.tier_mix(df, [by[0]]).set_index(by[0])
    rows["results_k"] = rows["results_k"].fillna(rows[by[0]].map(pitcher["results_k"])).fillna(float(RESULTS_K))
    rows["results_tier"] = rows["results_tier"].fillna(rows[by[0]].map(pitcher["results_tier"])).fillna("outcome")
    return rows


def grade_rows(df: pd.DataFrame, ref: dict, models: dict | None, scale: str = "d3",
               d3ref: dict | None = None) -> pd.DataFrame:
    """One row per pitcher pitch type with Stuff, Shape, Location, Results and Pitch+."""
    by = ["PitcherId", "shape_family"]
    rows = aggregate(df, by=by, min_pitches=1)
    rows["bucket"] = rows["shape_family"].map(BUCKET)
    rows = d3_stuff(rows, models, d3ref if scale == "d3" else None)
    rows = rows.merge(results_score(df, by), on=by, how="left")
    rows["n_game"] = rows["n_game"].fillna(0).astype(int)
    rows = add_results_tier(rows, df, by)
    from prism_data.devices import device_of

    dev_col = device_of(df)
    main_dev = dev_col.groupby([df[c] for c in by]).agg(lambda s: s.mode().iloc[0]).rename("device").reset_index()
    rows = rows.merge(main_dev, on=by, how="left")
    rows["device"] = rows["device"].fillna("unknown")
    rows = location.grade(rows, df, by)   # run value of his locations vs D3 average (location.py)
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
    rows["location_mlb"] = rows["location_plus"].clip(*CLIP).where(graded & (rows["n_located"] >= MIN_LOCATED))
    rows["results_mlb"] = mlb("results_rv100", "results_rv100", False).where(graded & (rows["n_game"] > 0))

    for comp in ["stuff", "shape", "location", "results"]:
        if scale in ("mlb", "d3"):
            rows[comp] = rows[f"{comp}_mlb"]
            rows[f"{comp}_basis"], rows[f"{comp}_mean"], rows[f"{comp}_sd"] = "mlb", 100.0, 10.0
        else:
            r = staff_rescale(rows, f"{comp}_mlb")
            rows[comp], rows[f"{comp}_basis"], rows[f"{comp}_mean"], rows[f"{comp}_sd"] = r["plus"], r["basis"], r["mean"], r["sd"]

    rows["pitch_plus"] = [
        pitch_plus(s, l, r, g, k) if n >= MIN_TYPE_PITCHES and g >= MIN_GAME_FOR_PITCH_PLUS else None
        for s, l, r, g, n, k in zip(rows["stuff"], rows["location"], rows["results"], rows["n_game"], rows["n"],
                                    rows["results_k"])
    ]
    rows["pitch_plus"] = rows["pitch_plus"].astype(float)

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
        x = d3.model_inputs(r.to_frame().T)[feats].astype(float)
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


# Who a pitch is compared with: the MLB pitcher-season pitch types of the same family most like it. Fastballs by
# their own shape; everything else by how it separates from his fastball (the separation is the pitch), plus the
# fastball it comes off (a changeup off a high-run sinker is compared with changeups off high-run fastballs).
SIMILAR_FEATURES = {
    "Fastball": {"ivb": 1.0, "hb": 1.0, "spin": 0.5, "rel_height": 0.5},
    "other": {"velo_diff": 1.0, "ivb_diff": 1.0, "hb_diff": 1.0, "fb_ivb": 0.5, "fb_hb": 0.5, "spin": 0.5},
}
SIMILAR_SHARE, SIMILAR_MIN, SIMILAR_MAX = 0.15, 60, 150
# Beyond this share of MLB at either end, the Stuff model has almost nothing like the pitch to learn from:
# a movement trait out there is part of what the pitch is, and isn't targeted toward the middle.
SUPPORT_Q = 0.03


_MLB_TAGS: dict = {}


def mlb_tags(fam: pd.DataFrame, kind: str) -> pd.Series:
    """Each MLB pitcher-season's subtype by the same rules as ours (cached per pitch type)."""
    from prism_model.shapes import subtype_for

    key = (kind, len(fam), float(fam["ivb"].sum()) if "ivb" in fam else 0.0)
    if key not in _MLB_TAGS:
        get = lambda row, c: row[c] if c in row and pd.notna(row[c]) else np.nan
        _MLB_TAGS[key] = pd.Series([
            subtype_for(kind, {"ivb": get(row, "ivb"), "hb": get(row, "hb"), "glove": -get(row, "hb"),
                               "ivb_diff": get(row, "ivb_diff"), "hb_diff": get(row, "hb_diff")})
            for _, row in fam.iterrows()], index=fam.index, dtype=object)
    return _MLB_TAGS[key]


def similar_pool(r: pd.Series, mlb_rows: pd.DataFrame) -> pd.DataFrame:
    """
    The MLB pitches of his pitch type most like his (15% of them, 60-150), nearest first, among those with
    the same subtype (a sweeper among sweepers, a cut fastball among cut fastballs), so a pitch is never
    graded toward a shape his delivery can't make. Weak shapes (deadzone four-seams, firm changeups) are
    compared with the whole pitch type so their targets can move them out.
    """
    from prism_model.shapes import OPEN_TAGS

    fam = mlb_rows[mlb_rows["shape_family"] == r["shape_family"]].copy()
    if fam.empty:
        return fam
    sub = r.get("subtype")
    if isinstance(sub, str) and sub not in OPEN_TAGS:
        same = fam[mlb_tags(fam, r["shape_family"]) == sub]
        if len(same) >= SIMILAR_MIN:
            fam = same
    if "ivb_diff" in fam:
        fam["fb_ivb"] = fam["ivb"] - fam["ivb_diff"]
        fam["fb_hb"] = fam["hb"] - fam["hb_diff"]
    me = r.copy()
    if pd.notna(me.get("ivb_diff")):
        me["fb_ivb"], me["fb_hb"] = me["ivb"] - me["ivb_diff"], me["hb"] - me["hb_diff"]
    feats = SIMILAR_FEATURES["Fastball" if r.get("bucket") == "Fastball" else "other"]
    feats = {c: w for c, w in feats.items() if c in fam and pd.notna(me.get(c)) and fam[c].notna().any()}
    if not feats:
        return fam
    d = np.zeros(len(fam))
    for c, w in feats.items():
        sd = fam[c].std() or 1.0
        d += w * ((fam[c] - float(me[c])) / sd).fillna(10.0).to_numpy() ** 2
    k = int(min(SIMILAR_MAX, max(SIMILAR_MIN, SIMILAR_SHARE * len(fam))))
    return fam.assign(_dist=d).nsmallest(k, "_dist")


def comparable_pitches(r: pd.Series, mlb_rows: pd.DataFrame) -> dict | None:
    """MLB pitcher-seasons with a similar pitch, and how they did versus the average of that pitch type."""
    pool = mlb_rows[mlb_rows["shape_family"] == r["shape_family"]].dropna(subset=["xrv100"])
    label = f"{r['shape_family'].lower()}s"
    if pool.empty or pd.isna(r["velo"]):
        return None
    ratio = r.get("velo_ratio", 1.0)
    velo = r["velo"] * (ratio if pd.notna(ratio) else 1.0)  # D3-equivalent MLB speed when translated
    type_avg = np.average(pool["xrv100"], weights=pool["n"])
    for dv, dm in ((1.5, 2.0), (2.5, 3.0), (4.0, 4.0)):
        near = pool[((pool["velo"] - velo).abs() <= dv) & ((pool["ivb"] - r["ivb"]).abs() <= dm)
                    & ((pool["hb"] - r["hb"]).abs() <= dm)]
        if len(near) >= 25:
            break
    if len(near) < 10:
        return {"n": int(len(near)), "text": f"Too few MLB {label} with this speed and shape to compare."}
    diff = np.average(near["xrv100"], weights=near["n"]) - type_avg
    word = "fewer" if diff < 0 else "more"
    return {
        "n": int(len(near)),
        "pitches": int(near["n"].sum()),
        "window": f"±{dv:g} mph, ±{dm:g} in",
        "runs_per_100_vs_avg": round(float(diff), 2),
        "mlb_equivalent_velo": round(float(velo), 1),
        "text": (f"{len(near)} comparable MLB {label} "
                 + (f"(around {velo:.0f} mph, his D3-equivalent speed, " if abs(velo - r['velo']) >= 0.5 else "(")
                 + f"±{dv:g} mph, ±{dm:g} in of movement) "
                 f"allowed {abs(diff):.2f} {word} runs per 100 pitches than the average MLB {label[:-1]}."),
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
    def edge(c, good):
        r, r0 = rates[f"{c}_rate"], rates[f"{c}_ref"]
        return good * (r - r0) / max(r0 if good > 0 else 1 - r0, 0.05)

    jobs = sorted(({"job": name, "does": does, "evidence": line(c), "edge": edge(c, good)}
                   for c, good, name, does in JOBS if calls[c] == good), key=lambda j: -j["edge"])
    jobs = jobs[:1]  # one job per pitch: the clearest edge; other strengths show as supporting evidence
    breakers = [{"issue": label, "evidence": line(c)} for c, (bad, label) in DEAL_BREAKERS.items() if calls[c] == bad]
    good_dirs = {c: g for c, g, _, _ in JOBS}
    great = [line(c) for c in RATE_COLUMNS if c in good_dirs and calls[c] == good_dirs[c]
             and not (jobs and line(c) == jobs[0]["evidence"])]
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
    if "velo_ratio" in season and season["velo_ratio"].notna().any():
        shifts = {season["PitcherId"].iloc[0]: float(season["velo_ratio"].iloc[0])}
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
    "Cut-ride": "a cut-riding four-seam: carry plus a little cut, so it stays on the barrel's top edge and away from arm-side hitters",
    "Ride-run": "a ride-run four-seam: carry and arm-side run together, best up and to the arm side",
    "Cut": "a cutting four-seam: little run, so it plays glove side and off the barrel more than up",
    "Run": "a running four-seam: lots of arm-side run, so it plays to the arm side and in on same-side hitters",
    "Deadzone": "a deadzone four-seam: not enough carry to ride and not enough run or cut to move off barrels, so location carries it",
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
    if flat and key in ("Deadzone", "Run", "Cut"):
        role = (f"His main fastball is {FB_ROLE[key].split(':')[0]}, but his release gives it a flat approach angle, "
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
        pct = traits.get("velo_pct_d3")
        if pct is not None and pct < VELO_FLAG_PCT:
            holding.insert(0, f"Velocity: his {traits['velo']} mph {traits['pitch'].lower()} is in the "
                              f"{fastball_traits.ordinal(pct)} percentile among D3 pitchers")
            priorities.append((0.5, "Add velocity across the whole arsenal: every pitch grades better when everything "
                                    "gets harder together, so the gaps off the fastball stay the same."))

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
        parts.append(f"his next step: {step[0].lower() + step[1:]}")
    return "; ".join(parts) + "."


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------
TRAITS = {  # trait -> (columns that move with it, step, label, unit)
    # This pitch harder (fastball moves with it, so the gap stays the same). Never asks for slower.
    "velo": (["velo", "perceived_velo"], 0.5, "Velocity", "mph"),
    # Separation from his fastball, with this pitch's own speed held (i.e. the fastball moves).
    # This pitch's own speed with his fastball held where it is: velo, perceived velo and separation move together.
    # (Moving velo_diff alone would mean an imagined faster fastball, which belongs on the fastball's card.)
    "gap": (["velo_diff", "velo", "perceived_velo"], 0.5, "Gap off fastball", "mph"),
    "ivb": (["ivb", "ivb_diff"], 1.0, "IVB (ride +)", "in"),
    "hb": (["hb", "hb_diff"], 1.0, "HB (arm side +)", "in"),
    "spin": (["spin"], 100.0, "Spin", "rpm"),
}
GRIP_SLOWED = ("Changeup", "Splitter")  # the only pitches where "slower" can be the right goal (via grip, not intent)
VELO_FLAG_PCT = 25   # fastball velocity below this D3 percentile becomes a headline priority
TARGET_GAIN = 5.0   # Stuff+ points a change has to be worth before it becomes a target (small nudges are noise)
TARGET_BAND = 2.0   # values within this many points of the best count as "in range"
VELO_UP_MAX = 4.0   # don't ask for more than ~4 mph
SPIN_MAX_CHANGE = 300.0
VELO_WINDOW = 2.0   # width of a secondary's velocity target range, set off his current fastball  # spin is hard to change; look at most +-300 rpm


def pitch_needing_work(pitches: list[dict]) -> dict | None:
    """
    The pitch to build a card for: a graded pitch that grades below average (Pitch+ < 100)
    or has a deal-breaker (not landing strikes, getting hit hard). Lowest Pitch+ first.
    """
    cand = [p for p in pitches if isinstance(p["grades"]["pitch_plus"], (int, float))
            and (p["grades"]["pitch_plus"] < 100 or p.get("deal_breakers"))]
    if not cand:
        return None
    p = min(cand, key=lambda x: x["grades"]["pitch_plus"])
    return {"pitch": p["pitch"], "verdict": p.get("verdict", ""), "subtype": p.get("subtype")}


def pitch_targets(r: pd.Series, mlb_rows: pd.DataFrame, models: dict | None, ref: dict) -> dict | None:
    """
    Shape targets from the Stuff model, kept among MLB pitches like his (similar_pool).

    Each trait (velocity, ride, run, spin) is moved on its own across the range the MLB pitches most like
    his actually cover (10th-90th percentile; velocity only up to +4 mph), with everything
    else held where he is. The Stuff+ the model gives at each value shows which changes matter: a
    trait becomes a target only if moving it is worth 3+ Stuff+, and its target range is every
    value within 2 points of the best one. Traits where he's already near the best stay green. Ride or
    run beyond 97% of MLB (either end) isn't targeted: the model has nothing like it to learn from.
    """
    kind, sub = r["shape_family"], r.get("subtype")
    sc = ref["scale"].get(kind)
    if not sc:
        return None
    family = mlb_rows[(mlb_rows["shape_family"] == kind)]
    pool = similar_pool(r, mlb_rows)
    plus = lambda x: 100 - 10 * (x - sc["stuff_xrv_mean"]) / sc["stuff_xrv_std"]
    base = predict(d3.model_inputs(pd.DataFrame([r])), models)["stuff_xrv"].iloc[0]
    if pd.isna(base):
        return None
    now_plus = float(plus(base))
    rows = []
    for key, (cols, step, label, unit) in TRAITS.items():
        main = cols[0]
        if pd.isna(r.get(main)):
            continue
        if key == "gap" and (r.get("group") == "Fastball" or pd.isna(r.get("velo_diff"))):
            continue
        if key == "velo":
            lo, hi = 0.0, VELO_UP_MAX
        elif key == "gap":
            # Harder: up to +4 mph (never past 1 mph under the fastball). Slower: only changeups/splitters
            # (by grip), down to the subtype's 10th-percentile separation.
            q10 = pool[main].dropna().quantile(0.1) if main in pool else np.nan
            lo = min(float(q10 - r[main]), 0.0) if kind in GRIP_SLOWED and pd.notna(q10) else 0.0
            hi = max(min(VELO_UP_MAX, -1.0 - float(r[main])), 0.0)
        else:
            q = pool[main].dropna().quantile([0.1, 0.9]) if main in pool else None
            if q is None or q.isna().any():
                continue
            if key in ("ivb", "hb") and main in family:
                edge = family[main].dropna().quantile([SUPPORT_Q, 1 - SUPPORT_Q])
                if not edge.iloc[0] <= r[main] <= edge.iloc[1]:
                    rows.append({"trait": key, "label": label, "unit": unit, "now": round(float(r[main]), 1),
                                 "lo": None, "hi": None, "gain": 0.0, "best_delta": 0.0, "in_range": True,
                                 "beyond": "more" if r[main] > edge.iloc[1] else "less",
                                 "open_hi": False, "open_lo": False})
                    continue
            lo, hi = float(q.iloc[0] - r[main]), float(q.iloc[1] - r[main])
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            zone = FB_ZONES.get(sub) if kind == "Four-seam" and sub not in OPEN_TAGS else None
            if zone is not None and key in ("ivb", "hb"):
                # stay inside his fastball's zone: ride and run targets don't turn one shape into another
                z_lo, z_hi = (zone[0], zone[1]) if key == "ivb" else (zone[2], zone[3])
                if z_lo is not None:
                    lo = min(max(lo, z_lo - float(r[main])), 0.0)
                if z_hi is not None:
                    hi = max(min(hi, z_hi - float(r[main])), 0.0)
            if key == "spin":
                lo, hi = max(lo, -SPIN_MAX_CHANGE), min(hi, SPIN_MAX_CHANGE)
        deltas = np.arange(np.floor(lo / step) * step, hi + step / 2, step)
        grid = pd.DataFrame([r] * len(deltas)).reset_index(drop=True)
        for c in cols:
            if c in grid:
                grid[c] = grid[c] + deltas
        g = plus(predict(d3.model_inputs(grid), models)["stuff_xrv"]).to_numpy(float)
        if np.isnan(g).all():
            continue
        best = np.nanmax(g)
        ok = deltas[g >= best - TARGET_BAND]
        gain = best - now_plus
        now = float(r[main] if key != "velo" else r["velo"])
        win = (now + ok.min(), now + ok.max())
        in_range = gain < TARGET_GAIN or ok.min() <= 0 <= ok.max()
        best_delta = float(deltas[int(np.nanargmax(g))])
        evidence = None
        if key in ("ivb", "hb", "spin") and not in_range:
            # Evidence check: among the MLB pitches most like his, did the ones with more of this trait actually
            # allow fewer runs (actual run values, not the model)? A target only stands when they agree.
            ev = pool.dropna(subset=[main, "xrv100"]) if "xrv100" in pool else pool.iloc[0:0]
            if len(ev) >= 20:
                hi_half, lo_half = ev[ev[main] >= ev[main].median()], ev[ev[main] < ev[main].median()]
                diff = (np.average(hi_half["xrv100"], weights=hi_half["n"])
                        - np.average(lo_half["xrv100"], weights=lo_half["n"]))   # > 0: more of it, more runs
                agrees = (diff < 0) == (best_delta > 0)
                evidence = {"runs_per_100": round(float(abs(diff)), 2), "agrees": bool(agrees), "n": int(len(ev))}
                if not agrees:
                    in_range = True   # the model and the results disagree: no target
        rows.append({"trait": key, "label": label, "unit": unit, "now": round(now, 1), "evidence": evidence,
                     "lo": round(win[0], 1), "hi": round(win[1], 1), "gain": round(gain, 0),
                     "best_delta": round(best_delta, 1), "in_range": bool(in_range),
                     "open_hi": bool(ok.max() >= deltas.max() - 1e-9 and ok.max() > 0),
                     "open_lo": bool(ok.min() <= deltas.min() + 1e-9 and ok.min() < 0)})
    if not rows:
        return None
    g_row = next((x for x in rows if x["trait"] == "gap"), None)
    fb_velo = None
    if g_row is not None:
        # Secondaries: one Velocity row whose target is a speed range set off his CURRENT fastball (Garnet style).
        # The separation window comes from the Stuff model (values within 2 Stuff+ of the best), narrowed to the
        # middle half of MLB pitches of this subtype when that overlaps, so the range is a real, usable target.
        # A 2 mph window (like Garnet's). When his separation already grades fine it's centered on where he is.
        # Otherwise it starts at the nearest edge of the band that grades within 2 Stuff+ of the best: the
        # smallest change that gets nearly all of the gain, not the extreme of the MLB range.
        fb_velo = float(r["velo"] - r["velo_diff"])
        now_d = float(r["velo_diff"])
        if g_row["in_range"]:
            lo, hi = now_d - VELO_WINDOW / 2, now_d + VELO_WINDOW / 2
        elif now_d > g_row["hi"]:   # too close to the fastball: window just below the band's near edge
            lo, hi = g_row["hi"] - VELO_WINDOW, g_row["hi"]
        else:                        # too far off the fastball: window just above the band's near edge
            lo, hi = g_row["lo"], g_row["lo"] + VELO_WINDOW
        rows = [x for x in rows if x["trait"] != "velo"]
        g_row["hide"] = True  # kept for the plan text, shown through the Velocity row
        rows.insert(0, {"trait": "velo", "label": "Velocity", "unit": "mph", "now": round(float(r["velo"]), 1),
                        "lo": round(fb_velo + lo, 1), "hi": round(fb_velo + hi, 1), "gain": g_row["gain"],
                        "best_delta": g_row["best_delta"], "in_range": g_row["in_range"], "vs_fb": True,
                        "fb_velo": round(fb_velo, 1), "open_hi": False, "open_lo": False})

    def word(x):
        more = x["best_delta"] > 0
        return {"velo": "more velocity", "gap": "less separation from his fastball" if more else "more separation from his fastball", "ivb": "more ride (less drop)" if more else "more drop",
                "hb": ("less sweep" if r["hb"] < 0 else "more run") if more else ("more sweep" if r["hb"] < 0 else "less run"),
                "spin": "more spin" if more else "less spin"}[x["trait"]]

    def where(x):
        if x.get("beyond"):
            return "no target (rare: beyond 97% of MLB)"
        if x["trait"] == "gap":  # velo_diff is negative; talk about it as mph off the fastball
            lo, hi = -x["hi"], -x["lo"]
            if x["open_lo"] and not x["open_hi"]:
                return f"{lo:.1f}+ mph off his fastball"
            if x["open_hi"] and not x["open_lo"]:
                return f"{hi:.1f} mph or less off his fastball"
            return f"{lo:.1f} to {hi:.1f} mph off his fastball"
        if x.get("vs_fb"):
            return f"{x['lo']:.1f}–{x['hi']:.1f} mph"
        if x["trait"] == "velo":  # an actual speed for this pitch, only ever up from where he is
            return f"{x['lo']:.1f}+ mph" if x["open_hi"] else (
                f"{x['lo']:.1f} to {x['hi']:.1f} mph" if x["lo"] != x["hi"] else f"about {x['lo']:.1f} mph")
        f = {"ivb": lambda v: f"{v:+.1f}", "hb": lambda v: f"{v:+.1f}", "spin": lambda v: f"{v:,.0f}"}.get(
            x["trait"], lambda v: f"{v:.1f}")
        if x["open_hi"] and not x["open_lo"]:
            return f"{f(x['lo'])} {x['unit']} or more"
        if x["open_lo"] and not x["open_hi"]:
            return f"{f(x['hi'])} {x['unit']} or less"
        return f"{f(x['lo'])} to {f(x['hi'])} {x['unit']}" if x["lo"] != x["hi"] else f"about {f(x['lo'])} {x['unit']}"

    for x in rows:
        x["target_text"] = where(x)
    shape_gaps = sorted((x for x in rows if not x["in_range"] and x["trait"] not in ("velo", "gap")), key=lambda x: -x["gain"])
    gap = next((x for x in rows if x["trait"] == "gap" and not x["in_range"]), None)
    velo = next((x for x in rows if x["trait"] == "velo" and not x["in_range"] and not x.get("vs_fb")), None)
    vfb = next((x for x in rows if x.get("vs_fb")), None)
    rng = (f"{vfb['lo']:.1f}–{vfb['hi']:.1f} mph with his current {vfb['fb_velo']:.1f} mph fastball" if vfb else "")
    parts = []
    if shape_gaps:
        g0 = shape_gaps[0]
        parts.append(f"The shape change worth the most is {word(g0)}: about {g0['gain']:+.0f} Stuff+ at "
                     f"{g0['target_text']} (he's at {(f'{g0[chr(110)+chr(111)+chr(119)]:,.0f}' if g0['trait'] == 'spin' else f'{g0[chr(110)+chr(111)+chr(119)]:+.1f}')}).")
        ev = g0.get("evidence")
        if ev and ev["agrees"]:
            up = g0["best_delta"] > 0
            if g0["trait"] == "hb" and r["hb"] < 0:   # glove-side pitch: a higher number is less sweep
                trait_phrase = "less sweep" if up else "more sweep"
            else:
                trait_phrase = f"{'more' if up else 'less'} {({'ivb': 'ride', 'hb': 'run', 'spin': 'spin'})[g0['trait']]}"
            parts.append(f"The results back it: of the {ev['n']} MLB {kind.lower()}s most like his, the half with "
                         f"{trait_phrase} allowed {ev['runs_per_100']:.2f} fewer runs per 100 pitches.")
        if len(shape_gaps) > 1:
            parts.append("After that: " + ", ".join(f"{word(x)} ({x['gain']:+.0f})" for x in shape_gaps[1:]) + ".")
    else:
        parts.append("His shape is already close to the best version of this pitch.")
    if gap:
        if gap["best_delta"] < 0 and kind in GRIP_SLOWED:
            # The one exception: a changeup/splitter too close to the fastball. Slowing it is the pitch's whole
            # design, but it has to come from the grip with fastball arm speed, not from easing up.
            parts.append(f"It's too close to his fastball's speed ({-gap['now']:.1f} mph off; {kind.lower()}s like his "
                         f"play best at {rng}: {gap['gain']:+.0f}). This is the one case where taking speed "
                         "off makes sense, but it has to come from the grip (deeper in the hand, more pronation, less "
                         "fingertip pressure) with the same arm speed as his fastball, never from easing up. Adding velocity "
                         "to the fastball widens the gap too.")
        else:
            parts.append(f"It plays better closer to his fastball's speed ({rng}; he's {-gap['now']:.1f} off: "
                         f"{gap['gain']:+.0f}). Throw this pitch harder; as his fastball gains velocity, this range "
                         "moves up with it.")
    if velo:
        parts.append(f"More velocity helps too: {velo['target_text']} on this pitch ({velo['gain']:+.0f}, he's at "
                     f"{velo['now']:.1f}), coming from more arm speed across his whole arsenal.")
    if not shape_gaps and not velo and not gap:
        parts.append("The work is landing it and finding its count.")
    plan = " ".join(parts)
    rare = [x for x in rows if x.get("beyond")]
    for x in rare:
        what = {"ivb": "ride" if r["ivb"] >= 0 else "drop", "hb": "sweep" if r["hb"] < 0 else "run"}[x["trait"]]
        plan = (plan + " " if plan else "") + (
            f"Its {what} ({x['now']:+.1f} in) is beyond 97% of MLB {kind.lower()}s, too rare for the data to say "
            f"whether more or less would help.")
    return {"rows": rows, "now_plus": round(now_plus), "pool": f"the {len(pool)} MLB {kind.lower()}s most like his",
            "plan": plan}


# Movement direction (clockwise from straight up toward arm side, degrees). From 2024-25 Statcast,
# comparing each pitcher's pitch to itself at similar total break (within 2 in), whiffs per swing
# when it drifts 8-60 deg one way vs his normal. Only pitch types with a clear direction get a call.
AXIS_RESEARCH = {
    "Four-seam": {"good": -1, "good_word": "more upright", "bad_word": "drifting arm side",
                  "stat": "26% whiffs per swing more upright vs 22% at his normal and 17% drifting arm side"},
    "Cutter": {"good": -1, "good_word": "more cut", "bad_word": "drifting toward ride",
               "stat": "25% whiffs per swing with more cut vs 22% at his normal"},
    "Changeup": {"good": +1, "good_word": "more drop", "bad_word": "drifting toward his fastball's direction",
                 "stat": "35-37% whiffs per swing with more drop vs 32% at his normal and 25-30% toward the fastball"},
    "Curveball": {"good": -1, "good_word": "more 12-6", "bad_word": "flatter (more sideways)",
                  "stat": "37-38% whiffs per swing more 12-6 vs 32% at his normal"},
}
AXIS_SPLIT_DEG = 12      # a pitch this far off his normal direction counts as a different version
AXIS_MIN_VERSION = 5     # pitches needed in a version to describe it


def clock(angle_deg: float, throws: str) -> str:
    """Movement direction as a clock face from behind the pitcher (RHP arm side = 3:00 side)."""
    a = angle_deg if throws != "Left" else -angle_deg
    minutes = int(round(((a % 360) / 30) * 60 / 15) * 15) % 720
    h, m = divmod(minutes, 60)
    return f"{h or 12}:{m:02d}"


def spin_axis(pp: pd.DataFrame, kind: str, throws: str) -> dict | None:
    """
    Good and bad versions of a pitch inside his own data, from the direction it moves.
    Rapsodo (like portable TrackMan) gives movement reliably; the direction it moves is the axis here.
    """
    d = pp.dropna(subset=["InducedVertBreak", "hb_arm", "RelSpeed"])
    if len(d) < 15:
        return None
    ang = np.degrees(np.arctan2(d["hb_arm"], d["InducedVertBreak"]))
    med = float(np.degrees(np.arctan2(np.sin(np.radians(ang)).median(), np.cos(np.radians(ang)).median())))
    dev = (ang - med + 180) % 360 - 180
    q1, q3 = dev.quantile([0.25, 0.75])
    out = {"normal": clock(med, throws), "spread_deg": round(float(q3 - q1)), "n": int(len(d))}
    res = AXIS_RESEARCH.get(kind)

    def version(mask):
        v = d[mask]
        if len(v) < AXIS_MIN_VERSION:
            return None
        a = float(np.degrees(np.arctan2(v["hb_arm"].mean(), v["InducedVertBreak"].mean())))
        return {"clock": clock(a, throws), "share": round(100 * len(v) / len(d)), "n": int(len(v)),
                "ivb": _num(v["InducedVertBreak"].mean()), "hb": _num(v["hb_arm"].mean()), "velo": _num(v["RelSpeed"].mean()),
                "total": _num(np.hypot(v["InducedVertBreak"], v["hb_arm"]).mean()),
                "rel_h": _num(v["RelHeight"].mean(), 2) if "RelHeight" in v else None,
                "rel_s": _num(v["rel_side_arm"].mean(), 2) if "rel_side_arm" in v else None}

    if res is None:
        out["call"] = None
        out["note"] = ("The data doesn't show one direction playing better for this pitch type, so this is "
                       "for reference only.")
        return out
    sign = res["good"]
    good = version(sign * dev >= AXIS_SPLIT_DEG)
    bad = version(-sign * dev >= AXIS_SPLIT_DEG)
    normal = version(dev.abs() < AXIS_SPLIT_DEG)
    out.update({"call": True, "good": good, "bad": bad, "normal_version": normal,
                "good_word": res["good_word"], "bad_word": res["bad_word"], "why": res["stat"]})
    if good:
        out["chase"] = good["clock"]
    if bad and normal:
        dv = bad["velo"] - normal["velo"]
        drel = np.hypot((bad["rel_h"] or 0) - (normal["rel_h"] or 0), (bad["rel_s"] or 0) - (normal["rel_s"] or 0)) * 12
        if abs(dv) >= 1.0 and drel < 3:
            out["drift"] = (f"the {bad['clock']} ones are {abs(dv):.1f} mph {'softer' if dv < 0 else 'harder'} from the same "
                            "release point, so it's effort and the hand, not his body.")
        elif drel >= 3:
            out["drift"] = (f"the {bad['clock']} ones come out {drel:.0f} in from his normal release point, so it starts "
                            "with where his arm is.")
    return out


def swing_split(pp: pd.DataFrame) -> str | None:
    """What separated his whiffs from contact on this pitch, in game swings."""
    sw = pp[pp["is_game"] & pp["swing"].fillna(False).astype(bool)]
    wh, ct = sw[sw["whiff"].fillna(False).astype(bool)], sw[~sw["whiff"].fillna(False).astype(bool)]
    if len(wh) < 4 or len(ct) < 4:
        return None
    diffs = []
    for col, label, unit, floor in [("RelSpeed", "mph", "mph", 0.8), ("InducedVertBreak", "ride", "in", 1.5),
                                    ("hb_arm", "arm-side run", "in", 1.5), ("PlateLocHeight", "height at the plate", "ft", 0.3)]:
        if col not in sw:
            continue
        d = wh[col].mean() - ct[col].mean()
        sd = max(sw[col].std(), 1e-6)
        if abs(d) >= floor:
            diffs.append((abs(d) / sd, col, label, unit, d))
    if not diffs:
        return f"His {len(wh)} whiffs and {len(ct)} contact swings came on the same version of the pitch."
    _, col, label, unit, d = max(diffs)
    if col == "RelSpeed":
        txt = f"his whiffs came on ones thrown {abs(d):.1f} mph {'harder' if d > 0 else 'slower'} than the ones hitters made contact with"
    elif col == "PlateLocHeight":
        txt = f"his whiffs came on ones {abs(d):.1f} ft {'higher' if d > 0 else 'lower'} at the plate than the ones hitters made contact with"
    else:
        txt = f"his whiffs came on ones with {abs(d):.1f} in {'more' if d > 0 else 'less'} {label} than the ones hitters made contact with"
    return f"{txt[0].upper()}{txt[1:]} ({len(wh)} whiffs, {len(ct)} contact)."


def _join(words: list[str]) -> str:
    words = [w.lower() for w in words]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def arsenal_plan(pitches: list[dict], work: dict | None) -> str:
    """One sentence, one order of operations (Garnet's arsenal recommendation)."""
    graded = [p for p in pitches if p.get("graded", isinstance(p["grades"]["pitch_plus"], (int, float)))]
    working = [p["pitch"] for p in graded if p.get("verdict") == "Working"]
    struggling = [p["pitch"] for p in graded if p.get("verdict") == "Struggling"]
    no_job = [p["pitch"] for p in graded if p.get("verdict") == "No clear job yet"]
    parts = []
    if struggling:
        parts.append(f"No new pitch until his {_join(struggling)} {'is' if len(struggling) == 1 else 'are'} fixed (see "
                     f"{'its card' if len(struggling) == 1 else 'their cards'}).")
    elif no_job:
        parts.append(f"No new pitch until his {_join(no_job)} {'has' if len(no_job) == 1 else 'have'} a defined job.")
    if working:
        below = [p for p in graded if p["pitch"] in working and isinstance(p["grades"]["pitch_plus"], (int, float))
                 and p["grades"]["pitch_plus"] < 100]
        keep = [w for w in working if w not in [p["pitch"] for p in below]]
        if keep:
            parts.append(f"Keep the {_join(keep)} as {'it is' if len(keep) == 1 else 'they are'}.")
        for p in below:
            parts.append(f"His {p['pitch'].lower()} is doing its job but grades below average (Pitch+ "
                         f"{p['grades']['pitch_plus']}); the add-ons on its card are how it gets better.")
    thin = [p["pitch"] for p in graded if p.get("verdict") in ("Too few game pitches", "Bullpen only")]
    if thin:
        parts.append(f"His {_join(thin)} need{'s' if len(thin) == 1 else ''} more game pitches before "
                     f"{'it' if len(thin) == 1 else 'they'} can be judged.")
    if graded and all(p.get("verdict") == "Working" for p in graded):
        parts.insert(0, "Everything he throws is doing a job." if len(graded) > 2 else
                     "Both pitches are working; the next piece is a third pitch that moves the opposite way of his breaking ball.")
    return " ".join(parts) if parts else "Not enough game pitches yet to set an arsenal plan."


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
    df = prepare(sessions.graded(df)[lambda d: d["shape_family"].notna()], ref)  # bullpens are stored, not graded
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
        traits = None
        if fb_model is not None:
            h = (heights or {}).get(p["Pitcher"].iloc[0])
            traits = fastball_traits.traits(p, fb_model, fb_offset, h, d3_scale)
        judged, pitches = {}, []
        for idx, r in prow.iterrows():
            kind = r["shape_family"]
            graded = r["n"] >= MIN_TYPE_PITCHES
            entry = {
                "pitch": kind, "subtype": r["subtype"] if isinstance(r["subtype"], str) else None,
                "bucket": r["bucket"], "n": int(r["n"]), "game_pitches": int(r["n_game"]), "usage_pct": _pct(r["usage"]),
                "grades": {
                    "pitch_plus": _num(r["pitch_plus"], 0) if graded and pd.notna(r["pitch_plus"]) else "N/A",
                    "stuff_plus": _num(r["stuff"], 0) if pd.notna(r["stuff"]) else "N/A",
                    "shape_plus": _num(r["shape"], 0) if pd.notna(r["shape"]) else "N/A",
                    "location_plus": _num(r["location"], 0) if pd.notna(r["location"]) else "N/A",
                    "results_plus": _num(r["results"], 0) if pd.notna(r["results"]) else "N/A",
                    "stuff_plus_mlb": _num(r["stuff_mlb"], 0) if pd.notna(r["stuff_mlb"]) else "N/A",
                    "compared_to": {"type": f"other {kind.lower()}s in this file", "group": f"{r['bucket'].lower()}s in this file",
                                    "all": "all pitches in this file", "mlb": f"MLB {kind.lower()}s"}.get(r["stuff_basis"], "N/A"),
                },
            }
            entry["graded"] = bool(graded)
            if graded and int(r["n_game"]) < MIN_GAME_FOR_PITCH_PLUS:
                entry["pitch_plus_gate"] = {"needed": MIN_GAME_FOR_PITCH_PLUS, "has": int(r["n_game"])}
            if not graded:
                entry["verdict"] = "Rarely used"
                entry["so"] = f"Only {int(r['n'])} thrown; not enough to grade."
                pitches.append(entry)
                continue
            parts = expl.get(idx, {})
            entry["results_weight"] = round(100 * r["n_game"] / (r["n_game"] + r["results_k"]))

            if r.get("results_tier"):
                entry["results_tier"] = r["results_tier"]
            entry["shape"] = {"velo": _num(r["velo"]), "ivb": _num(r["ivb"]), "hb_arm": _num(r["hb"]),
                              "spin": _num(r["spin"], 0), "velo_diff": _num(r.get("velo_diff")),
                              "rel_height": _num(r.get("rel_height"), 2)}
            if parts:
                entry["stuff_explained"] = ", ".join(f"{k} {v:+.0f}" for k, v in parts.items())
                entry["stuff_parts"] = {k: round(v) for k, v in parts.items()}
                weakest = min(parts, key=parts.get)
                flat_fb = (traits is not None and traits["pitch"] == kind and weakest == "Release"
                           and traits["vaa_ae_pct_mlb"] >= fastball_traits.HIGH)
                if parts[weakest] <= -5 and not flat_fb:
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
        dev = development(p, prow, judged, sit, df, traits)
        throws = p["PitcherThrows"].iloc[0]
        for e in pitches:
            if not e["graded"]:
                continue
            row = prow[prow["shape_family"] == e["pitch"]].iloc[0]
            pp = p[p["shape_family"] == e["pitch"]]
            e["targets"] = pitch_targets(row, mlb_rows, models, ref)
            e["spin_axis"] = spin_axis(pp, e["pitch"], throws)
            e["swing_split"] = swing_split(pp)
        work = pitch_needing_work(pitches)
        if work is not None:
            work["targets"] = next(e.get("targets") for e in pitches if e["pitch"] == work["pitch"])
        from prism_data.devices import describe, device_of

        devs = device_of(p).value_counts()
        report = {
            "pitcher": name, "pitcher_id": str(pid), "throws": p["PitcherThrows"].iloc[0],
            "devices": [dict(describe(k), pitches=int(v)) for k, v in devs.items()],
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
            "needs_work": work,
            "arsenal_plan": arsenal_plan(pitches, work),
            "points": p[["shape_family", "InducedVertBreak", "hb_arm", "RelSpeed"]].dropna().round(1).to_dict("records"),
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
          "| Pitch | Subtype | n | Game | Usage | Pitch+ | Stuff+ | Location+ | Results+ | Shape+ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in r["pitch_by_pitch"]:
        g = p["grades"]
        L.append(f"| {p['pitch']} | {_cell(p['subtype'])} | {p['n']} | {p['game_pitches']} | {_cell(p['usage_pct'])}% | "
                 f"{g['pitch_plus']} | {g['stuff_plus']} | {g['location_plus']} | {g['results_plus']} | {g['shape_plus']} |")

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
