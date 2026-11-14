"""
Stuff+ and Shape+.

Follows the Garnet Pitch Lab model card:

- Target: expected run value (xRV) per 100 pitches. Balls, strikes and whiffs
  keep their actual run value; balls in play are valued by count and expected
  wOBA, because hit outcomes carry too much luck to learn from.
- Training rows: pitcher-season pitch types with 100+ pitches, weighted by
  pitch count.
- One gradient-boosted model per pitch group (fastball, breaking, offspeed),
  monotonic in velocity so a harder pitch never grades worse than the same
  pitch thrown softer.
- Features: fastballs use perceived velocity, IVB, HB, spin, release height
  and side. Secondaries use perceived velocity, IVB, HB and spin plus velocity,
  IVB and HB differences off the pitcher's fastball.
- Shape+: the same models with velocity removed.
- Validation: 5-fold cross-validation grouped by pitcher, scored only on
  out-of-fold predictions.

Grades are put on a 100 scale against a reference group (a staff, D3, or MLB)
for each pitch type: 100 = that group's average, 10 points = one standard
deviation. Lower expected runs is better, so the sign is flipped.

Usage (from the model/ folder):
    python -m prism_model.stuff train
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

ARTIFACT_DIR = DATA_DIR / "models"
MODELS_PATH = ARTIFACT_DIR / "stuff_models.joblib"
REPORT_PATH = ARTIFACT_DIR / "stuff_report.json"

MIN_PITCHES = 100
MLB_AVG_EXTENSION_FT = 6.3
MPH_PER_FT_EXTENSION = 1.6  # each extra foot of extension plays like ~1.6 mph

GROUPS = {
    "Fastball": {
        "types": ["Four-seam", "Sinker", "Cutter"],
        "features": ["perceived_velo", "ivb", "hb", "spin", "rel_height", "rel_side"],
    },
    "Breaking": {
        "types": ["Slider", "Curveball"],
        "features": ["perceived_velo", "ivb", "hb", "spin", "velo_diff", "ivb_diff", "hb_diff"],
    },
    "Offspeed": {
        "types": ["Changeup", "Splitter"],
        "features": ["perceived_velo", "ivb", "hb", "spin", "velo_diff", "ivb_diff", "hb_diff"],
    },
}
VELO_FEATURE = "perceived_velo"

LOAD_COLUMNS = [
    "Season", "PitcherId", "Pitcher", "PitcherThrows", "shape_family", "subtype",
    "RelSpeed", "Extension", "InducedVertBreak", "hb_arm", "SpinRate", "RelHeight",
    "rel_side_arm", "fb_velo", "fb_ivb", "fb_hb", "Balls", "Strikes",
    "PitchCall", "RunValue", "xwOBA",
]


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
def add_xrv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected run value for every pitch.

    Non-contact pitches keep their actual run value. A ball in play gets the
    average run value of balls in play hit with similar expected wOBA in the
    same count, which removes defense and luck from the outcome.
    """
    df = df.copy()
    df["count"] = df["Balls"].astype(str) + "-" + df["Strikes"].astype(str)
    bip = df["PitchCall"] == "InPlay"
    df["xwoba_bin"] = (df["xwOBA"].clip(0, 2) / 0.025).round()

    table = df[bip & df["xwOBA"].notna()].groupby(["count", "xwoba_bin"])["RunValue"].mean()
    by_count = df[bip].groupby("count")["RunValue"].mean()

    df["xRV"] = df["RunValue"]
    keyed = pd.MultiIndex.from_frame(df.loc[bip, ["count", "xwoba_bin"]])
    expected = pd.Series(table.reindex(keyed).to_numpy(), index=df.index[bip])
    fallback = df.loc[bip, "count"].map(by_count)
    df.loc[bip, "xRV"] = expected.fillna(fallback)
    return df.drop(columns=["xwoba_bin"])


# --------------------------------------------------------------------------
# Pitcher-season pitch types
# --------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, by: list[str] | None = None, min_pitches: int = MIN_PITCHES) -> pd.DataFrame:
    """One row per pitcher-season pitch type: average shape, pitch count, and xRV per 100."""
    by = by or ["PitcherId", "Season", "shape_family"]
    d = df.dropna(subset=["shape_family", "RelSpeed", "InducedVertBreak", "hb_arm"]).copy()
    d["perceived_velo"] = d["RelSpeed"] + MPH_PER_FT_EXTENSION * (d["Extension"] - MLB_AVG_EXTENSION_FT)
    d["perceived_velo"] = d["perceived_velo"].fillna(d["RelSpeed"])

    agg = d.groupby(by).agg(
        n=("RelSpeed", "size"),
        pitcher=("Pitcher", "first"),
        subtype=("subtype", "first"),
        velo=("RelSpeed", "mean"),
        perceived_velo=("perceived_velo", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("hb_arm", "mean"),
        spin=("SpinRate", "mean"),
        rel_height=("RelHeight", "mean"),
        rel_side=("rel_side_arm", "mean"),
        fb_velo=("fb_velo", "first"),
        fb_ivb=("fb_ivb", "first"),
        fb_hb=("fb_hb", "first"),
        xrv100=("xRV", "mean") if "xRV" in d else ("RelSpeed", lambda s: np.nan),
    ).reset_index()
    agg["xrv100"] = agg["xrv100"] * 100
    agg["velo_diff"] = agg["velo"] - agg["fb_velo"]
    agg["ivb_diff"] = agg["ivb"] - agg["fb_ivb"]
    agg["hb_diff"] = agg["hb"] - agg["fb_hb"]
    agg["group"] = agg["shape_family"].map({t: g for g, spec in GROUPS.items() for t in spec["types"]})
    return agg[agg["n"] >= min_pitches].reset_index(drop=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Candidate settings, from flexible to very conservative. Pitch-level run value
# is noisy, so the simpler settings usually win; ``train`` picks by
# cross-validated rank correlation.
CONFIGS = {
    "flexible": dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
    "moderate": dict(max_iter=200, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=80, l2_regularization=5.0),
    "conservative": dict(max_iter=150, learning_rate=0.03, max_leaf_nodes=5, min_samples_leaf=150, l2_regularization=10.0),
    "very_conservative": dict(max_iter=100, learning_rate=0.03, max_leaf_nodes=4, min_samples_leaf=250, l2_regularization=20.0),
}
DEFAULT_CONFIG = "moderate"


def make_model(features: list[str], config: str = DEFAULT_CONFIG):
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Lower expected runs is better, so velocity must push predictions down.
    monotonic = [-1 if f == VELO_FEATURE else 0 for f in features]
    return HistGradientBoostingRegressor(**CONFIGS[config], monotonic_cst=monotonic, random_state=7)


def evaluate(rows: pd.DataFrame, features: list[str], config: str = DEFAULT_CONFIG) -> dict:
    """5-fold, pitcher-grouped cross-validation scored on out-of-fold predictions."""
    from sklearn.model_selection import GroupKFold

    X, y, w = rows[features], rows["xrv100"], rows["n"]
    oof = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=rows["PitcherId"]):
        m = make_model(features, config).fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        oof[te] = m.predict(X.iloc[te])

    resid = y - oof
    r2 = 1 - np.average(resid ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    rank_corr = pd.Series(oof).corr(y.reset_index(drop=True), method="spearman")
    q = pd.qcut(oof, 5, labels=False)
    top = np.average(y[q == 0], weights=w[q == 0])      # lowest predicted runs = best stuff
    bottom = np.average(y[q == 4], weights=w[q == 4])
    return {
        "config": config,
        "rows": int(len(rows)),
        "r2": round(float(r2), 3),
        "rank_corr": round(float(rank_corr), 3),
        "top_vs_bottom_quintile_runs_per_100": round(float(bottom - top), 2),
    }


def train(years: list[int]) -> dict:
    from joblib import dump

    df = pd.concat(
        [pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=LOAD_COLUMNS) for y in years],
        ignore_index=True,
    )
    df = df[df["PitchCall"].notna() & df["RunValue"].notna()]
    rows = aggregate(add_xrv(df))

    models, report = {}, {"years": years, "min_pitches": MIN_PITCHES, "groups": {}}
    for group, spec in GROUPS.items():
        g = rows[rows["group"] == group].dropna(subset=spec["features"] + ["xrv100"])
        shape_features = [f for f in spec["features"] if f != VELO_FEATURE]
        # Pick the setting that ranks held-out pitchers best for this group.
        trials = {c: evaluate(g, spec["features"], c) for c in CONFIGS}
        best = max(trials, key=lambda c: trials[c]["rank_corr"])
        print(f"{group}: " + ", ".join(f"{c} {t['rank_corr']:.3f}" for c, t in trials.items()) + f" -> {best}")
        report["groups"][group] = {
            "pitch_types": g["shape_family"].value_counts().to_dict(),
            "stuff": trials[best],
            "shape": evaluate(g, shape_features, best),
        }
        models[group] = {
            "config": best,
            "features": spec["features"],
            "stuff": make_model(spec["features"], best).fit(g[spec["features"]], g["xrv100"], sample_weight=g["n"]),
            "shape_features": shape_features,
            "shape": make_model(shape_features, best).fit(g[shape_features], g["xrv100"], sample_weight=g["n"]),
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(models, MODELS_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------
# Smoothing. A boosted-tree model is a staircase: a pitch whose averages move by a sampling-sized amount
# (a few tenths of a mph, half an inch, 20 rpm) can step across a split and move several Stuff+ points
# (out of fold on MLB: 3.3 points typical, 7.4 at the 95th percentile). So every prediction is the
# average over a small cloud of nearby pitches, the same cloud for every pitch. That halved the jumps
# (1.7 typical, 3.6 at the 95th) and predicted next-season run value slightly better (0.280 against
# 0.271), and velocity still never lowers a grade. Widths are about the sampling error of a 30-pitch average.
SMOOTH = {"perceived_velo": 0.5, "ivb": 1.0, "hb": 1.0, "spin": 40.0, "rel_height": 0.1, "rel_side": 0.1}
SMOOTH_DIFF = {"perceived_velo": "velo_diff", "ivb": "ivb_diff", "hb": "hb_diff"}  # the gap moves with the pitch
SMOOTH_DRAWS = 128


def _smooth_offsets() -> dict:
    """The fixed cloud: SMOOTH_DRAWS offsets per feature, in mirrored pairs so each averages exactly zero."""
    rng = np.random.default_rng(7)
    out = {}
    for f, sd in SMOOTH.items():
        half = rng.normal(0, sd, SMOOTH_DRAWS // 2)
        out[f] = np.concatenate([half, -half])
    return out


_OFFSETS = _smooth_offsets()


def smooth_predict(model, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    """``model``'s prediction for each row of ``X``, averaged over the fixed cloud of nearby pitches."""
    X = X[features].astype(float).reset_index(drop=True)
    n = len(X)
    big = pd.concat([X] * SMOOTH_DRAWS, ignore_index=True)
    for f, off in _OFFSETS.items():
        step = np.repeat(off, n)
        if f in big:
            big[f] = big[f] + step
        if SMOOTH_DIFF.get(f) in big:
            big[SMOOTH_DIFF[f]] = big[SMOOTH_DIFF[f]] + step
    return model.predict(big).reshape(SMOOTH_DRAWS, n).mean(axis=0)


def predict(rows: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Add predicted xRV per 100 for Stuff (``stuff_xrv``) and Shape (``shape_xrv``), smoothed (see SMOOTH)."""
    if models is None:
        from joblib import load

        models = load(MODELS_PATH)
    rows = rows.copy()
    rows["stuff_xrv"] = np.nan
    rows["shape_xrv"] = np.nan
    for group, m in models.items():
        mask = rows["group"] == group
        ok = mask & rows[m["features"]].notna().all(axis=1)
        if ok.any():
            rows.loc[ok, "stuff_xrv"] = smooth_predict(m["stuff"], rows.loc[ok], m["features"])
            rows.loc[ok, "shape_xrv"] = smooth_predict(m["shape"], rows.loc[ok], m["shape_features"])
    return rows


def to_plus(xrv: pd.Series, ref_mean: float, ref_sd: float) -> pd.Series:
    """Put predicted runs on a 100 scale: 100 = reference average, 10 = one SD, higher = better."""
    if not ref_sd or np.isnan(ref_sd):
        return pd.Series(np.nan, index=xrv.index)
    return 100 - 10 * (xrv - ref_mean) / ref_sd


def scale_within(rows: pd.DataFrame, reference: pd.DataFrame, col: str, by: str = "shape_family") -> pd.Series:
    """Scale ``col`` against ``reference``, separately for each pitch type."""
    stats = reference.groupby(by)[col].agg(["mean", "std"])
    out = pd.Series(np.nan, index=rows.index)
    for kind, grp in rows.groupby(by):
        if kind in stats.index:
            out.loc[grp.index] = to_plus(grp[col], stats.loc[kind, "mean"], stats.loc[kind, "std"])
    return out


def print_report(report: dict) -> None:
    print(f"\nTraining rows: pitcher-season pitch types with {report['min_pitches']}+ pitches, seasons {report['years']}\n")
    print(f"{'Group':<10}{'rows':>7}{'R²':>8}{'rank r':>9}{'top-bottom runs/100':>22}")
    for group, r in report["groups"].items():
        for kind in ("stuff", "shape"):
            e = r[kind]
            label = f"{group} {'Stuff+' if kind == 'stuff' else 'Shape+'}"
            print(f"{label:<22}{e['rows']:>5}{e['r2']:>8.3f}{e['rank_corr']:>9.3f}{e['top_vs_bottom_quintile_runs_per_100']:>14.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()
    print_report(train(args.years))
    print(f"\nSaved models to {MODELS_PATH}")


if __name__ == "__main__":
    main()
