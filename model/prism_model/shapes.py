"""
Pitch classification: bucket, pitch type, and subtype.

PRISM ignores device pitch tags. Every pitch is classified from its movement
and speed relative to the pitcher's own primary fastball, so the same rules
work for an MLB arm and a Division III arm. Three levels:

  Bucket     Fastball | Breaking | Offspeed   (Stuff+ compares within a bucket)
  Pitch type Four-seam, Sinker, Cutter | Slider, Curveball | Changeup, Splitter, Knuckleball
  Subtype    e.g. Ride four-seam, Sweeper, 12-6 curveball, Fade changeup

Subtypes are descriptive tags only. They never change a grade: a pitch is
graded on how well it works within its bucket, not on whether it fits the
usual shape for its name.

How it works
------------
1. ``add_fastball_reference`` finds each pitcher's primary fastball from their
   hardest pitches (no tags used) and records its speed and movement.
2. ``shape_features`` describes every pitch relative to that fastball, plus its
   own movement and spin direction. Raw velocity is deliberately excluded so a
   slower D3 fastball still reads as a fastball.
3. A classifier learns the pitch types from MLB's pitch tags, then predicts a
   pitch type for any pitch from its shape alone.
4. ``assign_subtypes`` applies movement thresholds (``SUBTYPE_RULES``) to each
   pitcher's average shape for each pitch type, so a pitcher's slider gets one
   subtype rather than flipping pitch to pitch.

Usage (from the model/ folder):
    python -m prism_model.shapes train      # train, validate, save the classifier
    python -m prism_model.shapes classify   # add shape_family to every Statcast season
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

ARTIFACT_DIR = DATA_DIR / "models"
CLASSIFIER_PATH = ARTIFACT_DIR / "shape_classifier.joblib"
REPORT_PATH = ARTIFACT_DIR / "shape_classifier_report.json"

# Statcast pitch-type codes -> PRISM pitch type. Sweepers are slider subtypes;
# slurves are curveball subtypes; two-seamers are sinkers. Knuckleballs are
# assigned by a spin rule rather than learned. Unlisted codes (eephus,
# pitchout, unknown) are left out of training.
FAMILY = {
    "FF": "Four-seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Slider",
    "SV": "Curveball",
    "CU": "Curveball",
    "KC": "Curveball",
    "CS": "Curveball",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Splitter",
}
FAMILIES = ["Four-seam", "Sinker", "Cutter", "Slider", "Curveball", "Changeup", "Splitter"]

BUCKET = {
    "Four-seam": "Fastball",
    "Sinker": "Fastball",
    "Cutter": "Fastball",
    "Slider": "Breaking",
    "Curveball": "Breaking",
    "Changeup": "Offspeed",
    "Splitter": "Offspeed",
    "Knuckleball": "Offspeed",
}

# A pitch with less total spin than this is a knuckleball, whatever its shape.
KNUCKLEBALL_MAX_RPM = 400


def is_slurve(m: dict) -> bool:
    """Roughly equal sweep and drop: its own shape, filed under curveballs."""
    return m["glove"] >= 6 and m["ivb"] <= -4 and m["glove"] >= -m["ivb"] * 0.9


# Subtype tags, applied to a pitcher's median shape for each pitch type.
# Movement in inches; "glove" is glove-side break (positive = toward the glove
# side). Rules are checked in order and the first match wins; the last entry is
# the default. Pitch types not listed here get no subtype. Starting points to
# tune by eye on real staffs.
SUBTYPE_RULES = {
    "Four-seam": [
        ("Ride", lambda m: m["ivb"] >= 14),
        ("Flat", lambda m: True),
    ],
    "Slider": [
        ("Sweeper", lambda m: m["glove"] >= 10 and m["ivb"] > -4),
        ("Gyro", lambda m: m["glove"] < 5),
        ("Traditional", lambda m: True),
    ],
    "Curveball": [
        ("Slurve", is_slurve),
        ("12-6", lambda m: m["glove"] < 5),
        ("Standard", lambda m: True),
    ],
    "Changeup": [
        # More arm-side run than the fastball vs. spin killed mainly for drop.
        ("Fade", lambda m: m["hb_diff"] >= 4),
        ("Drop", lambda m: True),
    ],
}

FEATURES = [
    "velo_diff",         # mph slower (negative) or faster than the primary fastball
    "InducedVertBreak",  # in
    "hb_arm",            # in, positive = arm side
    "ivb_diff",          # in, relative to the primary fastball
    "hb_diff",           # in, relative to the primary fastball
    "axis_sin",          # spin direction, mirrored for lefties
    "axis_cos",
    "spin_ratio",        # spin rate / the primary fastball's spin rate
]

# A pitcher-season (or session) needs this many tracked pitches to get a
# fastball reference.
MIN_PITCHES = 20


def add_fastball_reference(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """
    Add the primary fastball's speed and movement to every pitch.

    The primary fastball is taken as the pitcher's hardest pitches: those
    within 3 mph of their 90th-percentile speed. Its reference values are the
    medians of those pitches. ``by`` sets the grouping (default: pitcher and
    season; for a single bullpen session, pass the session's grouping).
    """
    by = by or ["PitcherId", "Season"]
    df = df.copy()
    speed = df["RelSpeed"]
    p90 = df.groupby(by)["RelSpeed"].transform(lambda s: s.quantile(0.9))
    hard = speed >= p90 - 3

    ref = (
        df[hard]
        .groupby(by)
        .agg(
            fb_velo=("RelSpeed", "median"),
            fb_ivb=("InducedVertBreak", "median"),
            fb_hb=("hb_arm", "median"),
            fb_spin=("SpinRate", "median"),
        )
        .reset_index()
    )
    counts = df.groupby(by)["RelSpeed"].count().rename("n_tracked").reset_index()
    df = df.merge(ref, on=by, how="left").merge(counts, on=by, how="left")
    too_few = df["n_tracked"] < MIN_PITCHES
    df.loc[too_few, ["fb_velo", "fb_ivb", "fb_hb", "fb_spin"]] = np.nan
    return df.drop(columns="n_tracked")


def shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features describing each pitch's shape relative to the primary fastball."""
    out = pd.DataFrame(index=df.index)
    out["velo_diff"] = df["RelSpeed"] - df["fb_velo"]
    out["InducedVertBreak"] = df["InducedVertBreak"]
    out["hb_arm"] = df["hb_arm"]
    out["ivb_diff"] = df["InducedVertBreak"] - df["fb_ivb"]
    out["hb_diff"] = df["hb_arm"] - df["fb_hb"]
    # Spin axis in degrees; mirror lefties so both hands share one frame.
    axis = df["SpinAxis"].where(df["PitcherThrows"] == "R", 360 - df["SpinAxis"])
    rad = np.radians(axis.astype(float))
    out["axis_sin"] = np.sin(rad)
    out["axis_cos"] = np.cos(rad)
    # Relative spin keeps the changeup/splitter signal without penalizing
    # arms that spin the ball less overall.
    out["spin_ratio"] = df["SpinRate"] / df["fb_spin"]
    return out[FEATURES]


def load_statcast(years: list[int]) -> pd.DataFrame:
    cols = ["Season", "PitcherId", "PitcherThrows", "TaggedPitchType", "RelSpeed",
            "InducedVertBreak", "hb_arm", "SpinAxis", "SpinRate"]
    df = pd.concat(
        [pd.read_parquet(PROCESSED_DIR / f"statcast_{y}.parquet", columns=cols) for y in years],
        ignore_index=True,
    )
    return df.dropna(subset=["RelSpeed", "InducedVertBreak", "hb_arm"])


def train(years: list[int]) -> dict:
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit

    df = add_fastball_reference(load_statcast(years))
    df["family"] = df["TaggedPitchType"].map(FAMILY)
    df = df.dropna(subset=["family", "fb_velo"])
    X, y = shape_features(df), df["family"]

    # Hold out 20% of pitchers entirely, so the score reflects pitchers the
    # model has never seen.
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    tr, te = next(split.split(X, y, groups=df["PitcherId"]))

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=7)
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])

    # PRISM grades a pitcher's pitch type as a whole, so also score the
    # majority prediction for each held-out pitcher's pitch type.
    held = df.iloc[te][["PitcherId", "Season", "TaggedPitchType", "family"]].copy()
    held["pred"] = pred
    by_type = held.groupby(["PitcherId", "Season", "TaggedPitchType"]).agg(
        family=("family", "first"), pred=("pred", lambda s: s.mode().iloc[0]), n=("pred", "size")
    )
    by_type = by_type[by_type["n"] >= 25]

    report = {
        "years": years,
        "train_pitches": int(len(tr)),
        "test_pitches": int(len(te)),
        "held_out_pitchers": int(df["PitcherId"].iloc[te].nunique()),
        "accuracy": round(float(accuracy_score(y.iloc[te], pred)), 4),
        "pitch_type_accuracy": round(float((by_type["family"] == by_type["pred"]).mean()), 4),
        "pitch_types_scored": int(len(by_type)),
        "accuracy_by_family": {
            fam: round(float((pred[y.iloc[te].values == fam] == fam).mean()), 4)
            for fam in FAMILIES if (y.iloc[te] == fam).any()
        },
        "confusion": {
            "labels": FAMILIES,
            "matrix": confusion_matrix(y.iloc[te], pred, labels=FAMILIES).tolist(),
        },
    }

    # Refit on everyone for the saved model.
    clf.fit(X, y)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, CLASSIFIER_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def subtype_for(pitch_type: str, shape: dict) -> str | None:
    """The subtype for one pitch type given its median shape, or None if unknown."""
    rules = SUBTYPE_RULES.get(pitch_type)
    if not rules or any(pd.isna(v) for v in shape.values()):
        return None
    for name, test in rules:
        if test(shape):
            return name
    return None


def assign_subtypes(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """
    Add ``subtype`` from each pitcher's median shape for each pitch type.

    Expects ``shape_family`` plus the shape features already computed.
    """
    by = (by or ["PitcherId", "Season"]) + ["shape_family"]
    feats = shape_features(df)
    work = pd.concat([df[by], feats], axis=1).dropna(subset=["shape_family"])
    med = work.groupby(by)[["InducedVertBreak", "hb_arm", "hb_diff", "spin_ratio"]].median().reset_index()
    shapes = [
        {"ivb": r.InducedVertBreak, "glove": -r.hb_arm, "hb_diff": r.hb_diff, "spin_ratio": r.spin_ratio}
        for r in med.itertuples()
    ]
    # A slurve-shaped "slider" is filed as a curveball, so every slurve shares one tag.
    med["new_family"] = [
        "Curveball" if fam == "Slider" and not any(pd.isna(v) for v in m.values()) and is_slurve(m) else fam
        for fam, m in zip(med["shape_family"], shapes)
    ]
    med["subtype"] = [subtype_for(fam, m) for fam, m in zip(med["new_family"], shapes)]
    df = df.merge(med[by + ["new_family", "subtype"]], on=by, how="left")
    df["shape_family"] = df["new_family"].where(df["new_family"].notna(), df["shape_family"])
    df["bucket"] = df["shape_family"].map(BUCKET)
    return df.drop(columns="new_family")


MERGE_VELO_MPH = 3.0
MERGE_MOVEMENT_IN = 5.0


def consolidate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Merge pitch-by-pitch label splits within a pitcher.

    A borderline pitch can be labeled two ways from one pitch to the next (a
    sinking four-seam called four-seam on some pitches and sinker on others).
    Within each pitcher and group, a smaller pitch type whose average speed and
    movement sit on top of a larger one (within 3 mph and 5 in) is folded
    into it. Genuinely different pitches (an MLB four-seam and sinker are
    usually ~10 in apart) are left alone.
    """
    df = df.copy()
    bucket = df["shape_family"].map(BUCKET)
    for _, grp in df[df["shape_family"].notna() & (df["shape_family"] != "Knuckleball")].groupby(by):
        g_bucket = bucket.loc[grp.index]
        for b in g_bucket.dropna().unique():
            sub = grp[g_bucket == b]
            stats = sub.groupby("shape_family").agg(n=("RelSpeed", "size"), conf=("shape_conf", "sum"), velo=("RelSpeed", "mean"),
                                                    ivb=("InducedVertBreak", "mean"), hb=("hb_arm", "mean"))
            stats = stats.sort_values(["n", "conf"], ascending=False)
            keep = []
            for kind, r in stats.iterrows():
                target = next((k for k in keep
                               if abs(r["velo"] - stats.loc[k, "velo"]) <= MERGE_VELO_MPH
                               and np.hypot(r["ivb"] - stats.loc[k, "ivb"], r["hb"] - stats.loc[k, "hb"]) <= MERGE_MOVEMENT_IN), None)
                if target is None:
                    keep.append(kind)
                else:
                    df.loc[sub.index[sub["shape_family"] == kind], "shape_family"] = target
    return df


def classify(df: pd.DataFrame, by: list[str] | None = None, merge_splits: bool = False) -> pd.DataFrame:
    """
    Add fastball reference, ``shape_family`` and ``shape_conf`` to a PRISM-format frame.

    ``merge_splits`` folds near-identical pitch types within a pitcher together
    (see ``consolidate``); used for team reports, not for MLB training data.
    """
    from joblib import load

    clf = load(CLASSIFIER_PATH)
    df = add_fastball_reference(df, by=by)
    feats = shape_features(df)
    ok = feats[["velo_diff", "InducedVertBreak", "hb_arm"]].notna().all(axis=1)
    df["shape_family"] = pd.NA
    df["shape_conf"] = np.nan
    if ok.any():
        proba = clf.predict_proba(feats[ok])
        df.loc[ok, "shape_family"] = clf.classes_[proba.argmax(axis=1)]
        df.loc[ok, "shape_conf"] = proba.max(axis=1)
    knuckle = df["SpinRate"] < KNUCKLEBALL_MAX_RPM
    df.loc[knuckle, "shape_family"] = "Knuckleball"
    if merge_splits:
        df = consolidate(df, by or ["PitcherId", "Season"])
    df["bucket"] = df["shape_family"].map(BUCKET)
    return assign_subtypes(df, by=by)


def print_subtype_summary(df: pd.DataFrame) -> None:
    """How subtypes line up with MLB's tags, as a sanity check on the thresholds."""
    pitched = df.dropna(subset=["subtype"])
    print("\nSubtype share within each pitch type:")
    share = pitched.groupby("shape_family")["subtype"].value_counts(normalize=True).round(3)
    print(share.to_string())
    print("\nMLB tag vs PRISM subtype (row share):")
    cross = pd.crosstab(pitched["TaggedPitchType"], pitched["subtype"], normalize="index").round(2)
    print(cross.loc[[t for t in ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS", "KN"] if t in cross.index]].to_string())
    print("\nPitch type share by MLB tag (row share):")
    typed = df.dropna(subset=["shape_family"])
    cross2 = pd.crosstab(typed["TaggedPitchType"], typed["shape_family"], normalize="index").round(2)
    print(cross2.loc[[t for t in ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS", "KN"] if t in cross2.index]].to_string())


def print_report(report: dict) -> None:
    print(f"Held-out pitchers: {report['held_out_pitchers']:,} ({report['test_pitches']:,} pitches)")
    print(f"Pitch-by-pitch agreement with MLB tags: {report['accuracy']:.1%}")
    print(f"Pitch-type agreement (majority vote, 25+ pitches): {report['pitch_type_accuracy']:.1%} "
          f"of {report['pitch_types_scored']:,} pitcher pitch types")
    for fam, acc in report["accuracy_by_family"].items():
        print(f"  {fam:<10} {acc:.1%}")
    labels, matrix = report["confusion"]["labels"], report["confusion"]["matrix"]
    print("\nMost common mix-ups (true -> predicted):")
    mixups = [(matrix[i][j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    for n, a, b in sorted(mixups, reverse=True)[:6]:
        print(f"  {a} -> {b}: {n:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "classify"])
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()

    if args.command == "train":
        print_report(train(args.years))
        print(f"\nSaved classifier to {CLASSIFIER_PATH}")
    else:
        for year in args.years:
            path = PROCESSED_DIR / f"statcast_{year}.parquet"
            out = classify(pd.read_parquet(path))
            out.to_parquet(PROCESSED_DIR / f"statcast_{year}_shapes.parquet", index=False)
            print(f"{year}: classified {out['shape_family'].notna().sum():,} pitches")
            if year == args.years[-1]:
                print_subtype_summary(out)


if __name__ == "__main__":
    main()
