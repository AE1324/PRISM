"""
Grade a session file: classify every pitch and produce Stuff+ and Shape+ for
each pitcher's pitch types.

Usage (from the model/ folder):
    python -m prism_model.grade path/to/trackman.csv
    python -m prism_model.grade path/to/trackman.csv --min-pitches 15

Each pitch type gets two scales:
  vs group  100 = average for that pitch type among the pitchers in this file
            (for a team's own files, this is the staff scale)
  vs MLB    100 = MLB average for that pitch type
Results are saved to data/grades/<file name>_grades.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prism_data.paths import DATA_DIR
from prism_model import references
from prism_data.trackman import read_trackman
from prism_model.shapes import classify
from prism_model.stuff import aggregate, predict, scale_within, to_plus

GRADES_DIR = DATA_DIR / "grades"


def mlb_reference() -> pd.DataFrame:
    """Mean and SD of predicted xRV for each pitch type among MLB pitcher-seasons."""
    ref, _ = references.load()
    return pd.DataFrame([{"shape_family": k, **v} for k, v in ref["scale"].items()])


def grade(df: pd.DataFrame, min_pitches: int = 20) -> pd.DataFrame:
    """Stuff+/Shape+ for each pitcher's pitch types in a PRISM-format frame."""
    df = classify(df, by=["PitcherId"])
    rows = aggregate(df, by=["PitcherId", "shape_family"], min_pitches=min_pitches)
    rows = predict(rows)

    rows["stuff_plus_group"] = scale_within(rows, rows, "stuff_xrv")
    rows["shape_plus_group"] = scale_within(rows, rows, "shape_xrv")

    ref = mlb_reference().set_index("shape_family")
    rows["stuff_plus_mlb"] = pd.NA
    for kind, grp in rows.groupby("shape_family"):
        if kind in ref.index:
            r = ref.loc[kind]
            rows.loc[grp.index, "stuff_plus_mlb"] = to_plus(grp["stuff_xrv"], r["stuff_xrv_mean"], r["stuff_xrv_std"])
    rows["stuff_plus_mlb"] = rows["stuff_plus_mlb"].astype(float)

    keep = ["pitcher", "PitcherId", "shape_family", "subtype", "n", "velo", "ivb", "hb", "spin",
            "stuff_plus_group", "shape_plus_group", "stuff_plus_mlb"]
    out = rows[keep].sort_values(["PitcherId", "n"], ascending=[True, False]).reset_index(drop=True)
    for col in ["velo", "ivb", "hb"]:
        out[col] = out[col].round(1)
    out["spin"] = out["spin"].round(0)
    for col in ["stuff_plus_group", "shape_plus_group", "stuff_plus_mlb"]:
        out[col] = out[col].round(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--min-pitches", type=int, default=20)
    args = parser.parse_args()

    df = read_trackman(args.path)
    out = grade(df, args.min_pitches)
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    dest = GRADES_DIR / f"{Path(args.path).stem}_grades.csv"
    out.to_csv(dest, index=False)

    pd.set_option("display.width", 200)
    print(f"{df['PitcherId'].nunique()} pitchers, {len(df):,} pitches; graded {len(out)} pitch types "
          f"with {args.min_pitches}+ pitches\n")
    print(out.drop(columns=["PitcherId"]).to_string(index=False))
    print(f"\nSaved to {dest}")


if __name__ == "__main__":
    main()
