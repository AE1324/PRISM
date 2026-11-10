"""
Build a team's pitch data from Rapsodo exports and AWRE pitch logs.

Steps
  1. Read every Rapsodo export (movement filled, device moves corrected).
  2. Flag misreads that don't need pitch types (prism_data.clean.flag_basic).
  3. Classify the clean pitches by shape, merging within-pitcher label splits.
  4. Flag shape outliers and one-off shapes (flag_shapes).
  5. Apply saved review decisions (keep/drop by pitch id).
  6. Match AWRE results to Rapsodo pitches and attach count and result.
  7. Add hitter handedness from a roster file, when there is one.
  8. Save the data and a review workbook.

Files (all inside the gitignored data/ folder)
  data/rapsodo/*.csv                         Rapsodo player exports
  data/awre/*PitchData-*.xls                 AWRE pitch logs
  data/rapsodo/review_decisions.csv          saved keep/drop calls (or a filled-in review workbook)
  data/rosters/roster.csv                    optional: Name, Bats (R/L/S), Throws, Position, Height
  data/rosters/inactive_pitchers.txt         optional: one name per line, left out entirely
  -> data/processed/<name>.parquet           all pitches, with issues, exclude flag and results
  -> data/review/<name>_review.xlsx          the review workbook

Usage (from the model/ folder):
    python -m prism_model.team_data --name oberlin_fall_2026
    python -m prism_model.team_data --name oberlin_fall_2026 --decisions ../data/review/oberlin_fall_2026_review.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prism_data import awre as awre_io
from prism_data import rapsodo
from prism_data.clean import apply_decisions, flag_basic, flag_shapes, load_decisions
from prism_data.match import match, merge
from prism_data.paths import DATA_DIR, PROCESSED_DIR
from prism_data.schema import ALL_COLUMNS

REVIEW_DIR = DATA_DIR / "review"
SHAPE_COLUMNS = ["shape_family", "shape_conf", "subtype", "bucket", "fb_velo", "fb_ivb", "fb_hb"]
TAG_NAMES = {"Fastball": "Fastball", "TwoSeamFastball": "Two-seam", "Cutter": "Cutter", "Slider": "Slider",
             "CurveBall": "Curveball", "ChangeUp": "Changeup", "Splitter": "Splitter"}


def classify_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Classify pitches with no basic issues; flagged pitches keep a blank pitch type."""
    from prism_model.shapes import classify

    df = df.copy()
    ok = (df["issues"].map(len) == 0) & df["RelSpeed"].notna() & df["InducedVertBreak"].notna()
    work = df[ok].assign(_row=df.index[ok])
    out = classify(work, by=["PitcherId"], merge_splits=True).set_index("_row")
    for col in SHAPE_COLUMNS:
        df[col] = np.nan if col not in ("shape_family", "subtype", "bucket") else None
        df.loc[out.index, col] = out[col]
    return df


def add_batter_side(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return df
    roster = pd.read_csv(path)
    names = roster["Name"] if "Name" in roster else roster["Batter"]
    side = dict(zip(names.map(rapsodo.clean_name), roster["Bats"].astype(str).str.upper().str[0]))
    known = df["Batter"].map(lambda b: side.get(rapsodo.clean_name(b)) if isinstance(b, str) else None)
    # Switch hitters bat opposite the pitcher.
    switch = known == "S"
    known = known.where(~switch, np.where(df["PitcherThrows"] == "R", "L", "R"))
    df["BatterSide"] = known.where(known.isin(["R", "L"]), df["BatterSide"])
    return df


def inactive(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {rapsodo.clean_name(x) for x in path.read_text().splitlines() if x.strip()}


def build(name: str, rapsodo_dir: Path, awre_dir: Path | None, decisions_path: Path | None,
          hitters_path: Path, inactive_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    gone = inactive(inactive_path) if inactive_path else set()
    rap = rapsodo.read_folder(rapsodo_dir)
    rap = rap[~rap["PitcherId"].isin(gone)].reset_index(drop=True)
    rap = flag_basic(rap)
    rap = classify_clean(rap)
    rap = flag_shapes(rap)
    rap = apply_decisions(rap, load_decisions(decisions_path) if decisions_path else {})

    pairs, awre = None, None
    if awre_dir and any(Path(awre_dir).glob("*PitchData-*.xls*")):
        awre = awre_io.read_folder(awre_dir)
        awre = awre[~awre["PitcherId"].map(rapsodo.clean_name).isin(gone)].reset_index(drop=True)
        pairs = match(awre, rap)
        rap = merge(awre, rap, pairs)
    rap = add_batter_side(rap, hitters_path)
    return rap, awre, pairs


# --------------------------------------------------------------------------
# Review workbook
# --------------------------------------------------------------------------
def _result_text(r) -> str:
    if not isinstance(r.get("PitchCall"), str):
        return ""
    txt = f"{r['Balls']}-{r['Strikes']} {r['PitchCall']}"
    if isinstance(r.get("PlayResult"), str) and r["PlayResult"] != "Undefined":
        txt += f" ({r['PlayResult']})"
    elif isinstance(r.get("KorBB"), str) and r["KorBB"] != "Undefined":
        txt += f" ({r['KorBB']})"
    return txt


def review_tables(df: pd.DataFrame, awre: pd.DataFrame | None, pairs: pd.DataFrame | None) -> dict:
    has_note = df["notes"].map(len) > 0 if "notes" in df else False
    flagged = df[(df["issues"].map(len) > 0) | has_note].sort_values(["PitcherId", "PitchTime"])
    sheet = pd.DataFrame({
        "Pitcher": flagged["PitcherId"],
        "Date": flagged["PitchTime"].dt.strftime("%Y-%m-%d"),
        "Time": flagged["PitchTime"].dt.strftime("%I:%M:%S %p"),
        "Pitch ID": flagged["PitchUID"],
        "Rapsodo tag": flagged["TaggedPitchType"].map(TAG_NAMES).fillna(""),
        "PRISM type": flagged["shape_family"].fillna(""),
        "Velo": flagged["RelSpeed"].round(1),
        "Spin": flagged["SpinRate"].round(0),
        "IVB": flagged["InducedVertBreak"].round(1),
        "HB (arm +)": flagged["hb_arm"].round(1),
        "Rel height": flagged["RelHeight"].round(2),
        "Rel side": flagged["RelSide"].round(2),
        "Result (AWRE)": [_result_text(r) for _, r in flagged.iterrows()],
        "Why it's flagged": [ "; ".join(list(a) + list(b)) for a, b in zip(flagged["issues"], flagged["notes"])],
        "PRISM's call": np.where(flagged["exclude"], "drop", "keep"),
        "Your call (keep/drop)": flagged["review"].replace("", np.nan),
        "Note": "",
    })

    summary = df.groupby("PitcherId").agg(
        hand=("PitcherThrows", "first"),
        rapsodo_pitches=("PitchUID", "size"),
        flagged=("issues", lambda s: int((s.map(len) > 0).sum())),
        notes=("notes", lambda s: int((s.map(len) > 0).sum())),
        dropped=("exclude", "sum"),
        graded_pitches=("exclude", lambda s: int((~s).sum())),
    )
    if pairs is not None:
        p = pairs.copy()
        p["pitcher"] = p["pitcher"].astype(str)
        conf = p.pivot_table(index="pitcher", columns="confidence", values="date", aggfunc="size", fill_value=0)
        conf = conf.reindex(columns=["strong", "time-only", "check", "awre-only", "rapsodo-only"], fill_value=0)
        awre_n = awre.groupby(awre["PitcherId"].map(rapsodo.clean_name)).size().rename("awre_pitches")
        summary = summary.join(awre_n, how="outer").join(conf, how="left").fillna(0)
        summary["matched_with_results"] = summary[["strong", "time-only", "check"]].sum(axis=1)
    summary = summary.reset_index().rename(columns={"index": "Pitcher", "PitcherId": "Pitcher"})

    tagged = df[df["TaggedPitchType"].notna() & df["shape_family"].notna()]
    tags = pd.crosstab([tagged["PitcherId"], tagged["TaggedPitchType"].map(TAG_NAMES)], tagged["shape_family"])
    tags.index.names = ["Pitcher", "Rapsodo tag"]
    tags = tags.reset_index()

    match_check = None
    if pairs is not None:
        chk = pairs[pairs["confidence"] == "check"].copy()
        chk["AWRE time"] = chk["awre_idx"].map(lambda i: awre.at[int(i), "PitchTime"].strftime("%I:%M:%S %p"))
        chk["Rapsodo time"] = chk["rap_idx"].map(lambda i: df.loc[df.index == int(i), "PitchTime"].iloc[0].strftime("%I:%M:%S %p")
                                                 if (df.index == int(i)).any() else "")
        match_check = chk[["pitcher", "date", "AWRE time", "Rapsodo time", "awre_velo", "rap_velo", "lag_s"]].rename(
            columns={"pitcher": "Pitcher", "date": "Date", "awre_velo": "AWRE velo", "rap_velo": "Rapsodo velo",
                     "lag_s": "AWRE minus Rapsodo (s)"})
    return {"Summary": summary, "Flagged pitches": sheet, "Tags vs PRISM": tags, "Match check": match_check}


def write_review(tables: dict, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    path.parent.mkdir(parents=True, exist_ok=True)
    notes = pd.DataFrame({"How to use this workbook": [
        "Flagged pitches: every pitch PRISM thinks is a misread or doesn't belong, with the reason.",
        "PRISM's call is what happens if you do nothing: likely misreads are dropped from grades; notes (e.g. a one-off pitch moved to his closest pitch type) are kept.",
        "To overrule it, type keep or drop in 'Your call'. Leave it blank to accept PRISM's call.",
        "Send the filled-in workbook back; your calls are saved and re-applied every time the data is rebuilt.",
        "Summary: pitches per pitcher, how many were flagged, and how many AWRE results were matched.",
        "Tags vs PRISM: Rapsodo's pitch tag against PRISM's shape-based type, for a sanity check. Tags are never used for grading.",
        "Match check: AWRE-Rapsodo pairs whose velocity or timing didn't agree well. Their results are still used.",
    ]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        notes.to_excel(xw, "Read me", index=False)
        for sheet, t in tables.items():
            if t is not None:
                t.to_excel(xw, sheet, index=False)
        head = PatternFill("solid", fgColor="1F2A37")
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = head
                c.alignment = Alignment(vertical="center", wrap_text=True)
            for i, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200)), 1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70)
        ws = xw.book["Flagged pitches"]
        hdr = [c.value for c in ws[1]]
        if "Your call (keep/drop)" in hdr and ws.max_row > 1:
            col = get_column_letter(hdr.index("Your call (keep/drop)") + 1)
            dv = DataValidation(type="list", formula1='"keep,drop"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{col}2:{col}{ws.max_row}")
            fill = PatternFill("solid", fgColor="FFF4CC")
            for r in range(2, ws.max_row + 1):
                ws[f"{col}{r}"].fill = fill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="team")
    parser.add_argument("--rapsodo", default=str(DATA_DIR / "rapsodo"))
    parser.add_argument("--awre", default=str(DATA_DIR / "awre"))
    parser.add_argument("--decisions", default=str(DATA_DIR / "rapsodo" / "review_decisions.csv"))
    parser.add_argument("--hitters", default=str(DATA_DIR / "rosters" / "roster.csv"))
    parser.add_argument("--inactive", default=str(DATA_DIR / "rosters" / "inactive_pitchers.txt"))
    args = parser.parse_args()

    df, awre, pairs = build(args.name, Path(args.rapsodo), Path(args.awre), Path(args.decisions), Path(args.hitters),
                             Path(args.inactive))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["issues"] = out["issues"].map(lambda f: "; ".join(f))
    out["notes"] = out["notes"].map(lambda f: "; ".join(f))
    out.to_parquet(PROCESSED_DIR / f"{args.name}.parquet", index=False)
    review = REVIEW_DIR / f"{args.name}_review.xlsx"
    tables = review_tables(df, awre, pairs)
    write_review(tables, review)

    flagged = int((df["issues"].map(len) > 0).sum())
    noted = int((df["notes"].map(len) > 0).sum())
    print(f"{df['PitcherId'].nunique()} pitchers, {len(df):,} Rapsodo pitches; {flagged} likely misreads, "
          f"{noted} with notes; {int(df['exclude'].sum())} left out of grades")
    if pairs is not None:
        print(pairs["confidence"].value_counts().to_string())
    print(f"\nSaved {PROCESSED_DIR / (args.name + '.parquet')}\nReview workbook: {review}")


if __name__ == "__main__":
    main()
