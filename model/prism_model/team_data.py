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
  8. Tag each session's device and setting (game | live | bullpen) from the sessions file.
  9. Save the data and a review workbook.

Files (all inside the gitignored data/ folder)
  data/rapsodo/*.csv                         Rapsodo player exports
  data/awre/*PitchData-*.xls                 AWRE pitch logs
  data/rapsodo/review_decisions.csv          saved keep/drop calls (or a filled-in review workbook)
  data/rosters/roster.csv                    optional: Name, Bats (R/L/S), Throws, Position, Height
  data/rosters/inactive_pitchers.txt         optional: one name per line, left out entirely
  data/sessions/sessions.csv                 Date, Device, Setting[, Pitcher]: untagged Rapsodo = bullpen
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
from prism_data.clean import (apply_decisions, apply_game_movement, attribute_wrong_pitcher, check_game_movement,
                              flag_basic, flag_shapes, load_decisions)
from prism_data.devices import apply_offsets
from prism_data.sessions import load_sessions, tag_sessions
from prism_data.trackman import read_trackman
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


TYPE_SCALE = {"RelSpeed": 2.0, "SpinRate": 150.0}   # mph, rpm: one "step" when matching by speed and spin
TYPE_MAX_STEPS = 3.0


def type_without_movement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give a pitch type to clean pitches that have speed and spin but no reliable movement, by matching
    them to the pitcher's own pitch types from pitches that do have movement (nearest by velocity and
    spin, within 3 steps of 2 mph / 150 rpm). Their movement stays blank, so they count toward velocity,
    location and results but not the movement averages. Pitchers with no classified pitches are left out.
    """
    df = df.copy()
    need = df["shape_family"].isna() & df["RelSpeed"].notna() & df["SpinRate"].notna() & (df["issues"].map(len) == 0)
    if not need.any():
        return df
    known = df[df["shape_family"].notna()]
    for pid, g in df[need].groupby("PitcherId"):
        k = known[known["PitcherId"] == pid]
        types = k.groupby("shape_family").filter(lambda x: len(x) >= 5).groupby("shape_family")
        if not len(types):
            continue
        centers = types[list(TYPE_SCALE)].median()
        info = types.agg({"subtype": lambda x: x.mode().iloc[0] if x.notna().any() else None,
                          "bucket": "first"})
        ref = k.iloc[0]
        for idx, r in g.iterrows():
            steps = ((centers - r[list(TYPE_SCALE)]).abs() / pd.Series(TYPE_SCALE)).max(axis=1)
            if steps.min() > TYPE_MAX_STEPS:
                continue
            kind = steps.idxmin()
            df.at[idx, "shape_family"] = kind
            df.at[idx, "subtype"] = info.at[kind, "subtype"]
            df.at[idx, "bucket"] = info.at[kind, "bucket"]
            for c in ("fb_velo", "fb_ivb", "fb_hb"):
                if c in df:
                    df.at[idx, c] = ref.get(c)
            df.at[idx, "notes"] = list(df.at[idx, "notes"]) + ["pitch type from speed and spin (no reliable movement)"]
    return df


def add_level_exit_velo(df: pd.DataFrame) -> pd.DataFrame:
    """
    ``ExitSpeed_mlb``: each exit velocity mapped onto MLB's by its percentile among every measured ball in
    play in these files (both teams), so "hard contact" means hard for this level. Needs 50+ batted balls.
    """
    from prism_model import results_tiers

    tiers = results_tiers.load()
    if tiers is None or "ExitSpeed" not in df:
        return df
    ev = pd.to_numeric(df["ExitSpeed"], errors="coerce")
    bip = (df["PitchCall"].astype(object) == "InPlay") & ev.notna() & pd.to_numeric(df["Angle"], errors="coerce").notna()
    return df.assign(ExitSpeed_mlb=results_tiers.level_ev_to_mlb(ev, ev[bip], tiers))


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
          hitters_path: Path, inactive_path: Path | None = None,
          sessions_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    gone = inactive(inactive_path) if inactive_path else set()
    rap = apply_offsets(rapsodo.read_folder(rapsodo_dir))   # device offsets marked applied (devices.py)
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
    rap = tag_sessions(rap, load_sessions(sessions_path))
    return rap, awre, pairs


def build_trackman(folder: Path, decisions_path: Path | None = None, team: str | None = None,
                   inactive_path: Path | None = None, sessions_path: Path | None = None) -> pd.DataFrame:
    """
    TrackMan game exports through the same cleaning as Rapsodo: read every .csv/.xlsx export, flag misreads
    per pitcher per game (release point checks use each game as its own stint), name the likely pitcher for
    wrong-side pitches, check each game's movement (blanked for games that can't be trusted), classify the
    clean pitches, flag shape outliers, give pitches without movement a type from speed and spin, apply saved
    review decisions, set the level's exit velocity reference from both teams, then keep one
    team's pitchers (``team`` = TrackMan PitcherTeam code). Results come from the
    export itself (count, call, exit velocity, launch angle), so there's no matching step.
    """
    gone = inactive(inactive_path) if inactive_path else set()
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".csv", ".xlsx") and not p.name.startswith("~"))
    frames = []
    for f in files:
        try:
            frames.append(read_trackman(f).assign(source_file=f.name))
        except (KeyError, ValueError):
            continue  # not a pitch-by-pitch export (e.g. a summary workbook)
    df = apply_offsets(pd.concat(frames, ignore_index=True))
    df = df[~df["Pitcher"].map(rapsodo.clean_name).isin(gone)].reset_index(drop=True)
    df["stint"] = df["PitcherId"].astype(str) + "|" + df["GameID"].astype(str)
    if "PitchTime" not in df or df["PitchTime"].isna().all():
        df["PitchTime"] = pd.to_datetime(df["Date"].astype(str), errors="coerce")
    decisions = load_decisions(decisions_path) if decisions_path else {}
    df = flag_basic(df)
    df = attribute_wrong_pitcher(df)
    games = check_game_movement(df)
    df = apply_game_movement(df, games, decisions)
    df = add_level_exit_velo(df)  # both teams' batted balls set the level
    df = classify_clean(df)
    df = flag_shapes(df)
    df = type_without_movement(df)
    df = apply_decisions(df, decisions)
    if team:  # last: every level reference above reads both teams' pitchers
        df = df[df["PitcherTeam"].astype(str) == team].reset_index(drop=True)
    df = tag_sessions(df, load_sessions(sessions_path))  # TrackMan exports default to game
    df.attrs["games"] = games
    return df


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
    # a whole game's blanked movement is shown once on the Games sheet, not on every pitch
    game_note = lambda n: [x for x in n if not str(x).startswith("movement not used")]
    has_note = df["notes"].map(lambda n: len(game_note(n)) > 0) if "notes" in df else False
    flagged = df[(df["issues"].map(len) > 0) | has_note].sort_values(["PitcherId", "PitchTime"])
    sheet = pd.DataFrame({
        "Pitcher": flagged["Pitcher"] if "Pitcher" in flagged else flagged["PitcherId"],
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
        "Why it's flagged": ["; ".join(list(a) + game_note(b)) for a, b in zip(flagged["issues"], flagged["notes"])],
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
        game_or_live=("Setting", lambda s: int(s.isin(["game", "live"]).sum())),
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
    games = None
    g = df.attrs.get("games")
    if g is not None and len(g):
        calls = df.drop_duplicates("GameID").set_index("GameID")["movement_trusted"] if "movement_trusted" in df else {}
        games = pd.DataFrame({
            "Game": g["GameID"],
            "Pitches": g["pitches"],
            "No reliable movement": (g["missing_share"] * 100).round(0).astype(int).astype(str) + "%",
            "Field ride offset (in)": g["field_ride_offset_in"],
            "Field run offset (in)": g["field_run_offset_in"],
            "Same-pitcher ride offset (in)": g["ride_offset_in"],
            "Why": g["reason"].replace("", "looks normal"),
            "PRISM's call": np.where(g["trusted"], "trust", "distrust"),
            "Your call (trust/distrust)": [
                ("trust" if calls.get(x) else "distrust") if bool(calls.get(x, t)) != bool(t) else np.nan
                for x, t in zip(g["GameID"], g["trusted"])],
        })
    return {"Summary": summary, "Games": games, "Flagged pitches": sheet, "Tags vs PRISM": tags,
            "Match check": match_check}


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
        "Games (TrackMan): whether each game's movement readings can be trusted. A distrusted game's pitches still count for velocity, location and results, just not movement, Stuff+ or Shape+. Type trust or distrust in 'Your call' to overrule.",
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
        if "Games" in xw.book.sheetnames and xw.book["Games"].max_row > 1:
            gs = xw.book["Games"]
            ghdr = [c.value for c in gs[1]]
            gcol = get_column_letter(ghdr.index("Your call (trust/distrust)") + 1)
            gdv = DataValidation(type="list", formula1='"trust,distrust"', allow_blank=True)
            gs.add_data_validation(gdv)
            gdv.add(f"{gcol}2:{gcol}{gs.max_row}")
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
    parser.add_argument("--sessions", default=str(DATA_DIR / "sessions" / "sessions.csv"))
    args = parser.parse_args()

    df, awre, pairs = build(args.name, Path(args.rapsodo), Path(args.awre), Path(args.decisions), Path(args.hitters),
                             Path(args.inactive), Path(args.sessions))
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
    print("Sessions by setting (pitches): " + ", ".join(f"{k} {v}" for k, v in df["Setting"].value_counts().items()))
    if (df["Setting"] == "bullpen").all():
        print(f"  No graded sessions: tag game/live dates in {args.sessions}")
    if pairs is not None:
        print(pairs["confidence"].value_counts().to_string())
    print(f"\nSaved {PROCESSED_DIR / (args.name + '.parquet')}\nReview workbook: {review}")


if __name__ == "__main__":
    main()
