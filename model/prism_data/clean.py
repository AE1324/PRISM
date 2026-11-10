"""
Flag pitch-tracking misreads for review.

Nothing is deleted. Each pitch gets two lists:
  issues  likely misreads; the pitch is left out of grades unless a person says keep
  notes   worth a look, or a single reading was fixed; the pitch stays in unless a person says drop
The review sheet lists every pitch with either, so a person can overrule any
call; those decisions are saved and re-applied on every rebuild.

Checks before classification
  duplicate        the same pitch id twice
  no velocity      can't be graded
  no movement      neither trajectory nor spin movement was read
  impossible       speed, spin, movement or release outside what a pitcher can do
  wrong hand       released from the opposite side to every other pitch he throws
  release point    far from his release point in that stint
  spin misread     (note) almost no spin but real movement: the spin reading is ignored,
                   the pitch is kept (the ball can't move like that without spin)
Checks after classification (pitch type assigned by shape)
  shape outlier    nothing like his other pitches of the same type
  one-off shape    (note) a pitch type he threw only once or twice
  movement reads   Rapsodo's flight-based and spin-based movement disagree far more than
                   usual for his pitches of that type
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VELO_RANGE = (45, 100)
SPIN_RANGE = (150, 3600)
MOVE_MAX_IN = 30
REL_HEIGHT_RANGE = (2.0, 8.0)
EXTENSION_RANGE = (3.5, 8.5)
RELEASE_OUTLIER_FT = 1.0
MOVEMENT_READ_GAP_IN = 10.0
SHAPE_Z = 4.0
SHAPE_FLOOR = {"RelSpeed": 1.5, "InducedVertBreak": 3.0, "hb_arm": 3.0}


def _add(flags: pd.Series, mask: pd.Series, text) -> None:
    for idx in mask[mask.fillna(False)].index:
        flags.at[idx] = flags.at[idx] + [text(idx) if callable(text) else text]


def _mad(s: pd.Series) -> float:
    return 1.4826 * float(np.nanmedian(np.abs(s - np.nanmedian(s))))


def flag_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Checks that don't need pitch types. Adds ``flags`` (list of reasons)."""
    df = df.copy()
    flags = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    notes = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    v, spin = df["RelSpeed"], df["SpinRate"]
    ivb0, hb0 = df["InducedVertBreak"], df["HorzBreak"]
    total0 = np.hypot(ivb0, hb0)
    # Spin misreads: almost no spin with real movement, or an impossible spin value. Keep the pitch, drop the reading.
    bad_spin = (((spin < 700) & (total0 > 8) & (v > 60)) | ~spin.between(*SPIN_RANGE)) & spin.notna()
    _add(notes, bad_spin, lambda i: f"spin misread ({spin[i]:.0f} rpm with {total0[i]:.0f} in of movement); spin ignored, pitch kept")
    df.loc[bad_spin, "SpinRate"] = np.nan
    spin = df["SpinRate"]
    ivb, hb = df["InducedVertBreak"], df["HorzBreak"]

    if "PitchUID" in df:
        _add(flags, df["PitchUID"].duplicated(keep="first") & df["PitchUID"].notna(), "duplicate pitch")
    _add(flags, v.isna(), "no velocity read")
    if "movement_source" in df:
        _add(flags, df["movement_source"] == "none", "no movement read")
    _add(flags, ~v.between(*VELO_RANGE) & v.notna(), lambda i: f"impossible velocity ({v[i]:.1f} mph)")
    _add(flags, (ivb.abs() > MOVE_MAX_IN) | (hb.abs() > MOVE_MAX_IN),
         lambda i: f"impossible movement ({ivb[i]:.1f} IVB, {hb[i]:.1f} HB)")
    _add(flags, ~df["RelHeight"].between(*REL_HEIGHT_RANGE) & df["RelHeight"].notna(),
         lambda i: f"impossible release height ({df.at[i, 'RelHeight']:.1f} ft)")
    _add(flags, ~df["Extension"].between(*EXTENSION_RANGE) & df["Extension"].notna(),
         lambda i: f"impossible extension ({df.at[i, 'Extension']:.1f} ft)")

    # Wrong side of the rubber for this pitcher.
    side = df["rel_side_arm"]
    _add(flags, (side < -0.5),
         lambda i: f"released from the wrong side for a {df.at[i, 'PitcherThrows']}HP ({df.at[i, 'RelSide']:+.1f} ft)")

    # Far from his release point within the same stint.
    by = "stint" if "stint" in df else "PitcherId"
    dh = df["RelHeight"] - df.groupby(by)["RelHeight"].transform("median")
    ds = df["RelSide"] - df.groupby(by)["RelSide"].transform("median")
    dist = np.hypot(dh, ds)
    _add(flags, (dist > RELEASE_OUTLIER_FT) & (side >= -0.5),
         lambda i: f"release point {dist[i]:.1f} ft from his usual spot")

    df["issues"] = flags
    df["notes"] = notes
    return df


def flag_shapes(df: pd.DataFrame, by: str = "PitcherId") -> pd.DataFrame:
    """Checks that need ``shape_family`` (PRISM's pitch type)."""
    df = df.copy()
    flags = df["issues"].copy() if "issues" in df else pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    notes = df["notes"].copy() if "notes" in df else pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    ok = df["shape_family"].notna() & df["RelSpeed"].notna()
    cols = list(SHAPE_FLOOR)
    for (pid, kind), g in df[ok].groupby([by, "shape_family"]):
        if len(g) < 6:
            continue
        center = g[cols].median()
        scale = pd.Series({c: max(_mad(g[c]), SHAPE_FLOOR[c]) for c in cols})
        z = ((g[cols] - center) / scale).abs()
        worst = z.max(axis=1)
        for i in worst[worst > SHAPE_Z].index:
            col = z.loc[i].idxmax()
            label = {"RelSpeed": "velocity", "InducedVertBreak": "vertical break", "hb_arm": "horizontal break"}[col]
            unit = "mph" if col == "RelSpeed" else "in"
            flags.at[i] = flags.at[i] + [f"unlike his other {kind.lower()}s ({label} {df.at[i, col]:.1f} vs usual "
                                         f"{center[col]:.1f} {unit})"]
    # One-off pitch types: thrown once or twice by someone with a real sample.
    counts = df[ok].groupby([by, "shape_family"]).size()
    totals = df[ok].groupby(by).size()
    for (pid, kind), n in counts.items():
        if n <= 2 and totals[pid] >= 30:
            for i in df[ok & (df[by] == pid) & (df["shape_family"] == kind)].index:
                notes.at[i] = notes.at[i] + [f"one-off {kind.lower()} shape (only {n} of {totals[pid]} pitches)"]
    # Rapsodo's two movement reads disagree far more than usual for this pitcher and pitch type
    # (the normal gap differs a lot by pitch type, e.g. gyro sliders).
    if {"vb_spin", "hb_spin", "movement_source"} <= set(df.columns):
        traj = df["movement_source"] == "trajectory"
        gv = (df["InducedVertBreak"] - df["vb_spin"]).where(traj & ok)
        gh = (df["HorzBreak"] - df["hb_spin"]).where(traj & ok)
        keys = [df[by], df["shape_family"]]
        off_v = (gv - gv.groupby(keys).transform("median")).abs()
        off_h = (gh - gh.groupby(keys).transform("median")).abs()
        for i in df.index[((off_v > MOVEMENT_READ_GAP_IN) | (off_h > MOVEMENT_READ_GAP_IN)).fillna(False)]:
            flags.at[i] = flags.at[i] + [f"flight and spin movement disagree ({df.at[i, 'InducedVertBreak']:.0f}/"
                                         f"{df.at[i, 'HorzBreak']:.0f} vs {df.at[i, 'vb_spin']:.0f}/{df.at[i, 'hb_spin']:.0f} in)"]
    df["issues"] = flags
    df["notes"] = notes
    return df


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
def load_decisions(path: str | Path) -> dict:
    """PitchUID -> 'keep' | 'drop' from a saved decisions CSV or a filled-in review workbook."""
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix.lower() in (".xlsx", ".xls"):
        d = pd.read_excel(path, sheet_name="Flagged pitches")
        d = d.rename(columns={"Your call (keep/drop)": "decision", "Pitch ID": "PitchUID"})
    else:
        d = pd.read_csv(path)
    d = d.dropna(subset=["decision"])
    d["decision"] = d["decision"].astype(str).str.strip().str.lower()
    d = d[d["decision"].isin(["keep", "drop"])]
    return dict(zip(d["PitchUID"].astype(str), d["decision"]))


def apply_decisions(df: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """``exclude`` = has issues (unless a person said keep) or a person said drop. Notes alone never exclude."""
    df = df.copy()
    call = df["PitchUID"].astype(str).map(decisions)
    flagged = df["issues"].map(len) > 0
    df["review"] = call.fillna("")
    df["exclude"] = (flagged & (call != "keep")) | (call == "drop")
    return df
