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
  spin misread     almost no spin with real movement (the ball can't move like that
                   without spin), or a spin value outside what's possible
Checks after classification (pitch type assigned by shape)
  shape outlier    nothing like his other pitches of the same type
  one-off shape    a pitch type he threw only once or twice: moved to his closest regular
                   pitch type when it looks like one (a note), otherwise a misread
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
    _add(flags, bad_spin, lambda i: f"impossible spin reading ({spin[i]:.0f} rpm with {total0[i]:.0f} in of movement)")
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


ONE_OFF_MAX = 2          # a pitch type thrown this few times (by someone with 30+ pitches) is a one-off
REASSIGN_Z = 3.0         # a one-off joins his nearest regular pitch type if it's within this many spreads of it


def _type_stats(g: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    center = g[cols].median()
    scale = pd.Series({c: max(_mad(g[c]), SHAPE_FLOOR[c]) for c in cols})
    return center, scale


def flag_shapes(df: pd.DataFrame, by: str = "PitcherId") -> pd.DataFrame:
    """
    Checks that need ``shape_family`` (PRISM's pitch type), in this order:
      1. One-off pitch types: moved to the pitcher's closest regular pitch type
         when they look like it, otherwise flagged as a misread.
      2. Pitches nothing like his others of the same type.
      3. Rapsodo's two movement reads disagreeing far more than usual for that pitch type.
    """
    df = df.copy()
    empty = lambda: pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)  # noqa: E731
    flags = df["issues"].copy() if "issues" in df else empty()
    notes = df["notes"].copy() if "notes" in df else empty()
    ok = df["shape_family"].notna() & df["RelSpeed"].notna()
    cols = list(SHAPE_FLOOR)
    label = {"RelSpeed": "velocity", "InducedVertBreak": "vertical break", "hb_arm": "horizontal break"}

    # 1. One-off pitch types.
    for pid, g in df[ok].groupby(by):
        counts = g["shape_family"].value_counts()
        if len(g) < 30:
            continue
        regular = [k for k, n in counts.items() if n >= 5]
        stats = {k: _type_stats(g[g["shape_family"] == k], cols) for k in regular}
        for kind, n in counts.items():
            if n > ONE_OFF_MAX:
                continue
            for i in g.index[g["shape_family"] == kind]:
                best, best_z = None, np.inf
                for k, (center, scale) in stats.items():
                    z = float(((df.loc[i, cols].astype(float) - center) / scale).abs().max())
                    if z < best_z:
                        best, best_z = k, z
                if best is not None and best_z <= REASSIGN_Z:
                    donor = g[g["shape_family"] == best].iloc[0]
                    for col in ("shape_family", "subtype", "bucket"):
                        if col in df:
                            df.at[i, col] = donor[col]
                    notes.at[i] = notes.at[i] + [f"one-off {kind.lower()} shape; moved to his {best.lower()} (closest match)"]
                else:
                    flags.at[i] = flags.at[i] + [f"one-off {kind.lower()} shape that matches none of his pitches"]

    # 2. Unlike his other pitches of the same type.
    for (pid, kind), g in df[ok].groupby([by, "shape_family"]):
        if len(g) < 6:
            continue
        center, scale = _type_stats(g, cols)
        z = ((g[cols] - center) / scale).abs()
        worst = z.max(axis=1)
        for i in worst[worst > SHAPE_Z].index:
            col = z.loc[i].idxmax()
            unit = "mph" if col == "RelSpeed" else "in"
            flags.at[i] = flags.at[i] + [f"unlike his other {kind.lower()}s ({label[col]} {df.at[i, col]:.1f} vs usual "
                                         f"{center[col]:.1f} {unit})"]

    # 3. Rapsodo's two movement reads disagree far more than usual for this pitcher and pitch type
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
