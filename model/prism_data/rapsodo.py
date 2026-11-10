"""
Read Rapsodo pitching exports (one CSV per player) into PRISM's format.

Each export starts with a few metadata lines ("Player ID:", "Player Name:")
and then one row per pitch. Rapsodo writes "-" for anything it didn't measure.

What needs care
---------------
Movement   Rapsodo reports movement two ways: from the ball's measured flight
           ("trajectory", the same idea as TrackMan's induced vertical and
           horizontal break) and estimated from spin ("spin"). Trajectory is
           what PRISM uses. When a pitch has only the spin estimate, it is
           converted with that pitcher's own typical gap between the two for
           the same kind of pitch, and marked ``movement_source = "spin-adjusted"``.
Release    The unit sits in a different spot from one session to the next, so
           release side and height can jump for every pitch in a stint. Each
           stint (a run of pitches with no gap over 15 minutes) is re-centered
           on the pitcher's usual release point; pitch-to-pitch variation is
           kept. The size of each correction is kept in ``release_shift_ft``.
Signs      Same as TrackMan: release side and horizontal break are positive
           toward a right-hander's arm side. Handedness isn't in the export, so
           it comes from the release side.
Location   Strike-zone side and height are in inches; PRISM stores feet.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prism_data.schema import add_normalized, conform

STINT_GAP_MIN = 15
EXTRA_COLUMNS = ["vb_spin", "hb_spin", "spin_eff", "spin_conf", "movement_source", "release_shift_ft", "stint",
                 "device_strike"]


def clean_name(name: str) -> str:
    """One spelling per person across sources (curly apostrophes, extra spaces)."""
    return re.sub(r"\s+", " ", str(name).replace("’", "'").replace("‘", "'")).strip()


def clock_to_axis(clock) -> float:
    """Rapsodo spin direction "1:30" (clock tilt) -> TrackMan spin axis in degrees (12:00 = 180°)."""
    if not isinstance(clock, str) or ":" not in clock:
        return np.nan
    h, m = clock.split(":")
    deg = (int(h) % 12) * 30 + int(m) * 0.5
    return (deg + 180) % 360


def read_rapsodo(path: str | Path) -> pd.DataFrame:
    """One Rapsodo player export -> PRISM columns plus the extras cleaning needs (not yet conformed)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    meta = {}
    for ln in lines[:10]:
        parts = [p.strip().strip('"') for p in ln.split(",", 1)]
        if len(parts) == 2 and parts[0].endswith(":"):
            meta[parts[0].rstrip(":")] = parts[1]
    start = next(i for i, ln in enumerate(lines) if ln.startswith('"No"'))
    raw = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["-"])

    df = pd.DataFrame(index=raw.index)
    name = clean_name(meta.get("Player Name", Path(path).stem))
    when = pd.to_datetime(raw["Date"].str.slice(4), format="%b %d %Y %I:%M:%S %p", errors="coerce")
    df["Source"] = "rapsodo"
    df["PitchTime"] = when
    df["Date"] = when.dt.date
    df["Season"] = when.dt.year.astype("Int64")
    df["GameID"] = "rapsodo-" + when.dt.strftime("%Y%m%d").fillna("unknown")
    df["PitchUID"] = raw["Unique ID"].astype(str)
    df["Pitcher"] = name
    df["PitcherId"] = name
    df["TaggedPitchType"] = raw["Pitch Type"]
    df["RelSpeed"] = raw["Velocity"]
    df["SpinRate"] = raw["Total Spin"]
    df["SpinAxis"] = raw["Spin Direction"].map(clock_to_axis)
    df["RelHeight"] = raw["Release Height"]
    df["RelSide"] = raw["Release Side"]
    df["Extension"] = raw["Release Extension (ft)"]
    df["InducedVertBreak"] = raw["VB (trajectory)"]
    df["HorzBreak"] = raw["HB (trajectory)"]
    df["PlateLocHeight"] = raw["Strike Zone Height"] / 12
    df["PlateLocSide"] = raw["Strike Zone Side"] / 12
    df["VertApprAngle"] = raw["Vertical Approach Angle"]
    df["HorzApprAngle"] = raw["Horizontal Approach Angle"]
    df["vb_spin"] = raw["VB (spin)"]
    df["hb_spin"] = raw["HB (spin)"]
    df["spin_eff"] = raw["Spin Efficiency (release)"]
    df["spin_conf"] = raw["Spin Confidence"]
    df["device_strike"] = raw["Is Strike"]
    # Rapsodo's zero placeholders for missing release data.
    for col in ["RelHeight", "RelSide"]:
        df.loc[(df["RelHeight"] == 0) & (df["RelSide"] == 0), col] = np.nan
    return df.sort_values("PitchTime").reset_index(drop=True)


def add_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Number each pitcher's stints: runs of pitches without a gap longer than STINT_GAP_MIN."""
    df = df.sort_values(["PitcherId", "PitchTime"]).copy()
    gap = df.groupby("PitcherId")["PitchTime"].diff() > pd.Timedelta(minutes=STINT_GAP_MIN)
    new = gap | df["PitcherId"].ne(df["PitcherId"].shift())
    df["stint"] = new.cumsum()
    return df


def fill_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Use trajectory movement; convert spin-only movement with the pitcher's own gap between the two."""
    df = df.copy()
    has_traj = df["InducedVertBreak"].notna() & df["HorzBreak"].notna()
    has_spin = df["vb_spin"].notna() & df["hb_spin"].notna()
    df["movement_source"] = np.where(has_traj, "trajectory", np.where(has_spin, "spin-adjusted", "none"))
    both = df[has_traj & has_spin].assign(dv=lambda d: d["InducedVertBreak"] - d["vb_spin"],
                                          dh=lambda d: d["HorzBreak"] - d["hb_spin"])
    # Group spin-estimated pitches by rough shape (speed band within the pitcher), so a fastball
    # gets the fastball gap and a breaking ball the breaking-ball gap.
    def band(d):
        top = d.groupby("PitcherId")["RelSpeed"].transform(lambda s: s.quantile(0.9))
        return np.where(d["RelSpeed"] >= top - 4, "hard", np.where(d["vb_spin"] < 3, "drop", "other"))

    both["band"] = band(both)
    fix = df[~has_traj & has_spin].copy()
    if not fix.empty:
        fix["band"] = band(fix)
        by_band = both.groupby(["PitcherId", "band"])[["dv", "dh"]].median()
        by_pitcher = both.groupby("PitcherId")[["dv", "dh"]].median()
        keys = pd.MultiIndex.from_frame(fix[["PitcherId", "band"]])
        dv = pd.Series(by_band["dv"].reindex(keys).to_numpy(), index=fix.index).fillna(fix["PitcherId"].map(by_pitcher["dv"])).fillna(0)
        dh = pd.Series(by_band["dh"].reindex(keys).to_numpy(), index=fix.index).fillna(fix["PitcherId"].map(by_pitcher["dh"])).fillna(0)
        df.loc[fix.index, "InducedVertBreak"] = fix["vb_spin"] + dv
        df.loc[fix.index, "HorzBreak"] = fix["hb_spin"] + dh
    return df


def correct_release(df: pd.DataFrame) -> pd.DataFrame:
    """Re-center each stint's release point on the pitcher's usual one (undoes moving the unit)."""
    df = df.copy()
    df["release_shift_ft"] = 0.0
    for col in ["RelSide", "RelHeight"]:
        stint_med = df.groupby("stint")[col].transform("median")
        usual = df.groupby("stint")[col].median().groupby(df.groupby("stint")["PitcherId"].first()).median()
        target = df["PitcherId"].map(usual)
        shift = (stint_med - target).fillna(0)
        df[col] = df[col] - shift
        df["release_shift_ft"] = np.hypot(df["release_shift_ft"], shift)
    return df


def infer_hand(df: pd.DataFrame) -> pd.Series:
    side = df.groupby("PitcherId")["RelSide"].transform("median")
    return pd.Series(np.where(side > 0, "R", np.where(side < 0, "L", None)), index=df.index)


def read_folder(folder: str | Path) -> pd.DataFrame:
    """Every Rapsodo export in a folder, prepared (movement filled, release corrected, hand added)."""
    files = [f for f in sorted(Path(folder).glob("*.csv")) if "Player ID" in f.read_text(encoding="utf-8-sig")[:200]]
    df = pd.concat([read_rapsodo(f) for f in files], ignore_index=True)
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = add_stints(df)
    df = fill_movement(df)
    df = correct_release(df)
    df["PitcherThrows"] = infer_hand(df)
    df = add_normalized(df)
    return df.reset_index(drop=True)


def to_prism(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the cleaning extras and return exactly PRISM's columns."""
    return conform(df.copy())
