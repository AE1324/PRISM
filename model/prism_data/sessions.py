"""
Session tags: what kind of session each pitch came from.

Every session is tagged with a device (trackman | rapsodo) and a setting:
  game     a real game or scrimmage
  live     live at-bats against hitters (practice)
  bullpen  no hitters

Only game and live sessions are graded; bullpens are stored but left out of every grade. The tag
alone decides: PRISM never guesses a session's setting from velocity or anything else in the data.

Tags come from a sessions file (data/sessions/sessions.csv), one row per session:
  Date, Device, Setting[, Pitcher]
Leave Pitcher blank to tag everyone that day on that device. A row naming a pitcher overrides the
blank row for that pitcher. Untagged sessions fall back to the device default: TrackMan game exports
are games; anything else is a bullpen until it's tagged.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from prism_data.devices import profile

DEVICES = ("trackman", "rapsodo")
SETTINGS = ("game", "live", "bullpen")
GRADED = ("game", "live")


def _person(name) -> str:
    """'Black, Grayson' / 'Grayson Black' / 'Grayson_Black' -> 'grayson black'."""
    s = str(name or "").strip()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first} {last}"
    return re.sub(r"[^a-z ]", "", re.sub(r"[_\s]+", " ", s.lower())).strip()


def load_sessions(path: str | Path | None) -> pd.DataFrame:
    """Read the sessions file; returns Date, Device, Setting, Pitcher (blank = everyone)."""
    cols = ["Date", "Device", "Setting", "Pitcher"]
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    s = pd.read_csv(path, dtype=str).rename(columns=str.strip)
    missing = {"Date", "Device", "Setting"} - set(s.columns)
    if missing:
        raise ValueError(f"sessions file needs columns Date, Device, Setting (missing {', '.join(sorted(missing))})")
    s["Pitcher"] = s["Pitcher"] if "Pitcher" in s else ""
    s["Date"] = pd.to_datetime(s["Date"].str.strip(), errors="coerce", format="mixed").dt.date
    s["Device"] = s["Device"].str.strip().str.lower()
    s["Setting"] = s["Setting"].str.strip().str.lower()
    s["Pitcher"] = s["Pitcher"].fillna("").map(_person)
    bad = s[s["Date"].isna() | ~s["Device"].isin(DEVICES) | ~s["Setting"].isin(SETTINGS)]
    if len(bad):
        raise ValueError(f"sessions file has rows PRISM can't read (row {bad.index[0] + 2}): device must be one of "
                         f"{', '.join(DEVICES)}; setting one of {', '.join(SETTINGS)}")
    return s[cols]


def tag_sessions(df: pd.DataFrame, sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``Device`` and ``Setting`` to every pitch (see module notes for the rules)."""
    df = df.copy()
    df["Device"] = df["Source"].astype(str).str.lower() if "Source" in df else "rapsodo"
    df["Setting"] = df["Device"].map(lambda k: profile(k).default_setting)  # each device's default (devices.py)
    if sessions is None or sessions.empty:
        return df
    day = pd.to_datetime(df["Date"].astype(str), errors="coerce").dt.date
    who = (df["Pitcher"] if "Pitcher" in df else df["PitcherId"]).map(_person)
    # blank-pitcher rows first, so a row naming the pitcher wins
    for _, r in sessions.sort_values("Pitcher", key=lambda x: x != "").iterrows():
        hit = (day == r["Date"]) & (df["Device"] == r["Device"])
        if r["Pitcher"]:
            hit &= who == r["Pitcher"]
        df.loc[hit, "Setting"] = r["Setting"]
    return df


def graded(df: pd.DataFrame) -> pd.DataFrame:
    """Pitches from sessions that count toward grades (game and live). Untagged data passes through."""
    return df[df["Setting"].isin(GRADED)] if "Setting" in df else df
