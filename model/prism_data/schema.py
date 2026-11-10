"""
PRISM's internal pitch format.

Every data source (MLB Statcast, TrackMan via Synergy or Oberlin's own unit,
and Rapsodo as an interim bullpen source) is converted into these columns
before anything else touches it. The model and the app only ever see this
format, so adding or retiring a source never changes downstream code.

Column names follow TrackMan's standard CSV export so a TrackMan file needs
little more than a column check. Units also follow TrackMan:
  speeds in mph, positions in feet, movement in inches, angles in degrees.

Sign conventions
----------------
Raw horizontal values (``HorzBreak``, ``RelSide``, ``PlateLocSide``) follow
TrackMan's convention. Because devices disagree on sign conventions, the
model never uses the raw horizontal columns directly. It uses the
handedness-normalized columns below, which are defined physically:

  ``hb_arm``        horizontal break, positive = toward the pitcher's arm side
  ``rel_side_arm``  release side, positive = toward the pitcher's arm side

With these, a left-handed and a right-handed pitcher with mirror-image
deliveries produce identical numbers, so one model serves both.

The TrackMan sign constants live in ``TRACKMAN_SIGNS`` and are the single
place to fix if a real TrackMan export shows a different convention.
"""
from __future__ import annotations

import pandas as pd

# Identity and context.
ID_COLUMNS = [
    "Source",            # statcast | trackman | rapsodo
    "GameID",
    "Date",
    "PitchTime",         # timestamp of the pitch when the source has one (used to match sources)
    "Season",
    "PitchNo",           # order within the game or session
    "PitchUID",          # the source's unique pitch id (used for review decisions and matching)
    "Pitcher",
    "PitcherId",
    "PitcherThrows",     # R | L
    "PitcherTeam",
    "Batter",
    "BatterId",
    "BatterSide",        # R | L
    "Balls",
    "Strikes",
    "TaggedPitchType",   # device or source label; PRISM reclassifies by shape
]

# Physical pitch characteristics (the Stuff model's inputs come from these).
PHYSICS_COLUMNS = [
    "RelSpeed",          # mph at release
    "SpinRate",          # rpm
    "SpinAxis",          # degrees
    "RelHeight",         # ft
    "RelSide",           # ft, TrackMan sign convention
    "Extension",         # ft
    "InducedVertBreak",  # in, movement from spin alone (gravity removed)
    "HorzBreak",         # in, TrackMan sign convention
    "PlateLocHeight",    # ft
    "PlateLocSide",      # ft, TrackMan sign convention
    "VertApprAngle",     # degrees at the front of the plate (negative = downward)
    "HorzApprAngle",     # degrees at the front of the plate
]

# Outcomes (targets and results; empty for bullpen data).
OUTCOME_COLUMNS = [
    "PitchCall",         # BallCalled | StrikeCalled | StrikeSwinging | FoulBall | InPlay | HitByPitch
    "PlayResult",        # Single, Double, Out, ... when InPlay
    "KorBB",             # Strikeout | Walk | Undefined
    "ExitSpeed",         # mph
    "Angle",             # launch angle, degrees
    "RunValue",          # change in run expectancy for this pitch (Statcast: delta_run_exp)
    "xwOBA",             # expected wOBA on contact
]

# Handedness-normalized columns the model actually uses.
NORMALIZED_COLUMNS = [
    "hb_arm",
    "rel_side_arm",
]

ALL_COLUMNS = ID_COLUMNS + PHYSICS_COLUMNS + OUTCOME_COLUMNS + NORMALIZED_COLUMNS

# TrackMan horizontal sign convention, expressed relative to a right-handed
# pitcher's arm side: +1 means a positive raw value is arm side for a RHP.
# TODO(verify): confirm against a real TrackMan export before training on
# TrackMan data. Statcast-derived data is converted using physical directions
# (see statcast.py), so only these constants depend on the assumption.
TRACKMAN_SIGNS = {
    "HorzBreak": +1,
    "RelSide": +1,
}

PITCH_CALLS = {
    "BallCalled",
    "StrikeCalled",
    "StrikeSwinging",
    "FoulBall",
    "InPlay",
    "HitByPitch",
}


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Add arm-side-positive columns from the raw TrackMan-convention ones."""
    hand = df["PitcherThrows"].map({"R": 1, "L": -1})
    df["hb_arm"] = df["HorzBreak"] * TRACKMAN_SIGNS["HorzBreak"] * hand
    df["rel_side_arm"] = df["RelSide"] * TRACKMAN_SIGNS["RelSide"] * hand
    return df


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with exactly ``ALL_COLUMNS``, in order, adding blanks."""
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[ALL_COLUMNS]
