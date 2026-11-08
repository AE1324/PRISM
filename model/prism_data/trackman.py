"""
Read a TrackMan CSV export into PRISM's internal format.

TrackMan's column names are PRISM's column names, so most of this is cleanup:
handedness written as Right/Left, extra pitch-result labels, junk dates, and a
check that the horizontal sign convention is what PRISM expects.

Works for game files and bullpen files; game-only columns (count, batter,
results) are simply blank for bullpens.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from prism_data.schema import add_normalized, conform

HAND = {"Right": "R", "Left": "L", "R": "R", "L": "L"}

PITCH_CALL = {
    "BallCalled": "BallCalled",
    "BallinDirt": "BallCalled",
    "BallIntentional": "BallCalled",
    "StrikeCalled": "StrikeCalled",
    "StrikeSwinging": "StrikeSwinging",
    "FoulBall": "FoulBall",
    "FoulBallNotFieldable": "FoulBall",
    "FoulBallFieldable": "FoulBall",
    "InPlay": "InPlay",
    "HitByPitch": "HitByPitch",
}

# Horizontal columns that share TrackMan's sign convention.
HORIZONTAL = ["HorzBreak", "RelSide", "PlateLocSide", "HorzApprAngle"]


def infer_hand(df: pd.DataFrame) -> pd.Series:
    """Fill missing handedness from each pitcher's release side (positive = right-handed in TrackMan)."""
    hand = df["PitcherThrows"]
    side = df.groupby("PitcherId")["RelSide"].transform("median")
    guess = np.where(side > 0, "R", np.where(side < 0, "L", None))
    return hand.fillna(pd.Series(guess, index=df.index))


def check_signs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure right-handers' fastballs show arm-side run as positive HorzBreak.

    PRISM assumes TrackMan's convention (confirmed on a real export). If a file
    ever arrives the other way around, flip the horizontal columns and warn
    rather than silently grading mirrored pitches.
    """
    righty = df[df["PitcherThrows"] == "R"]
    if righty.empty:
        return df
    hard = righty[righty["RelSpeed"] >= righty.groupby("PitcherId")["RelSpeed"].transform(lambda s: s.quantile(0.8))]
    if hard["HorzBreak"].median() < 0:
        warnings.warn("TrackMan file has mirrored horizontal signs; flipping them to PRISM's convention.")
        for col in HORIZONTAL:
            if col in df:
                df[col] = -df[col]
    return df


def read_trackman(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = raw.copy()

    df["Source"] = "trackman"
    df["PitcherId"] = df["PitcherId"].astype("string")
    df["PitcherThrows"] = df["PitcherThrows"].map(HAND)
    df["BatterSide"] = df["BatterSide"].map(HAND) if "BatterSide" in df else pd.NA
    df["Pitcher"] = df["Pitcher"] if "Pitcher" in df else df["PitcherId"]

    dates = pd.to_datetime(df["Date"], errors="coerce", format="mixed") if "Date" in df else pd.Series(pd.NaT, index=df.index)
    dates = dates.where(dates.dt.year > 2000)
    df["Date"] = dates.dt.date
    df["Season"] = dates.dt.year.astype("Int64")
    if "GameID" not in df:
        team = df["PitcherTeam"].astype(str) if "PitcherTeam" in df else ""
        df["GameID"] = dates.dt.strftime("%Y%m%d").fillna("unknown") + "-" + team

    if "PitchCall" in df:
        df["PitchCall"] = df["PitchCall"].map(PITCH_CALL)
    for col in ["Balls", "Strikes", "PitchNo"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "KorBB" in df:
        df["KorBB"] = df["KorBB"].replace({"Undefined": pd.NA}).fillna("Undefined")

    # Drop pitches TrackMan itself had low confidence in, when it reports that.
    if "PitchMovementConfidence" in df:
        df = df[df["PitchMovementConfidence"].fillna("High") != "Low"]

    df = df.dropna(subset=["RelSpeed"])
    # Check the sign convention using pitchers with a recorded hand, then use the
    # corrected release side to fill in any missing hands.
    df = check_signs(df)
    df["PitcherThrows"] = infer_hand(df)
    df = add_normalized(df)
    return conform(df).reset_index(drop=True)
