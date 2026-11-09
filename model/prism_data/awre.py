"""
Read AWRE pitch logs (the PitchData-*.xls exports) into PRISM's format.

AWRE logs every pitch of a game or intrasquad with the count, pitch result and
at-bat result, plus velocity and a hand-entered pitch type. It has no movement,
spin, release or location, so on its own it feeds Results and the velocity
scale; matched to Rapsodo (see ``prism_data.match``) it gets the pitch shape too.

Conventions in the export:
- velocity 0 means it wasn't entered
- pitch type "Not Tracked" / "Other" / blank means it wasn't entered
- a hit batter is logged as PitchResult "Ball" with AtBatResult "HBP"
- the second "PAofInning" column is really the pitch number within the at-bat
- no batter or pitcher handedness
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prism_data.schema import conform

PITCH_CALL = {
    "Ball": "BallCalled",
    "Strike Looking": "StrikeCalled",
    "Strike Swing and Miss": "StrikeSwinging",
    "Strike Foul": "FoulBall",
    "Strike In Play": "InPlay",
}
PLAY_RESULT = {
    "1B": "Single", "2B": "Double", "3B": "Triple", "HR": "HomeRun",
    "Ground Out": "Out", "Fly Out": "Out", "Line Out": "Out",
    "Error": "Error", "Fielders Choice": "FieldersChoice", "Sacrifice": "Sacrifice",
}
KOR_BB = {"Strike Out": "Strikeout", "BB": "Walk"}
TAG = {"Fast Ball": "Fastball", "Breaking Ball": "Breaking", "Change Up": "ChangeUp"}


def read_awre(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    raw = raw.rename(columns={"PAofInning.1": "PitchOfPA"})
    df = pd.DataFrame(index=raw.index)
    df["Source"] = "awre"
    when = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str), errors="coerce", format="mixed")
    df["PitchTime"] = when
    df["Date"] = when.dt.date
    df["Season"] = when.dt.year.astype("Int64")
    df["GameID"] = "awre-" + when.dt.strftime("%Y%m%d").fillna("unknown")
    df["PitchNo"] = pd.to_numeric(raw["PitchNo"], errors="coerce").astype("Int64")
    df["Pitcher"] = raw["Pitcher"].astype(str).str.strip()
    df["PitcherId"] = df["Pitcher"]                 # AWRE has no IDs; names are stable within a team
    df["PitcherTeam"] = raw["PitcherTeam"]
    df["Batter"] = raw["Batter"]
    df["BatterId"] = raw["Batter"]
    df["Balls"] = pd.to_numeric(raw["Balls"], errors="coerce").astype("Int64")
    df["Strikes"] = pd.to_numeric(raw["Strikes"], errors="coerce").astype("Int64")
    df["TaggedPitchType"] = raw["PitchType"].map(TAG)
    velo = pd.to_numeric(raw["PitchVelo"], errors="coerce")
    df["RelSpeed"] = velo.where(velo > 0)

    ab = raw["AtBatResult"]
    df["PitchCall"] = raw["PitchResult"].map(PITCH_CALL)
    df.loc[ab == "HBP", "PitchCall"] = "HitByPitch"
    df["PlayResult"] = np.where(df["PitchCall"] == "InPlay", ab.map(PLAY_RESULT).fillna("Undefined"), "Undefined")
    df["KorBB"] = ab.map(KOR_BB).fillna("Undefined")
    return conform(df).reset_index(drop=True)


def read_folder(folder: str | Path) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*PitchData-*.xls*"))
    return pd.concat([read_awre(f) for f in files], ignore_index=True)
