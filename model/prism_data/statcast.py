"""
Pull MLB Statcast pitch data and convert it to PRISM's internal format.

Usage (from the model/ folder):
    python -m prism_data.statcast --years 2023 2024 2025

Raw pulls are saved one month at a time under data/raw/statcast/, so an
interrupted run picks up where it left off. Converted seasons are written to
data/processed/statcast_<year>.parquet.
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from prism_data.paths import PROCESSED_DIR, RAW_DIR
from prism_data.schema import add_normalized, conform

RAW_STATCAST = RAW_DIR / "statcast"

# Statcast description -> TrackMan PitchCall.
PITCH_CALL = {
    "ball": "BallCalled",
    "blocked_ball": "BallCalled",
    "pitchout": "BallCalled",
    "intent_ball": "BallCalled",
    "called_strike": "StrikeCalled",
    "swinging_strike": "StrikeSwinging",
    "swinging_strike_blocked": "StrikeSwinging",
    "foul_tip": "StrikeSwinging",
    "missed_bunt": "StrikeSwinging",
    "foul": "FoulBall",
    "foul_bunt": "FoulBall",
    "bunt_foul_tip": "FoulBall",
    "foul_pitchout": "FoulBall",
    "hit_into_play": "InPlay",
    "hit_into_play_no_out": "InPlay",
    "hit_into_play_score": "InPlay",
    "hit_by_pitch": "HitByPitch",
}

PLAY_RESULT = {
    "single": "Single",
    "double": "Double",
    "triple": "Triple",
    "home_run": "HomeRun",
    "field_error": "Error",
    "fielders_choice": "FieldersChoice",
    "fielders_choice_out": "FieldersChoice",
    "sac_fly": "Sacrifice",
    "sac_bunt": "Sacrifice",
}

KOR_BB = {
    "strikeout": "Strikeout",
    "strikeout_double_play": "Strikeout",
    "walk": "Walk",
    "intent_walk": "Walk",
}

# Statcast measures y from home plate toward the mound; the front of the
# plate is 17 inches out and the release reference plane is y = 50 ft.
FRONT_OF_PLATE_FT = 17 / 12
RELEASE_PLANE_FT = 50.0


def approach_angles(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vertical and horizontal approach angles at the front of the plate, in degrees."""
    vy_f = -np.sqrt(df["vy0"] ** 2 - 2 * df["ay"] * (RELEASE_PLANE_FT - FRONT_OF_PLATE_FT))
    t = (vy_f - df["vy0"]) / df["ay"]
    vz_f = df["vz0"] + df["az"] * t
    vx_f = df["vx0"] + df["ax"] * t
    vaa = -np.degrees(np.arctan(vz_f / vy_f))
    haa = -np.degrees(np.arctan(vx_f / vy_f))
    return vaa, haa


def convert(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw Statcast rows to PRISM's internal format.

    Statcast's x axis is from the catcher's view (positive = first-base side).
    A right-hander's arm side is the third-base side, so the arm-side value of
    a horizontal measurement is -x for a righty and +x for a lefty. The raw
    TrackMan-convention columns are written so that schema.add_normalized
    produces exactly those physical arm-side values.
    """
    df = raw[raw["game_type"] == "R"].copy()

    out = pd.DataFrame(index=df.index)
    out["Source"] = "statcast"
    out["GameID"] = df["game_pk"].astype("string")
    out["Date"] = pd.to_datetime(df["game_date"]).dt.date
    out["Season"] = df["game_year"].astype("Int64")
    out["PitchNo"] = (df["at_bat_number"] * 100 + df["pitch_number"]).astype("Int64")
    out["Pitcher"] = df.get("player_name")
    out["PitcherId"] = df["pitcher"].astype("string")
    out["PitcherThrows"] = df["p_throws"]
    out["Batter"] = pd.NA
    out["BatterId"] = df["batter"].astype("string")
    out["BatterSide"] = df["stand"]
    out["Balls"] = df["balls"].astype("Int64")
    out["Strikes"] = df["strikes"].astype("Int64")
    out["TaggedPitchType"] = df["pitch_type"]

    out["RelSpeed"] = df["release_speed"]
    out["SpinRate"] = df["release_spin_rate"]
    out["SpinAxis"] = df["spin_axis"]
    out["RelHeight"] = df["release_pos_z"]
    out["RelSide"] = -df["release_pos_x"]
    out["Extension"] = df["release_extension"]
    out["InducedVertBreak"] = df["pfx_z"] * 12
    out["HorzBreak"] = -df["pfx_x"] * 12
    out["PlateLocHeight"] = df["plate_z"]
    out["PlateLocSide"] = -df["plate_x"]
    vaa, haa = approach_angles(df)
    out["VertApprAngle"] = vaa
    out["HorzApprAngle"] = -haa

    out["PitchCall"] = df["description"].map(PITCH_CALL)
    in_play = out["PitchCall"] == "InPlay"
    out["PlayResult"] = np.where(in_play, df["events"].map(PLAY_RESULT).fillna("Out"), None)
    out["KorBB"] = df["events"].map(KOR_BB).fillna("Undefined")
    out["ExitSpeed"] = df["launch_speed"].where(in_play)
    out["Angle"] = df["launch_angle"].where(in_play)
    out["RunValue"] = df["delta_run_exp"]
    out["xwOBA"] = df["estimated_woba_using_speedangle"].where(in_play)

    out = add_normalized(out)
    return conform(out).reset_index(drop=True)


def pull_month(year: int, month: int) -> Path | None:
    """Download one month of Statcast data, skipping months already saved."""
    path = RAW_STATCAST / f"{year}-{month:02d}.parquet"
    if path.exists():
        print(f"  {path.name}: already downloaded")
        return path

    from pybaseball import cache, statcast  # imported here so tests don't need it

    cache.enable()
    last_day = calendar.monthrange(year, month)[1]
    print(f"  {year}-{month:02d}: downloading…", flush=True)
    df = statcast(start_dt=f"{year}-{month:02d}-01", end_dt=f"{year}-{month:02d}-{last_day}", verbose=False)
    if df is None or df.empty:
        print(f"  {year}-{month:02d}: no games")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  {year}-{month:02d}: {len(df):,} pitches")
    return path


def build_season(year: int) -> Path:
    """Pull every month of a season (March through October) and convert it."""
    print(f"Season {year}")
    parts = [pull_month(year, m) for m in range(3, 11)]
    raw = pd.concat([pd.read_parquet(p) for p in parts if p is not None], ignore_index=True)
    converted = convert(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"statcast_{year}.parquet"
    converted.to_parquet(out, index=False)
    print(f"  saved {len(converted):,} regular-season pitches to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()
    for year in args.years:
        build_season(year)


if __name__ == "__main__":
    main()
