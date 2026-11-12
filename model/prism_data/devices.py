"""
Device profiles: everything PRISM assumes about a tracking device, in one place.

Rapsodo and TrackMan measure the same pitch in very different ways (a camera unit behind the plate vs a
radar in the stands), so the engine reads how far to trust each kind of number from the device's profile
instead of scattering device rules through the code. A new device is a new profile.

What a profile says
  location        measured | projected: how the device gets plate location (TrackMan measures it at the plate;
                  Rapsodo projects it from early flight, about 6 in of error per pitch). Location+ holds a
                  blurred device to the D3 average as it would read through the same blur (location.py).
  release         measured | estimated: how far release height, side and extension can be trusted.
  contact         measured | none: whether the device records exit velocity and launch angle. With none,
                  Results+ can at best be charted (someone recording contact) or outcomes only.
  default_setting what an untagged session is (game | live | bullpen). TrackMan game exports are games;
                  Rapsodo sessions are bullpens until tagged.
  movement_checks the device-specific misread checks that apply:
                    read_gap          Rapsodo: its flight and spin movement reads disagreeing
                    low_confidence    TrackMan: movement it marks low confidence is blanked
                    game_calibration  TrackMan: a whole game reading off (calibration is per game)
  offsets         known differences from the reference device (TrackMan), with the evidence behind them.
                  Only offsets marked applied change any number; the rest are recorded until enough
                  pitchers confirm them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Offset:
    metric: str           # PRISM column, e.g. RelHeight
    value: float          # device minus TrackMan, in the column's units
    unit: str
    applied: bool         # True: subtract it from this device's readings before grading
    evidence: str


@dataclass(frozen=True)
class DeviceProfile:
    key: str
    label: str
    location: str             # measured | extrapolated
    release: str              # measured | estimated
    contact: str              # measured | none
    default_setting: str      # game | live | bullpen
    movement_checks: tuple[str, ...]
    offsets: tuple[Offset, ...] = field(default_factory=tuple)
    location_noise_in: float = 0.0   # plate-location error per pitch (inches); Location+ grades through it
    notes: str = ""

    @property
    def best_results_tier(self) -> str:
        return "measured" if self.contact == "measured" else "charted"


_WARNER = ("Warner Gearhart (LHP): 24 Rapsodo pitches, indoor live, Feb 25 2026, vs 46 TrackMan game pitches, "
           "June 2026 (NECBL).")
_GRAYSON = ("Grayson Black (RHP): 172 Rapsodo pitches, outdoor lives, Sep-Oct 2026, vs 75 TrackMan bullpen pitches at "
            "Driveline over three sessions, Jul 7 - Aug 7 2026. He changed a lot between those sessions (fastball run "
            "12.1 -> 15.9 in, curveball spin 2219 -> 2437 rpm), so only gaps bigger than his own drift mean much.")


PROFILES: dict[str, DeviceProfile] = {
    "trackman": DeviceProfile(
        key="trackman", label="TrackMan", location="measured", release="measured", contact="measured",
        default_setting="game", movement_checks=("low_confidence", "game_calibration"),
        notes="Reference device: radar measures release, flight, plate location and batted balls directly. "
              "Calibration is set per game, so a bad day shifts every pitcher's movement the same way. Its practice "
              "(bullpen) export writes approach angles x 39.37; the reader corrects it.",
    ),
    "rapsodo": DeviceProfile(
        key="rapsodo", label="Rapsodo", location="projected", release="estimated", contact="none",
        default_setting="bullpen", movement_checks=("read_gap",),
        # Plate-location error: Oberlin's Rapsodo locations graded 0.010 runs per pitch worse than NECBL's
        # TrackMan locations at the same strike rate (58%) and scattered more (four-seam sd 0.98 / 1.04 ft vs
        # 0.86 / 0.90). Blurring the TrackMan locations by 6 in reproduces both: value +0.0124 vs +0.0121 and
        # spread 0.99 / 1.03 ft. (Umpire calls on live-BP takes suggested less, but those calls are too fuzzy
        # to measure it.) Location+ grades Rapsodo pitches through this blur (location.noise_penalty).
        location_noise_in=6.0,
        offsets=(
            # Two pitchers so far, each months apart on the two devices, so nothing is applied yet.
            # Agreed: fastball spin axis; approach angle at the same plate height (Rapsodo -0.2 / +0.4 deg vs TrackMan,
            # opposite directions). Differed in direction between the two, so not device offsets: velocity
            # (+1.1 / -0.5 mph), fastball spin rate (-6 / -94 rpm), extension (+0.05 / +0.6-0.9 ft), release height
            # (-0.22 / -0.01 ft) and release side (+0.31 / +0.08 ft wider).
            Offset("horizontal_break", -2.3, "in", False, "Rapsodo reads less horizontal break (toward zero) on every "
                   "pitch compared: Warner fastball -1.2, curveball -3.2; Grayson fastball -3.6 (-0.5 vs his first "
                   "session), slider -1.6, curveball -2.1. The most consistent difference so far. " + _WARNER + " " + _GRAYSON),
            Offset("InducedVertBreak", -1.6, "in", False, "Fastball ride lower on Rapsodo for both: Warner -1.1, "
                   "Grayson -2.2. Not consistent on breaking balls. " + _WARNER + " " + _GRAYSON),
        ),
        notes="Camera unit behind the plate: spin is read at release; movement from the flight it sees (with a "
              "spin-based backup); plate location is projected from early flight, so it scatters more.",
    ),
}

# Anything unknown is treated as conservatively as the least certain device.
UNKNOWN = DeviceProfile(key="unknown", label="Unknown device", location="projected", release="estimated",
                        contact="none", default_setting="bullpen", movement_checks=(), location_noise_in=6.0)


def profile(key) -> DeviceProfile:
    return PROFILES.get(str(key).strip().lower(), UNKNOWN) if pd.notna(key) else UNKNOWN


def device_of(df: pd.DataFrame) -> pd.Series:
    """Each pitch's device key: the session tag's Device if present, else the data Source."""
    if "Device" in df:
        return df["Device"].astype(str).str.lower()
    if "Source" in df:
        return df["Source"].astype(str).str.lower()
    return pd.Series("unknown", index=df.index)


def apply_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each device's applied offsets from its pitches (none are applied yet)."""
    dev = device_of(df)
    out = df
    for key, prof in PROFILES.items():
        for o in prof.offsets:
            if o.applied and o.metric in df:
                if out is df:
                    out = df.copy()
                hit = dev == key
                out.loc[hit, o.metric] = out.loc[hit, o.metric] - o.value
    return out


def describe(key) -> dict:
    """A profile as plain data (for the staff file and the site)."""
    p = profile(key)
    return {"key": p.key, "label": p.label, "location": p.location, "release": p.release, "contact": p.contact,
            "best_results_tier": p.best_results_tier}
