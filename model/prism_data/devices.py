"""
Device profiles: everything PRISM assumes about a tracking device, in one place.

Rapsodo and TrackMan measure the same pitch in very different ways (a camera unit behind the plate vs a
radar in the stands), so the engine reads how far to trust each kind of number from the device's profile
instead of scattering device rules through the code. A new device is a new profile.

What a profile says
  location        measured | extrapolated. Extrapolated plate locations (Rapsodo projects them from early
                  flight) scatter more and can sit at a different level, so a staff's Location+ level is set
                  from its strike rate vs D3 and location only ranks pitchers within the staff. Measured
                  locations (TrackMan) are used as they are, regressed by sample size.
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
    notes: str = ""

    @property
    def calibrate_location(self) -> bool:
        return self.location == "extrapolated"

    @property
    def best_results_tier(self) -> str:
        return "measured" if self.contact == "measured" else "charted"


_WARNER = ("Warner Gearhart (LHP), 24 Rapsodo pitches indoors Feb 25 2026 vs 46 TrackMan pitches June 2026 (NECBL). "
           "One pitcher, four months apart: provisional.")

PROFILES: dict[str, DeviceProfile] = {
    "trackman": DeviceProfile(
        key="trackman", label="TrackMan", location="measured", release="measured", contact="measured",
        default_setting="game", movement_checks=("low_confidence", "game_calibration"),
        notes="Reference device: radar measures release, flight, plate location and batted balls directly. "
              "Calibration is set per game, so a bad day shifts every pitcher's movement the same way.",
    ),
    "rapsodo": DeviceProfile(
        key="rapsodo", label="Rapsodo", location="extrapolated", release="estimated", contact="none",
        default_setting="bullpen", movement_checks=("read_gap",),
        offsets=(
            # Agreed with TrackMan (fastball): spin rate (-6 rpm), spin axis (+0.9 deg), extension (+0.05 ft),
            # approach angle at the same plate height (+0.2 deg).
            Offset("RelHeight", -0.22, "ft", False, _WARNER + " Same on every pitch type, but that session was "
                   "indoors, where the mound often isn't game height: likely the setting, not the device."),
            Offset("rel_side_arm", 0.31, "ft", False, _WARNER + " Wider on fastball and curveball (0.29-0.31), "
                   "less on the changeup (0.10). Not explained by the mound; needs more pitchers."),
            Offset("RelSpeed", 1.1, "mph", False, _WARNER + " Can't separate from February vs mid-summer velocity."),
            Offset("InducedVertBreak", -1.1, "in", False, _WARNER + " Fastball; about 2 standard errors."),
            Offset("hb_arm", -1.2, "in", False, _WARNER + " Fastball; about 2 standard errors."),
        ),
        notes="Camera unit behind the plate: spin is read at release; movement from the flight it sees (with a "
              "spin-based backup); plate location is projected from early flight, so it scatters more.",
    ),
}

# Anything unknown is treated as conservatively as the least certain device.
UNKNOWN = DeviceProfile(key="unknown", label="Unknown device", location="extrapolated", release="estimated",
                        contact="none", default_setting="bullpen", movement_checks=())


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
