"""
Fastball traits that are graded on top of Stuff: approach angle (VAA) and
perceived velocity from extension.

VAA is mostly decided by where a pitch crosses the plate, how high it is
released and how hard it is thrown, so a raw VAA number mostly measures
location. From flight physics we compute the VAA an MLB-average fastball
(average release height, extension and carry) would have at the pitch's own
velocity and plate height, then calibrate each tracking system's offset from
physics (Rapsodo reads ~0.4 deg steeper than Statcast). "VAA above expected"
(positive = flatter) is what is left: slot, extension and carry, the things
that make a fastball play flat at the top of the zone (think Bryan Woo). The
residual is scaled to MLB speed (angle differences grow as velocity drops) so
a D3 82 mph fastball and an MLB 95 mph one are compared fairly.

Perceived velocity adds ~1.6 mph per foot of extension beyond the MLB
average, because the hitter has less time to react.

Usage (from the model/ folder):
    python -m prism_model.fastball_traits fit     # builds data/models/fastball_traits.joblib
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

PATH = DATA_DIR / "models" / "fastball_traits.joblib"
FASTBALLS = ["Four-seam", "Sinker"]
VAA_FEATURES = ["PlateLocHeight", "RelHeight", "Extension", "RelSpeed"]
MPH_PER_FT = 1.6
MLB_AVG_EXTENSION_FT = 6.3
MIN_PITCHER_FB = 100
# Rough MLB norm: average extension (~6.35 ft) over average pitcher height (~6.25 ft).
EXT_PER_FT_HEIGHT = 1.015

# Highlight thresholds (MLB pitcher-level percentiles).
HIGH, LOW = 85, 15


G_FT = 32.174
MLB_FB = {"RelHeight": 5.76, "Extension": 6.47, "InducedVertBreak": 13.1}
MLB_FB_VELO = 94.0  # angle differences scale ~1/velocity, so residuals are put on an MLB-speed footing


def ordinal(n: int) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fastballs(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["shape_family"].isin(FASTBALLS)].copy()
    for c in VAA_FEATURES + ["VertApprAngle", "InducedVertBreak"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=VAA_FEATURES + ["VertApprAngle", "InducedVertBreak"])


def vaa_physics(velo, rel_h, ext, plate_h, ivb):
    """
    Approach angle from constant-acceleration flight: release at (rel_h, 60.5 - ext),
    crossing the front of the plate at plate_h, with gravity minus the lift implied by IVB
    (measured over the last 40 ft). Speeds use typical drag (95.5% average, 91.5% at the plate).
    """
    v = np.asarray(velo, float) * 1.4667
    vavg, vend = v * 0.955, v * 0.915
    t = (60.5 - np.asarray(ext, float) - 1.417) / vavg
    t40 = 40 / vavg
    a = -G_FT + 2 * (np.asarray(ivb, float) / 12) / t40 ** 2
    vz0 = (np.asarray(plate_h, float) - np.asarray(rel_h, float) - 0.5 * a * t ** 2) / t
    return np.degrees(np.arctan((vz0 + a * t) / vend))


def calibrate(fb: pd.DataFrame) -> tuple[float, float]:
    """
    Offset between a tracking system's measured VAA and the physics, as a line in plate height.
    Uses each pitch's own release, extension and IVB, so it only captures measurement/model
    differences (Rapsodo reads ~0.4 deg steeper than Statcast), not how flat a pitcher is.
    """
    phys = vaa_physics(fb["RelSpeed"], fb["RelHeight"], fb["Extension"], fb["PlateLocHeight"], fb["InducedVertBreak"])
    r = fb["VertApprAngle"].to_numpy(float) - phys
    x = fb["PlateLocHeight"].to_numpy(float) - 2.5
    b, c = np.polyfit(x, r, 1)
    return float(c), float(b)


def expected_vaa(fb: pd.DataFrame, offset: tuple[float, float]) -> np.ndarray:
    """VAA an MLB-average fastball (slot, extension, carry) would have at this velocity and plate height."""
    n = len(fb)
    phys = vaa_physics(fb["RelSpeed"], np.full(n, MLB_FB["RelHeight"]), np.full(n, MLB_FB["Extension"]),
                       fb["PlateLocHeight"], np.full(n, MLB_FB["InducedVertBreak"]))
    c, b = offset
    return phys + c + b * (fb["PlateLocHeight"].to_numpy(float) - 2.5)


def fit(years=(2024, 2025)) -> dict:
    cols = ["PitcherId", "Season", "shape_family", "VertApprAngle", "InducedVertBreak"] + VAA_FEATURES
    m = pd.concat([pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols) for y in years])
    fb = _fastballs(m)
    fb = fb[fb["PlateLocHeight"].between(0.5, 5.0) & fb["RelSpeed"].between(80, 105)]
    offset = calibrate(fb)
    fb["vaa_ae"] = (fb["VertApprAngle"] - expected_vaa(fb, offset)) * fb["RelSpeed"] / MLB_FB_VELO
    per = fb.groupby(["PitcherId", "Season"]).agg(n=("vaa_ae", "size"), vaa_ae=("vaa_ae", "mean"),
                                                  ext=("Extension", "mean"))
    per = per[per["n"] >= MIN_PITCHER_FB]
    q = np.arange(1, 100)
    model = {
        "offset_statcast": offset,
        "vaa_ae_pct": np.percentile(per["vaa_ae"], q),
        "ext_pct": np.percentile(per["ext"], q),
        "n_pitchers": int(len(per)),
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATH)
    return model


def load() -> dict | None:
    return joblib.load(PATH) if PATH.exists() else None


def _pct(value: float, table: np.ndarray) -> float:
    return float(np.interp(value, table, np.arange(1, 100), left=0.5, right=99.5))


def height_ft(h) -> float | None:
    """Parse roster heights like 6'2" into feet."""
    if not isinstance(h, str) or "'" not in h:
        return None
    ft, rest = h.split("'", 1)
    inch = rest.replace('"', "").strip() or "0"
    try:
        return int(ft) + int(inch) / 12
    except ValueError:
        return None


def traits(p: pd.DataFrame, model: dict, offset: tuple[float, float], height: float | None = None,
           d3_scale: pd.DataFrame | None = None) -> dict | None:
    """
    Fastball VAA and extension profile for one pitcher, with highlights.
    ``offset`` is the tracking system's VAA calibration (``calibrate`` on that system's fastballs).
    """
    fb = _fastballs(p)
    if len(fb) < 10:
        return None
    main = fb["shape_family"].value_counts().idxmax()
    fb = fb[fb["shape_family"] == main]
    velo = float(fb["RelSpeed"].mean())
    ext = float(fb["Extension"].mean())
    perceived = velo + MPH_PER_FT * (ext - MLB_AVG_EXTENSION_FT)
    vaa_ae = float(((fb["VertApprAngle"] - expected_vaa(fb, offset)) * fb["RelSpeed"] / MLB_FB_VELO).mean())
    plate_h = float(fb["PlateLocHeight"].mean())
    rel_h = float(fb["RelHeight"].mean())
    out = {
        "pitch": main, "n": int(len(fb)),
        "velo": round(velo, 1), "extension_ft": round(ext, 2), "perceived_velo": round(perceived, 1),
        "extension_bonus_mph": round(perceived - velo, 1),
        "extension_pct_mlb": round(_pct(ext, model["ext_pct"])),
        "vaa": round(float(fb["VertApprAngle"].mean()), 2),
        "vaa_above_expected": round(vaa_ae, 2),
        "vaa_ae_pct_mlb": round(_pct(vaa_ae, model["vaa_ae_pct"])),
        "avg_plate_height_ft": round(plate_h, 2), "release_height_ft": round(rel_h, 2),
    }
    if d3_scale is not None:
        lookup = lambda v: round(float(np.interp(v, d3_scale["mph"], d3_scale["percentile"], left=0.5, right=99.5)))
        out["velo_pct_d3"] = lookup(velo)
        out["perceived_pct_d3"] = lookup(perceived)
    if height:
        exp_ext = height * EXT_PER_FT_HEIGHT
        out["height_ft"] = round(height, 2)
        out["expected_extension_ft"] = round(exp_ext, 2)

    strengths, limits = [], []
    ep, vp = out["extension_pct_mlb"], out["vaa_ae_pct_mlb"]
    d3 = d3_scale is not None
    shift = (f" (D3 velocity percentile {ordinal(out['velo_pct_d3'])} → {ordinal(out['perceived_pct_d3'])})" if d3 else "")

    # Extension / perceived velocity.
    if ep >= HIGH:
        strengths.append(f"Extension is a weapon: {ext:.1f} ft ({ordinal(ep)} percentile vs MLB) makes his "
                         f"{velo:.1f} mph {main.lower()} play like {perceived:.1f}{shift}")
    elif ep <= LOW:
        text = (f"Extension is costing velocity: {ext:.1f} ft ({ordinal(ep)} percentile vs MLB) makes his "
                f"{velo:.1f} mph {main.lower()} play like {perceived:.1f}{shift}")
        if height and ext < out["expected_extension_ft"] - 0.3:
            text += (f"; at {int(height)}'{round((height % 1) * 12)}\" he'd typically get about "
                     f"{out['expected_extension_ft']:.1f} ft, so this looks mechanical (stride/landing), not size")
        gain = MPH_PER_FT * max(MLB_AVG_EXTENSION_FT - ext, 0.5)
        limits.append((text, f"Gain extension (stride length, getting down the mound): each half foot gained plays "
                             f"like ~{MPH_PER_FT / 2:.1f} mph; reaching MLB-average extension would be worth ~{gain:.1f} mph perceived."))
    elif height and ext < out["expected_extension_ft"] - 0.4:
        gap = out["expected_extension_ft"] - ext
        limits.append((f"Extension is short for his size: {ext:.1f} ft vs about {out['expected_extension_ft']:.1f} ft typical "
                       f"for his height", f"Room to gain extension; closing the gap would play like ~{MPH_PER_FT * gap:.1f} mph."))

    # Approach angle (height-, velocity- and system-adjusted; positive = flatter than expected).
    angle = (f"{vaa_ae:+.1f}° vs what an average fastball would have at his velocity and location "
             f"({ordinal(vp)} percentile vs MLB)")
    low_slot = rel_h <= 5.4
    if main == "Four-seam":
        if vp >= HIGH:
            why = "his low release" if low_slot else "his carry"
            woo = "; low slot plus a flat angle is the Bryan Woo type of fastball" if low_slot else ""
            strengths.append(f"Flat approach angle: {angle}, driven by {why}{woo}. It should miss bats at the top of the zone")
            if plate_h < 2.7:
                limits.append((f"He isn't using the flat angle: his four-seam averages {plate_h:.1f} ft at the plate",
                               "Elevate the four-seam more; the flat angle plays best at and above the top of the zone."))
    else:
        if vp <= LOW:
            strengths.append(f"Steep approach angle: {angle}, which suits a sinker working low for ground balls")
        elif vp >= HIGH:
            strengths.append(f"Unusually flat angle for a sinker: {angle}, from his low release; it can also play "
                             "at the top of the zone as a running fastball, not just down")

    # Only show the approach angle when it helps him (flat four-seam, or steep sinker).
    out["show_vaa"] = (main == "Four-seam" and vp >= HIGH) or (main != "Four-seam" and (vp <= LOW or vp >= HIGH))
    out["strengths"] = strengths
    out["limits"] = [t for t, _ in limits]
    out["fixes"] = [f for _, f in limits]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit"])
    ap.parse_args()
    m = fit()
    print(f"VAA reference from {m['n_pitchers']} MLB pitcher-seasons; Statcast offset {m['offset_statcast']}")
    print("VAA above expected, pitcher-level p10/50/90:", np.round(m["vaa_ae_pct"][[9, 49, 89]], 2))
    print("Extension p10/50/90:", np.round(m["ext_pct"][[9, 49, 89]], 2))


if __name__ == "__main__":
    main()
