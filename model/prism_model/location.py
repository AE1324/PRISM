"""
Location+: what a pitch's locations are worth, in runs, judged against Division III.

1. Location value (MLB). A model learns, from every 2024-25 Statcast pitch, the run value a pitch tends to have
   at each plate location, by pitch type, count and whether the hitter is same- or opposite-handed (batted
   balls valued by exit velocity and launch angle, not where they landed). A pitch's location value is that
   prediction minus the average for its pitch type, count and matchup: how much its spot helped or hurt,
   independent of the count it was thrown in. Where the count or batter side is unknown (bullpen-style
   sessions), it's averaged over how often each happens in MLB.
   Validation (MLB pitcher-season pitch types, 100+ pitches, random half splits): it predicts runs on the
   other half of his pitches (r = 0.10) where the old competitive-zone-minus-middle score didn't (r = -0.03),
   and it's steadier half to half (0.54 vs 0.40).

2. Division III center (100). Synergy charts every D3 pitch into a 33-cell grid (the zone in nine, a
   one-ball-wide ring, eight regions beyond). Pooled grids from D3 staffs and their opponents' pitchers
   (data/synergy/zones/d3_zones_2026.csv) give a cell-based value: each cell worth what MLB pitches in it
   are worth. That understates how much D3 locations cost, because D3 misses land farther out inside the
   same cell. The correction is a blur ladder: MLB pitches are scattered by 0-12 extra inches, and at each
   step both the cell-based value and the true value are known. D3's cell-based value picks its step on
   the ladder; the true value at that step is the D3 center. Checked on NECBL TrackMan games (exact
   locations, a check only, not an input): the ladder recovered the measured value within one standard
   error for fastballs, sliders and changeups, where the cell-based value alone was 40-60% too low.
   Two adjustments:
   - Charting error. Synergy locations are charted by eye, which adds scatter that isn't the pitcher's.
     CHARTING_NOISE_IN (2 in) is taken off the ladder step. It is an assumption, not a measurement: the
     fastball center is 0.0145 with none, 0.0132 with 2 in and about 0.0117 with 3 in. Two outside reads sit
     at or a little below that: NECBL TrackMan fastballs measured 0.0106 (a stronger group than average D3
     staffs), and Oberlin's and the Sharks' fastball strike rates imply 0.011, give or take 0.002.
   - All of D3. The sampled staffs throw slightly more strikes than D3 as a whole; across the samples a
     point of strike rate is worth about 0.0009 of location value, which moves the center to the D3-wide
     strike rate (data/synergy/team_results_2026.csv, strength-weighted as in d3.py).
   Cutters take the fastball center (Synergy has too few).

3. Sample size: a pitcher's command counts n / (n + 150) of the way from D3 average (n = located pitches), and
   each pitch type n / (n + 430) of the way from his command. The first is measured on D3 pitchers: Synergy
   fastball grids for 39 pitchers on five staffs (data/synergy/zones/pitcher_zones_2026.csv), each put through
   the ladder. Their spread beyond sampling noise is 0.0037 runs per pitch; the same method on MLB pitchers
   recovers about three quarters of the spread exact locations show, so true D3 spread is about 0.0037-0.0049,
   1.3-1.75 times MLB's 0.0028. Against a per-pitch spread of 0.055 that is n + 125 to n + 215 (resampling
   pitchers: about 85 to 400); 150 is used. MLB's own figure is about 220-270. The second is MLB's: in
   college data, pitch-type differences in command within a pitcher weren't detectable beyond chance
   (64 pitch types), so there's no evidence to loosen it.

4. Scale: Location+ = 100 - 10 x (estimate - D3 center) / 0.0043, the spread of MLB pitcher-season pitch types.

Devices: Rapsodo projects plate location from early flight, with about 6 in of error per pitch (devices.py).
Blur moves pitches off good spots on average, so a blurred device is held to the D3 average as it would read
through the same blur: the same ladder, stepped from D3's scatter to D3's scatter plus the device's
(noise_penalty). TrackMan measures location.

    python -m prism_model.location build     # trains data/models/location_model.joblib
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR, PROCESSED_DIR

PATH = DATA_DIR / "models" / "location_model.joblib"
YEARS = (2024, 2025)
FEATURES = ["x_in", "PlateLocHeight", "Balls", "Strikes", "same", "fam"]
K_PITCHER = 150   # measured on D3 pitchers' Synergy grids (MLB: 220-270); see note 3
K_TYPE = 430
COUNTS = [(b, s) for b in range(4) for s in range(3)]
TAKE_GROUP = {"Four-seam": "Fastball", "Sinker": "Fastball", "Cutter": "Cutter", "Slider": "Slider",
              "Curveball": "Curveball", "Changeup": "Changeup", "Splitter": "Changeup"}
CHARTING_NOISE_IN = 2.0     # assumed scatter added by Synergy's by-eye charting; see note 2
ZONES_PATH = DATA_DIR / "synergy" / "zones" / "d3_zones_2026.csv"
MIN_GRID_PITCHES = 100      # a team's grid for a pitch type needs this many located pitches
LADDER_IN = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)
LADDER_SAMPLE = 150_000
SYNERGY_FAMS = {"Fastball": ["Four-seam", "Sinker"], "Slider": ["Slider"], "Curveball": ["Curveball"],
                "Changeup": ["Changeup", "Splitter"]}
# Synergy's grid, from the pitcher's view: half-width and height of the zone, and the ring's width (ft).
GRID_HALF, GRID_LOW, GRID_HIGH, GRID_BALL = 0.83, 1.5, 3.5, 0.24
GRID_OUTER = ["out_tl", "out_t", "out_tr", "out_l", "out_r", "out_bl", "out_b", "out_br"]

_CACHE: dict = {}


def _inputs(df: pd.DataFrame, fams: list[str]) -> pd.DataFrame:
    side = df["BatterSide"].astype(object) if "BatterSide" in df else pd.Series(None, index=df.index)
    x = pd.to_numeric(df["PlateLocSide"], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["x_in"] = np.where(side == "L", 1.0, -1.0) * x          # + = inside to the hitter
    out["PlateLocHeight"] = pd.to_numeric(df["PlateLocHeight"], errors="coerce")
    out["Balls"] = pd.to_numeric(df.get("Balls"), errors="coerce")
    out["Strikes"] = pd.to_numeric(df.get("Strikes"), errors="coerce")
    out["same"] = np.where(side.isin(["R", "L"]), (df["PitcherThrows"].astype(object) == side).astype(float), np.nan)
    out["fam"] = df["shape_family"].map({f: i for i, f in enumerate(fams)})
    out["_side_known"] = side.isin(["R", "L"])
    return out


def pitch_values(df: pd.DataFrame, model: dict | None = None) -> pd.Series:
    """Location value of each located pitch (runs; positive = its spot cost runs). NaN when not located."""
    model = model or load()
    fams, mdl, base = model["fams"], model["model"], model["base"]
    X = _inputs(df, fams)
    ok = X["PlateLocHeight"].notna() & np.isfinite(X["x_in"].where(X["_side_known"], 0.0)) & X["fam"].notna()
    out = pd.Series(np.nan, index=df.index)
    if not ok.any():
        return out
    X = X[ok]
    full = X["_side_known"] & X["Balls"].notna() & X["Strikes"].notna()
    if full.any():   # count and hitter side known: one prediction each
        F = X[full]
        pred = mdl.predict(F[FEATURES].astype(float))
        keys = zip(F["fam"].astype(int), F["Balls"].astype(int), F["Strikes"].astype(int), F["same"].astype(int))
        out.loc[F.index] = pred - np.array([base.get(k, np.nan) for k in keys])
    X = X[~full]
    if X.empty:
        return out
    # unknown hitter side: average over both, mirrored (x is relative to the hitter); unknown count: MLB count mix
    rows, wts, idx = [], [], []
    for i, r in X.iterrows():
        sides = [(r["x_in"], r["same"], 1.0)] if r["_side_known"] else []
        if not sides:
            raw = float(pd.to_numeric(df.at[i, "PlateLocSide"], errors="coerce"))
            p_same = model["same_share"].get(int(r["fam"]), 0.45)
            # same-handed hitter stands on the pitcher's arm side
            throws = df.at[i, "PitcherThrows"]
            x_same = raw * (-1.0 if throws == "R" else 1.0)
            sides = [(x_same, 1.0, p_same), (-x_same, 0.0, 1.0 - p_same)]
        counts = [((r["Balls"], r["Strikes"]), 1.0)] if pd.notna(r["Balls"]) and pd.notna(r["Strikes"]) else \
            [((b, s), w) for (b, s), w in model["count_mix"][int(r["fam"])].items()]
        for xi, same, ws in sides:
            for (b, s), wc in counts:
                rows.append((xi, r["PlateLocHeight"], b, s, same, r["fam"]))
                wts.append(ws * wc)
                idx.append(i)
    G = pd.DataFrame(rows, columns=FEATURES)
    pred = mdl.predict(G.astype(float))
    keys = list(zip(G["fam"].astype(int), G["Balls"].astype(int), G["Strikes"].astype(int), G["same"].astype(int)))
    val = pred - np.array([base.get(k, np.nan) for k in keys])
    agg = pd.DataFrame({"i": idx, "v": val * np.array(wts), "w": wts}).groupby("i")[["v", "w"]].sum()
    out.loc[agg.index] = agg["v"] / agg["w"]
    return out


def d3_center(kind: str, model: dict | None = None, noise_in: float = 0.0) -> float:
    """
    The D3 average location value for a pitch type, as measured by a device with ``noise_in`` inches of plate-
    location error: blur moves pitches off good spots on average, so the average a blurred device should be
    held to is the true average plus that cost (measured on MLB pitches, by pitch type).
    """
    model = model or load()
    c = float(model["d3_center"].get(TAKE_GROUP.get(kind, "Fastball"), model["d3_center"]["Fastball"]))
    return c + noise_penalty(kind, noise_in, model)


def noise_penalty(kind: str, noise_in: float, model: dict | None = None) -> float:
    """Average location value lost to ``noise_in`` inches of measurement blur, for this pitch type."""
    if not noise_in:
        return 0.0
    model = model or load()
    table = model.get("noise_penalty", {})
    if not table:
        return 0.0
    grid = sorted(table)
    vals = [table[g].get(kind, np.nanmean(list(table[g].values()))) for g in grid]
    return float(np.interp(noise_in, grid, vals))


def grade(rows: pd.DataFrame, df: pd.DataFrame, by: list[str], model: dict | None = None) -> pd.DataFrame:
    """
    Location+ for each pitcher pitch type (``rows`` keyed by ``by`` = [PitcherId, shape_family]). Adds
    ``n_located``, ``location_raw`` (mean location value, runs per pitch), ``location`` (regressed, in the same
    units) and ``location_plus``.
    """
    model = model or load()
    v = df.get("loc_value")
    if v is None:
        v = pitch_values(df, model)
    d = df.assign(_v=v).dropna(subset=["_v"])
    agg = d.groupby(by)["_v"].agg(["size", "mean"]).rename(columns={"size": "n_located", "mean": "location_raw"}).reset_index()
    rows = rows.drop(columns=[c for c in ("n_located", "location_raw", "location") if c in rows]).merge(agg, on=by, how="left")
    rows["n_located"] = rows["n_located"].fillna(0).astype(int)
    from prism_data.devices import profile

    noise = rows["device"].map(lambda k: profile(k).location_noise_in) if "device" in rows else pd.Series(0.0, index=rows.index)
    center = pd.Series([d3_center(k, model, nz) for k, nz in zip(rows["shape_family"], noise)], index=rows.index)
    dev = rows["location_raw"] - center
    n = rows["n_located"].astype(float).where(dev.notna(), 0.0)
    by_p = pd.DataFrame({"p": rows[by[0]], "x": (dev * n).fillna(0), "n": n}).groupby("p")[["x", "n"]].sum()
    raw_p = (by_p["x"] / by_p["n"].replace(0, np.nan)).fillna(0.0)
    cmd = by_p["n"] / (by_p["n"] + K_PITCHER) * raw_p                 # his command, regressed toward D3 average
    pc = rows[by[0]].map(cmd)
    est = pc + n / (n + K_TYPE) * (dev - pc)                          # each pitch type, toward his command
    rows["location"] = (center + est).where(dev.notna())
    rows["location_plus"] = (100 - 10 * est / model["scale_sd"]).where(dev.notna())
    return rows


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build() -> dict:
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingRegressor

    from prism_model import references as R

    ref, _ = R.load()
    cols = ["shape_family", "PitcherThrows", "BatterSide", "Balls", "Strikes", "PlateLocSide", "PlateLocHeight",
            "PitchCall", "RunValue", "PitcherId", "Season", "ExitSpeed", "Angle", "PlayResult", "KorBB"]
    m = pd.concat([pd.read_parquet(PROCESSED_DIR / f"statcast_{y}_shapes.parquet", columns=cols) for y in YEARS],
                  ignore_index=True)
    m = m.dropna(subset=["shape_family", "PlateLocSide", "PlateLocHeight", "PitchCall", "RunValue"])
    m = m[m["shape_family"] != "Knuckleball"].reset_index(drop=True)
    m["rv"] = R.result_values(m, ref["results"])      # batted balls by exit velocity and launch angle
    m = m.dropna(subset=["rv"]).reset_index(drop=True)
    fams = sorted(m["shape_family"].unique())
    X = _inputs(m, fams)[FEATURES].astype(float)
    mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=400,
                                        categorical_features=[5], random_state=0)
    mdl.fit(X, m["rv"])
    m["pred"] = mdl.predict(X)
    m["fam"], m["same"] = X["fam"].astype(int), X["same"].astype(int)
    base = m.groupby(["fam", "Balls", "Strikes", "same"])["pred"].mean()
    base = {(int(f), int(b), int(s), int(sm)): float(v) for (f, b, s, sm), v in base.items()}
    m["locv"] = m["pred"] - [base[(f, int(b), int(s), sm)] for f, b, s, sm in zip(m["fam"], m["Balls"], m["Strikes"], m["same"])]
    count_mix = {int(f): {(int(b), int(s)): float(w) for (b, s), w in g.groupby(["Balls", "Strikes"]).size().div(len(g)).items()}
                 for f, g in m.groupby("fam")}
    same_share = m.groupby("fam")["same"].mean().to_dict()

    key = ["PitcherId", "Season", "shape_family"]
    rows = m.groupby(key)["locv"].agg(["size", "mean"])
    rows = rows[rows["size"] >= 100]
    scale_sd = float(rows["mean"].std())

    model = {"model": mdl, "fams": fams, "base": base, "count_mix": count_mix,
             "same_share": {int(k): float(v) for k, v in same_share.items()}, "scale_sd": scale_sd}
    m["grp"] = m["shape_family"].map({f: g for g, fs in SYNERGY_FAMS.items() for f in fs})
    ladder = _blur_ladder(m[m["grp"].notna()], X, model)
    model["ladder"] = ladder
    model["d3"] = _d3_centers(ladder)
    model["d3_center"] = {k: v["center"] for k, v in model["d3"].items()}
    model["noise_penalty"] = _noise_penalties(model)
    dump(model, PATH)
    return model


# --------------------------------------------------------------------------
# D3 center from Synergy grids (note 2)
# --------------------------------------------------------------------------
def grid_cells(side, height) -> np.ndarray:
    """Synergy grid cell (0-24: the 5x5 of zone and ring, by row from the top; 25-32: GRID_OUTER) for each pitch."""
    sx, z = np.asarray(side, float), np.asarray(height, float)
    xe = np.array([-GRID_HALF, -GRID_HALF / 3, GRID_HALF / 3, GRID_HALF])
    ze = np.array([GRID_HIGH, GRID_LOW + 2 * (GRID_HIGH - GRID_LOW) / 3, GRID_LOW + (GRID_HIGH - GRID_LOW) / 3, GRID_LOW])
    in_ring = (np.abs(sx) <= GRID_HALF + GRID_BALL) & (z <= GRID_HIGH + GRID_BALL) & (z >= GRID_LOW - GRID_BALL)
    col = np.where(sx < -GRID_HALF, 0, np.where(sx > GRID_HALF, 4, 1 + np.clip(np.searchsorted(xe, sx) - 1, 0, 2)))
    row = np.where(z > GRID_HIGH, 0, np.where(z < GRID_LOW, 4, 1 + np.clip(np.searchsorted(-ze, -z) - 1, 0, 2)))
    h = np.where(sx < -GRID_HALF, 0, np.where(sx > GRID_HALF, 2, 1))
    v = np.where(z > GRID_HIGH, 0, np.where(z < GRID_LOW, 2, 1))
    outer = np.array([[0, 1, 2], [3, -1, 4], [5, 6, 7]])[v, h]      # index into GRID_OUTER
    return np.where(in_ring, row * 5 + col, 25 + outer)


def grid_counts(row) -> np.ndarray:
    """The 33 cell counts of one Synergy grid row (columns r0-r4 hold five space-separated counts each)."""
    return np.array([int(x) for i in range(5) for x in str(row[f"r{i}"]).split()] + [int(row[c]) for c in GRID_OUTER], float)


def _blur_ladder(m: pd.DataFrame, X: pd.DataFrame, model: dict, seed: int = 0) -> dict:
    """
    By Synergy pitch type: the value of each grid cell on MLB pitches, and for each step of extra scatter the
    cell-based value and the true value of the same scattered pitches.
    """
    rng = np.random.default_rng(seed)
    mdl, out = model["model"], {}
    m = m.assign(cell=grid_cells(m["PlateLocSide"], m["PlateLocHeight"]))
    for kind, g in m.groupby("grp"):
        cell_value = g.groupby("cell")["locv"].mean().reindex(range(33)).fillna(0.0).to_numpy()
        s = g.sample(min(LADDER_SAMPLE, len(g)), random_state=seed)
        F = X.loc[s.index, FEATURES].astype(float)
        base = (s["pred"] - s["locv"]).to_numpy()
        sign = np.where(s["BatterSide"].astype(object) == "L", 1.0, -1.0)   # x_in = sign * PlateLocSide
        cells, truth = [], []
        for inches in LADDER_IN:
            dx, dz = rng.normal(0, inches / 12, len(s)), rng.normal(0, inches / 12, len(s))
            B = F.copy()
            B["x_in"] = F["x_in"] + sign * dx
            B["PlateLocHeight"] = F["PlateLocHeight"] + dz
            truth.append(float((mdl.predict(B) - base).mean()))
            c = grid_cells(s["PlateLocSide"].to_numpy(float) + dx, s["PlateLocHeight"].to_numpy(float) + dz)
            cells.append(float(cell_value[c].mean()))
        out[kind] = {"cell_value": cell_value, "inches": list(LADDER_IN), "cell": cells, "truth": truth}
    return out


def ladder_value(kind: str, inches: float, ladder: dict) -> float:
    """True location value of MLB pitches of this type scattered by ``inches`` more."""
    lad = ladder[kind]
    return float(np.interp(inches, lad["inches"], lad["truth"]))


def grid_estimate(counts, kind: str, ladder: dict, charting_in: float = CHARTING_NOISE_IN) -> dict:
    """Location value implied by a grid of cell counts: cell-based value -> ladder step -> true value there."""
    lad = ladder[kind]
    counts = np.asarray(counts, float)
    cell = float((counts * lad["cell_value"]).sum() / counts.sum())
    charted = float(np.interp(cell, lad["cell"], lad["inches"]))
    scatter = float(np.sqrt(max(charted ** 2 - charting_in ** 2, 0.0)))
    return {"cell": cell, "charted_in": charted, "scatter_in": scatter, "value": ladder_value(kind, scatter, ladder)}


def _source_strike_rates() -> pd.DataFrame:
    """Strike rate by (team, matchup, pitch type) for the staffs behind the grids, where Synergy results exist."""
    parts = []
    f = DATA_DIR / "synergy" / "velo" / "d3_velo_bands_2026.csv"
    if f.exists():
        b = pd.read_csv(f).groupby(["team", "matchup", "pitch_type"])[["ball", "strike"]].sum().reset_index()
        b["matchup"] = b["matchup"].replace({"Own": "All"})
        parts.append(b.assign(strike_rate=b["strike"] / (b["strike"] + b["ball"]))[["team", "matchup", "pitch_type", "strike_rate"]])
    f = DATA_DIR / "synergy" / "team_results_2026.csv"
    if f.exists():
        t = pd.read_csv(f)
        t["team"] = t["team"].str.lower().str.replace(r"[ \-]", "_", regex=True)
        parts.append(t.assign(matchup="All", strike_rate=t["strike"] / t["total"])[["team", "matchup", "pitch_type", "strike_rate"]])
    if not parts:
        return pd.DataFrame(columns=["team", "matchup", "pitch_type", "strike_rate"])
    return pd.concat(parts).drop_duplicates(["team", "matchup", "pitch_type"])


def _d3_centers(ladder: dict) -> dict:
    """D3 average location value by Synergy pitch type, with the pieces that built it."""
    from prism_model import d3

    z = pd.read_csv(ZONES_PATH)
    z = z[z["matchup"].isin(["All", "OppPitchers"]) & z["pitch_type"].isin(ladder)].copy()
    z["counts"] = z.apply(grid_counts, axis=1)
    z["n"] = z["counts"].map(np.sum)
    z = z[z["n"] >= MIN_GRID_PITCHES].merge(_source_strike_rates(), on=["team", "matchup", "pitch_type"], how="left")
    z["value"] = [grid_estimate(c, k, ladder)["value"] for c, k in zip(z["counts"], z["pitch_type"])]
    # what a point of strike rate is worth in location value, across the sampled staffs (pitch types pooled)
    k = z.dropna(subset=["strike_rate"])
    D = pd.get_dummies(k["pitch_type"], dtype=float)
    D["strike_rate"] = k["strike_rate"]
    w = np.sqrt(k["n"].to_numpy(float))
    slope = float(np.linalg.lstsq(D.to_numpy() * w[:, None], k["value"].to_numpy() * w, rcond=None)[0][-1])
    rates = d3._d3_rates()
    out = {}
    for kind, g in z.groupby("pitch_type"):
        est = grid_estimate(np.sum(g["counts"].to_list(), axis=0), kind, ladder)
        gk = g.dropna(subset=["strike_rate"])
        sample = float(np.average(gk["strike_rate"], weights=gk["n"])) if len(gk) else np.nan
        d3_rate = rates.get(kind, {}).get("strike", np.nan)
        adj = slope * (d3_rate - sample) if np.isfinite(sample) and np.isfinite(d3_rate) else 0.0
        out[kind] = {**est, "pitches": int(g["n"].sum()), "sources": int(len(g)), "sample_strike": sample,
                     "d3_strike": float(d3_rate), "strike_slope": slope, "strike_adjustment": float(adj),
                     "uncharted_value": grid_estimate(np.sum(g["counts"].to_list(), axis=0), kind, ladder, 0.0)["value"],
                     "center": float(est["value"] + adj),
                     "scatter_in": float(np.interp(est["value"] + adj, ladder[kind]["truth"], ladder[kind]["inches"]))}
    return out


NOISE_GRID_IN = (0, 2, 4, 6, 8)


def _noise_penalties(model: dict) -> dict:
    """
    For each device blur level, the location value an average D3 pitch loses to it, by pitch family: the
    ladder stepped from D3's own scatter to D3's scatter plus the device's (independent, so they add in
    squares). D3's scatter, not MLB's: blur costs less where pitches are already off the best spots.
    """
    ladder, d3c = model["ladder"], model["d3"]
    out = {}
    for inches in NOISE_GRID_IN:
        row = {}
        for fam in model["fams"]:
            kind = TAKE_GROUP.get(fam)
            kind = kind if kind in d3c else "Fastball"
            s = d3c[kind]["scatter_in"]
            row[fam] = ladder_value(kind, float(np.hypot(s, inches)), ladder) - ladder_value(kind, s, ladder)
        out[inches] = row
    return out


def necbl_check(trackman_dir=None) -> pd.DataFrame:
    """
    Check of the ladder, not an input: on NECBL TrackMan games (exact locations, both teams), the measured
    location value by pitch type against the estimate from those same pitches reduced to grid cells.
    """
    from pathlib import Path

    from prism_model.team_data import build_trackman

    model = load()
    a = build_trackman(Path(trackman_dir) if trackman_dir else DATA_DIR / "trackman")
    a.attrs = {}
    a = a[~a["exclude"] & a["shape_family"].isin(model["fams"])].dropna(subset=["PlateLocSide", "PlateLocHeight"])
    a = a.assign(locv=pitch_values(a, model), grp=a["shape_family"].map(TAKE_GROUP)).dropna(subset=["locv"])
    rows = []
    for kind, d in a.groupby("grp"):
        if kind not in model["ladder"] or len(d) < 80:
            continue
        counts = np.bincount(grid_cells(d["PlateLocSide"], d["PlateLocHeight"]), minlength=33)
        est = grid_estimate(counts, kind, model["ladder"], charting_in=0.0)
        rows.append({"pitch": kind, "n": len(d), "measured": d["locv"].mean(), "se": d["locv"].std() / np.sqrt(len(d)),
                     "cell_based": est["cell"], "ladder": est["value"]})
    return pd.DataFrame(rows)


def load() -> dict:
    if "m" not in _CACHE:
        from joblib import load as jl

        if not PATH.exists():
            raise FileNotFoundError(f"{PATH} missing: run python -m prism_model.location build")
        _CACHE["m"] = jl(PATH)
    return _CACHE["m"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build"])
    ap.parse_args()
    m = build()
    for k, v in m["d3"].items():
        print(f"D3 center {k}: {v['center']:.4f}  ({v['pitches']:,} pitches, {v['sources']} grids; cell-based {v['cell']:.4f}, "
              f"ladder {v['uncharted_value']:.4f}, less {CHARTING_NOISE_IN:g} in charting {v['value']:.4f}, "
              f"strike rate {v['strike_adjustment']:+.4f})")
    print("device blur penalty at 6 in:", {k: round(v, 4) for k, v in m["noise_penalty"][6].items()})
    try:
        print("NECBL check (measured vs ladder):\n", necbl_check().round(4).to_string(index=False))
    except Exception as e:  # the check needs the TrackMan files; the build doesn't
        print("NECBL check skipped:", e)
    print("scale sd", round(m["scale_sd"], 4))
    print(f"saved {PATH}")


if __name__ == "__main__":
    main()
