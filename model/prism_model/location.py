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

2. Division III center (100). Synergy has D3 take rates but no locations. On MLB, a pitch type's location
   value tracks its taken-ball rate (r = 0.55-0.66). College hitters chase more, though, so take rates flatter
   college command: on NECBL TrackMan games (both teams, measured locations) location value was much worse than
   their take rates implied. So the D3 center is the NECBL measured level moved by the D3 vs NECBL difference
   in take rates (MLB slope): the take-rate estimate for D3 plus the measured college correction.

3. Sample size: a pitcher's command counts n / (n + 100) of the way from D3 average (n = located pitches), and
   each pitch type n / (n + 430) of the way from his command. The first is measured on college pitchers: true
   command varies about twice as much between college pitchers as between MLB pitchers (MLB alone gives 270),
   so a smaller sample says more (NECBL both teams, 20 pitchers: 111; Oberlin, 18: 80). The second is MLB's:
   in college data, pitch-type differences in command within a pitcher weren't detectable beyond chance
   (64 pitch types), so there's no evidence to loosen it.

4. Scale: Location+ = 100 - 10 x (estimate - D3 center) / 0.0043, the spread of MLB pitcher-season pitch types.

Devices: Rapsodo projects plate location from early flight, with about 6 in of error per pitch (devices.py).
Blur moves pitches off good spots on average, so a blurred device is held to the D3 average as it would read
through the same blur (noise_penalty, measured by blurring NECBL TrackMan locations). TrackMan measures location.

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
K_PITCHER = 100   # college-measured (MLB: 270); see note 3
K_TYPE = 430
COUNTS = [(b, s) for b in range(4) for s in range(3)]
TAKE_GROUP = {"Four-seam": "Fastball", "Sinker": "Fastball", "Cutter": "Cutter", "Slider": "Slider",
              "Curveball": "Curveball", "Changeup": "Changeup", "Splitter": "Changeup"}
COLLEGE_MIN_PITCHES = 200   # a pitch group needs this many NECBL pitches for its own college correction

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
def build(trackman_dir=None) -> dict:
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

    # D3 center: MLB take-rate fit per pitch group, corrected by measured college (NECBL TrackMan) data
    m["tb"], m["ts"] = (m["PitchCall"] == "BallCalled").astype(float), (m["PitchCall"] == "StrikeCalled").astype(float)
    g = m.groupby(key).agg(n=("locv", "size"), locv=("locv", "mean"), tb=("tb", "mean"), ts=("ts", "mean")).reset_index()
    g = g[g["n"] >= 100]
    g["grp"] = g["shape_family"].map(TAKE_GROUP)
    fits = {}
    for k, d in g.groupby("grp"):
        Xf = np.c_[np.ones(len(d)), d["tb"], d["ts"]]
        w = np.sqrt(d["n"].to_numpy())
        fits[k] = np.linalg.lstsq(Xf * w[:, None], d["locv"] * w, rcond=None)[0].tolist()
    model = {"model": mdl, "fams": fams, "base": base, "count_mix": count_mix,
             "same_share": {int(k): float(v) for k, v in same_share.items()}, "scale_sd": scale_sd, "fits": fits}
    college = _college_correction(model, trackman_dir)
    syn = pd.read_csv(DATA_DIR / "synergy" / "teams_2026.csv")
    syn = syn[syn["cut"] == "all"] if "cut" in syn else syn
    centers = {}
    for k, b in fits.items():
        d = syn[syn["pitch_type"] == k]
        if d.empty:
            d = syn[syn["pitch_type"] == "Overall"]
        tb, ts = d["take_ball"].sum() / d["pitches"].sum(), d["take_strike"].sum() / d["pitches"].sum()
        corr = college["by_group"].get(k, college["pooled"])
        centers[k] = float(b[0] + b[1] * tb + b[2] * ts + corr)
    model.update(d3_center=centers, college=college)
    model["noise_penalty"] = _noise_penalties(college.pop("pitches_df"), model)
    dump(model, PATH)
    return model


NOISE_GRID_IN = (0, 2, 4, 6, 8)


def _noise_penalties(college: pd.DataFrame, model: dict, draws: int = 20, seed: int = 0) -> dict:
    """
    For each blur level, the average location value lost when college (NECBL TrackMan, both teams) locations are
    blurred, by pitch group (groups with under 200 pitches use the pooled value). College, not MLB: the D3
    center is a college level, and blur costs less where pitches are already scattered off the best spots.
    """
    rng = np.random.default_rng(seed)
    cols = ["shape_family", "PitcherThrows", "BatterSide", "Balls", "Strikes", "PlateLocSide", "PlateLocHeight"]
    smp = college[cols].copy()
    clean = pitch_values(smp, model)
    grp = smp["shape_family"].map(TAKE_GROUP)
    out = {0: {k: 0.0 for k in model["fams"]}}
    for g in NOISE_GRID_IN[1:]:
        diffs = []
        for _ in range(draws):
            x = smp.copy()
            x["PlateLocSide"] = x["PlateLocSide"] + rng.normal(0, g / 12, len(x))
            x["PlateLocHeight"] = x["PlateLocHeight"] + rng.normal(0, g / 12, len(x))
            diffs.append(pitch_values(x, model) - clean)
        diff = pd.concat(diffs, axis=1).mean(axis=1)
        pooled = float(diff.mean())
        by_grp = {k: float(v.mean()) for k, v in diff.groupby(grp) if len(v) >= COLLEGE_MIN_PITCHES}
        out[g] = {f: by_grp.get(TAKE_GROUP.get(f), pooled) for f in model["fams"]}
    return out


def _college_correction(model: dict, trackman_dir=None) -> dict:
    """Measured minus take-rate-implied location value on NECBL TrackMan games (both teams), by pitch group."""
    from pathlib import Path

    from prism_model.team_data import build_trackman

    folder = Path(trackman_dir) if trackman_dir else DATA_DIR / "trackman"
    a = build_trackman(folder)
    a.attrs = {}
    a = a[~a["exclude"] & a["shape_family"].isin(model["fams"])].dropna(subset=["PlateLocSide", "PlateLocHeight"])
    a = a.assign(locv=pitch_values(a, model), grp=a["shape_family"].map(TAKE_GROUP))
    call = a["PitchCall"].astype(object)
    a["tb"], a["ts"] = (call == "BallCalled").astype(float), (call == "StrikeCalled").astype(float)
    by_group, n_all, x_all = {}, 0, 0.0
    for k, d in a.dropna(subset=["locv"]).groupby("grp"):
        b = model["fits"].get(k)
        if b is None:
            continue
        gap = float(d["locv"].mean() - (b[0] + b[1] * d["tb"].mean() + b[2] * d["ts"].mean()))
        n_all += len(d)
        x_all += gap * len(d)
        if len(d) >= COLLEGE_MIN_PITCHES:
            by_group[k] = gap
    return {"by_group": by_group, "pooled": x_all / n_all if n_all else 0.0, "pitches": int(n_all),
            "pitches_df": a.dropna(subset=["locv"])}


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
    print("D3 centers (runs per pitch):", {k: round(v, 4) for k, v in m["d3_center"].items()})
    print("college correction:", {k: round(v, 4) for k, v in m["college"]["by_group"].items()},
          "pooled", round(m["college"]["pooled"], 4), "on", m["college"]["pitches"], "NECBL pitches")
    print("scale sd", round(m["scale_sd"], 4))
    print(f"saved {PATH}")


if __name__ == "__main__":
    main()
