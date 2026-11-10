"""
Export every pitcher's PRISM report into one staff file for the website.

The site's workspace loads this file in the browser ("Load staff file"); it never goes to the server
or GitHub, so player data stays private even though the site and repo are public. The output lands in
the git-ignored data/ folder.

    python -m prism_model.export_staff
    python -m prism_model.export_staff --data ../data/processed/oberlin_fall_2026.parquet \
        --label "Fall 2026" --out ../data/reports/prism_staff_fall_2026.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prism_data.paths import DATA_DIR
from prism_model import arsenal, fastball_traits, references

FORMAT = "prism-staff"
VERSION = 1


def _clean(o):
    """Make a report JSON-safe: numpy scalars to Python, NaN/inf to null, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def heights(roster_csv: Path) -> dict:
    if not roster_csv.exists():
        return {}
    r = pd.read_csv(roster_csv)
    if "Name" not in r or "Height" not in r:
        return {}
    return dict(zip(r["Name"], r["Height"].map(fastball_traits.height_ft)))


def export(data: Path, out: Path, label: str, team: str, roster_csv: Path) -> dict:
    df = pd.read_parquet(data)
    if "exclude" in df:
        df = df[~df["exclude"].fillna(False).astype(bool)].copy()
    ref, mlb = references.load()
    reports = arsenal.analyze(df, ref, mlb, None, "d3", None, heights(roster_csv))
    staff = {
        "format": FORMAT,
        "version": VERSION,
        "team": team,
        "label": label,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pitchers": sorted((_clean(r) for r in reports), key=lambda r: r["pitcher"].split()[-1]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(staff, separators=(",", ":"), allow_nan=False))
    return staff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_DIR / "processed" / "oberlin_fall_2026.parquet")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "reports" / "prism_staff_fall_2026.json")
    ap.add_argument("--label", default="Fall 2026")
    ap.add_argument("--team", default="Oberlin")
    ap.add_argument("--roster", type=Path, default=DATA_DIR / "rosters" / "roster.csv")
    a = ap.parse_args()
    staff = export(a.data, a.out, a.label, a.team, a.roster)
    size = a.out.stat().st_size / 1024
    print(f"Wrote {len(staff['pitchers'])} pitchers to {a.out} ({size:,.0f} KB)")


if __name__ == "__main__":
    main()
