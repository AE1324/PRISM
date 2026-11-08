# PRISM model and data pipeline

Everything that builds PRISM's grades lives here, separate from the web app.
Data is never committed: it is written to the git-ignored `data/` folder at the
repo root.

## Layout

| Path | What it does |
|------|--------------|
| `prism_data/schema.py` | PRISM's internal pitch format (TrackMan column names and units) |
| `prism_data/statcast.py` | Pulls MLB Statcast pitches and converts them to the internal format |
| `prism_data/d3dashboard.py` | Pulls D3 pitching stats, run expectancy, and reference tables |
| `tests/` | Checks for the conversions |

## One-time setup

From the repo root:

```bash
cd model
python -m venv .venv
# macOS/Linux: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Open `model/.env` and paste your D3 Dashboard API key after
`D3_DASHBOARD_API_KEY=`. The file is git-ignored, so the key never reaches GitHub.
Keys expire every 90 days.

## Pull the data

With the virtual environment active, from the `model/` folder:

```bash
python -m prism_data.d3dashboard --years 2023 2024 2025   # about a minute
python -m prism_data.statcast --years 2023 2024 2025      # 30-60 minutes the first time
```

The Statcast pull saves one month at a time, so if it stops partway, run the
same command again and it continues where it left off.

Output:

- `data/processed/statcast_<year>.parquet`: MLB pitches in PRISM's format
- `data/raw/d3dashboard/*.parquet`: D3 reference tables

## Run the checks

```bash
python -m pytest
```

## Sign conventions

Devices disagree on which direction is positive for horizontal movement and
release side. The model uses the handedness-normalized columns `hb_arm` and
`rel_side_arm` (positive = the pitcher's arm side), so lefties and righties are
graded on the same scale. The only assumption about TrackMan's raw convention
is the `TRACKMAN_SIGNS` table in `schema.py`; confirm it against a real
TrackMan export before training on TrackMan data.
