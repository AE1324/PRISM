# PRISM

**Pitch Rating Integrated Stuff Model**

PRISM is a pitching analytics and pitch grading system built for Oberlin College
Baseball. It turns pitch-tracking data into grades and explanations that coaches
and pitchers can act on: what each pitch does well, what is holding it back, and
what to work on next.

## Status

PRISM is in active development, built on top of an existing pitching-dashboard
app (see [Credits](#credits)).

**Working today:** upload a Rapsodo pitching CSV and get a full set of
development dashboards for movement, velocity, release, command, and spin.

**In progress:**

- TrackMan as the native data format, with adapters that convert other sources
  (MLB Statcast for training, Rapsodo as an interim bullpen source)
- Shape-based pitch classification, independent of device pitch tags
- **Stuff+** and **Shape+** grades from gradient-boosted models, scaled to the
  Oberlin staff and to Division III
- Plain-English explanations of what is driving each grade
- **Pitch+**, blending stuff, location, and game results
- Saved sessions, pitcher pages, and season trends

## Current dashboards

| Panel | What it shows |
|-------|---------------|
| **KPI row** | Tracked pitches, strike %, peak & average velocity, arsenal size, spin efficiency |
| **Arsenal table** | Per-pitch-type profile: usage, velo, spin, spin-eff, break, strike % |
| **Movement profile** | Horizontal vs vertical break, with each pitch's average shape |
| **Velocity trend** | Velocity across the session by pitch order (consistency & fatigue) |
| **Release point** | Release side vs height (delivery repeatability & tunneling) |
| **Plate location** | Where pitches crossed the plate, coloured by strike/ball (command) |
| **Command / Spin bars** | Strike rate and spin efficiency ranked by pitch type |

## Current input format

Uploads are currently Rapsodo pitching exports: two metadata lines
(`"Player ID:"`, `"Player Name:"`), a blank line, then the pitch header row and
rows. The required columns are served live at `GET /analyze/schema`, and a
working example lives at
[frontend/public/sample-pitch-data.csv](frontend/public/sample-pitch-data.csv).
TrackMan support is the next major change.

## Data policy

This repository is public. **No raw pitch data, player data, or licensed data
(Synergy, TrackMan, Rapsodo exports) is ever committed.** Keep data in the
git-ignored `data/` folder and API keys in a local `.env` file (see
[`backend/.env.example`](backend/.env.example)).

## Run with Docker (recommended)

```bash
git clone https://github.com/AE1324/PRISM
cd PRISM
docker compose up --build
```

- Frontend: http://localhost:5173
- API + interactive docs: http://localhost:8000/docs

## Hosted version (Render)

The repo includes a production `Dockerfile` and a Render Blueprint
(`render.yaml`). Render builds one service that serves both the site and the
API from a single URL, and redeploys automatically on every push to `main`.

To set it up: create a Render account, connect GitHub, choose
**New → Blueprint**, and select this repository.

## Run locally (without Docker)

**Backend** (Python 3.12+):

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (Node 20+):

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 and upload a CSV.

## Configuration

Everything runs with zero config; these environment variables override defaults
on the backend:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API (comma-separated) |
| `CORS_ORIGIN_REGEX` | *(unset)* | Optional regex of extra allowed origins. Opt-in only — used for LAN dev; never a broad pattern in production |
| `CORS_ALLOW_CREDENTIALS` | `false` | Credentialed CORS. Leave off unless you add cookie/session auth |
| `MAX_UPLOAD_BYTES` | `26214400` (25 MB) | Reject uploads larger than this |
| `SHOW_API_DOCS` | `true` | Serve `/docs`, `/redoc`, `/openapi.json`. Set `false` to hide them |
| `RETRY_ATTEMPTS` | `3` | Analysis-pipeline retry attempts (transient failures only) |

The frontend reads `VITE_API_BASE` (default `http://localhost:8000`) to locate
the API.

### Deploying to production

The defaults are tuned for local development. For a public deployment:

- Set `CORS_ORIGINS` to your real frontend origin(s) and leave
  `CORS_ORIGIN_REGEX` unset — the API is then closed to everything else.
- Put a reverse proxy (nginx, Caddy, or your platform) in front of the API and
  cap the request body size (e.g. nginx `client_max_body_size 25m;`) and add
  basic rate limiting. The API is stateless and unauthenticated by design, so
  abuse protection belongs at this layer, not in per-process app state.
- Optionally set `SHOW_API_DOCS=false` to hide the interactive docs.

## Model and data pipeline

The grading model and its data pulls live in [`model/`](model/). See
[model/README.md](model/README.md) for setup and how to pull Statcast and
D3 Dashboard data.

## Documentation

Design docs for contributors live in [`docs/`](docs/):

- [docs/backend.md](docs/backend.md) — the stateless FastAPI analysis service.
- [docs/frontend.md](docs/frontend.md) — the React/Vite dashboard app.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code
conventions, and the one rule that keeps the two halves in sync (the report
contract).

## Credits

- PRISM began as a fork of [BSBL-Scout](https://github.com/yandersonpark1/BSBL-Scout)
  by yandersonpark1, which provided the original upload pipeline and dashboards.
- The grading methodology draws on the publicly shared write-up of Swarthmore
  College Baseball's Garnet Pitch Lab.

## License

MIT, see [LICENSE](LICENSE). The original BSBL-Scout copyright notice is retained.
