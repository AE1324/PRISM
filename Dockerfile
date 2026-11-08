# Production image: builds the React site, then serves it and the API from one
# FastAPI process. Used by Render (render.yaml). Local development still uses
# docker-compose.yml with separate frontend and backend containers.

# ── Stage 1: build the frontend ─────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Empty API base = the site calls the API on its own address.
ENV VITE_API_BASE=""
RUN npm run build

# ── Stage 2: API + built site ───────────────────────────────────────
FROM python:3.12-slim
WORKDIR /backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend /frontend/dist ./static
ENV STATIC_DIR=/backend/static
# Render supplies PORT; default to 8000 elsewhere.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
