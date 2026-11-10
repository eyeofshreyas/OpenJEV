#!/usr/bin/env bash
# Run both halves of the playground. Backend port must match vite.config.js's
# proxy target (8001) unless you edit that file too.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8001}"

cd "$ROOT"
TINY="${TINY:-1}" "$ROOT/.venv/bin/uvicorn" playground.backend.app:app --port "$BACKEND_PORT" &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null' EXIT

cd "$ROOT/playground/frontend"
[ -d node_modules ] || npm install
npm run dev
