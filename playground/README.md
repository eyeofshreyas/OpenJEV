# openjev playground

A small FastAPI + React app for exercising `openjev.Agent.predict` by hand:
edit a state and a set of typed questions, hit ask, see calibrated
probability bars come back from one forward call.

## Run it

```bash
./playground/start.sh
```

Starts the backend (`TINY=1` by default) and the frontend together, and
stops the backend when you Ctrl-C the frontend. Drop `TINY=1` for the real
backbone, or set `OPENJEV_CHECKPOINT=/path/to/checkpoint` once one exists:

```bash
TINY= OPENJEV_CHECKPOINT=/path/to/checkpoint ./playground/start.sh
```

Or run each half yourself:

```bash
# backend (from the repo root, with the project's .venv active)
TINY=1 uvicorn playground.backend.app:app --port 8001   # --reload for dev

# frontend
cd playground/frontend
npm install
npm run dev
```

Open the URL Vite prints. Its dev server proxies `/api/*` to `localhost:8001`
(`vite.config.js`); change the port there if you run the backend elsewhere.

The backend builds its `Agent` once at startup, the same way `2_one_pass.py`
does, and reuses that script's `resolve_checkpoint`/`build_agent` so a bad
`OPENJEV_CHECKPOINT` fails loudly instead of reaching the Hub, and the UI's
caveat banner can never disagree with what actually got loaded (`GET
/api/status`).

## API

- `GET /api/example` — the fraud-review case from `common.py`, in the shape
  `Agent.predict` takes.
- `GET /api/status` — `{mode, caveat}`: what's actually loaded (`tiny` /
  `untrained` / `checkpoint`) and the honesty caveat to show alongside any
  result.
- `POST /api/predict` — body `{state, questions}`, returns `Agent.predict`'s
  `{answers, usage}` plus `mode`/`caveat`.
