"""The playground's API: a thin FastAPI wrapper around ``openjev.Agent.predict``.

Reuses ``2_one_pass.py``'s ``resolve_checkpoint``/``build_agent`` rather than
re-deriving checkpoint handling here — that logic went through two fix rounds
closing a predicate-drift bug (an untrained model silently announcing itself
as the real comparison), and a second implementation is a second place that
bug can come back.
"""
import importlib.util
import pathlib
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import common  # noqa: E402

_spec = importlib.util.spec_from_file_location("one_pass", REPO_ROOT / "2_one_pass.py")
one_pass = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(one_pass)

import os  # noqa: E402

_tiny = bool(os.environ.get("TINY"))
_checkpoint = one_pass.resolve_checkpoint(os.environ.get("OPENJEV_CHECKPOINT"))
AGENT, MODE = one_pass.build_agent(_tiny, _checkpoint)

CAVEATS = {
    "tiny": "Untrained tiny weights (TINY=1). This shows the mechanism, not accuracy.",
    "untrained": "Real ModernBERT-base backbone, but no checkpoint is loaded "
                 "(OPENJEV_CHECKPOINT is unset) - the decision head is untrained.",
    "checkpoint": f"Running the trained checkpoint at {getattr(MODE, 'path', '')}.",
}

app = FastAPI(title="openjev playground")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    state: str
    questions: dict


@app.get("/api/example")
def example():
    return {"state": common.CONTEXT, "questions": one_pass.to_questions(common.SCHEMA)}


@app.get("/api/status")
def status():
    return {"mode": MODE.kind, "caveat": CAVEATS[MODE.kind]}


@app.post("/api/predict")
def predict(req: PredictRequest):
    result = AGENT.predict(req.state, req.questions)
    result["mode"] = MODE.kind
    result["caveat"] = CAVEATS[MODE.kind]
    return result
