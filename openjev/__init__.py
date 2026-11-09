"""The Agent API: typed questions about a state in, calibrated probabilities out.

``Agent.predict`` is the one function a caller needs. It answers every question
about a single state in one forward pass: the state is validated and tokenized
exactly once, spliced into each question's sequence, and the whole batch runs
through the model together — so asking N questions costs one encoder forward,
not N, and ``usage.forward_passes`` makes that claim checkable rather than
asserted. ``usage.truncated`` surfaces whenever the state lost tokens to the
head budget, since a caller has no other way to know it got a partial answer.
"""

import time

import torch
from transformers import AutoTokenizer

from openjev import encode
from openjev.model import DEFAULT_BACKBONE, build_model
from openjev.model import probabilities as compute_probabilities
from openjev.train import load_checkpoint


def _labels(q: dict) -> list[str]:
    """Option labels in the same render order ``build_head`` lays out markers in."""
    qtype = q["type"]
    if qtype == "choice":
        return list(q["criteria"].keys())
    if qtype == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    if qtype == "noul":
        return ["false", "true"]
    raise ValueError(f"unknown question type {qtype!r}")


def _answer(qid: str, q: dict, labels: list[str], probs: list[float]) -> dict:
    answer = {"probabilities": dict(zip(labels, probs))}
    top = max(range(len(probs)), key=probs.__getitem__)
    answer["answer_confidence"] = probs[top]

    if q["type"] == "choice":
        answer["choice"] = labels[top]
    elif q["type"] == "score":
        answer["score"] = sum(i * p for i, p in enumerate(probs))
    elif q["type"] == "noul":
        answer["noul"] = probs[labels.index("true")]
    return answer


class Agent:
    """Loads a decision model and answers typed questions about a state."""

    def __init__(self, path: str | None = None, backbone: str | None = None, device=None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if path is not None:
            model, tok, _config = load_checkpoint(path)
        else:
            tok = AutoTokenizer.from_pretrained(backbone or DEFAULT_BACKBONE)
            model = build_model(backbone=backbone)
        self.model = model.to(self.device).eval()
        self.tok = tok

    def predict(self, state, questions: dict) -> dict:
        start = time.perf_counter()

        state_text = encode.strip_mask(self.tok, encode.validate_state(state))
        state_ids = self.tok(state_text, add_special_tokens=False)["input_ids"]

        qids = list(questions)
        encoded = [
            encode.build_sequence(self.tok, state, questions[qid], qid=qid, state_ids=state_ids)
            for qid in qids
        ]
        batch = encode.collate(encoded, self.tok.pad_token_id)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        forward_passes = 0
        with torch.inference_mode():
            logits = self.model(**batch)
            forward_passes += 1
            probs = compute_probabilities(
                logits, batch["marker_mask"], qtype=batch["qtype"], temperature=self.model.temperature
            )

        answers = {}
        for i, qid in enumerate(qids):
            q = questions[qid]
            labels = _labels(q)
            row = probs[i, : len(labels)].tolist()
            answers[qid] = _answer(qid, q, labels, row)

        usage = {
            "forward_passes": forward_passes,
            "questions": len(qids),
            "ms": (time.perf_counter() - start) * 1000.0,
            "truncated": any(e.truncated for e in encoded),
            "state_tokens": encoded[0].state_tokens,
            # Per-question head length varies (instructions/options differ), so the
            # room left for the shared state can too. Report the tightest budget any
            # question actually got, so "truncated" and "state_tokens_used" never
            # contradict each other.
            "state_tokens_used": min(e.state_tokens_used for e in encoded),
        }
        return {"answers": answers, "usage": usage}


def load(path, device=None) -> Agent:
    """Load a published checkpoint into an ``Agent``, ready to ``predict``."""
    return Agent(path=path, device=device)
