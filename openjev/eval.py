"""Evaluation metrics: the measurement that makes the calibration claim checkable.

Three numbers matter together: ``accuracy`` says how often the model is
right, ``ece`` (expected calibration error) says whether its confidence can
be trusted, and ``brier`` says the same thing from a different angle — ECE
only looks at the top prediction and depends on how it is binned, so a
confidently-wrong model with a lucky binning could otherwise hide. The
``random_baseline`` lets a reader judge accuracy without this paper in hand.
"""

import time

import numpy as np
import torch

from openjev.data import target_vector
from openjev.encode import QTYPES, build_sequence, collate, render_options, strip_mask
from openjev.model import probabilities as compute_probabilities

_QTYPE_NAMES = {idx: name for name, idx in QTYPES.items()}


def ece(confidence, correct, bins: int = 15) -> float:
    """Expected calibration error: the count-weighted mean |accuracy - confidence| per bin.

    Predictions are binned by confidence into ``bins`` equal-width bins over
    ``[0, 1]``. Empty bins are excluded rather than counted as a zero gap —
    an empty bin has no accuracy to compare against, so including it would
    silently dilute the score toward zero.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, bins - 1)

    gap_sum = 0.0
    for b in range(bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        gap_sum += n * abs(correct[mask].mean() - confidence[mask].mean())
    return float(gap_sum / len(confidence))


def brier(probs, targets) -> float:
    """Mean squared difference between the reported distribution and the target, over options."""
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean(np.sum((probs - targets) ** 2, axis=-1)))


def random_baseline(cases) -> float:
    """Mean of ``1 / n_options`` across every question asked, across all cases."""
    reciprocals = [1.0 / len(render_options(q)) for case in cases for q in case["questions"].values()]
    return float(np.mean(reciprocals))


def _cases_to_items(cases):
    """Flatten cases into ``(state, qid, question, target)`` tuples, one per question asked."""
    items = []
    for case in cases:
        state = case["state"]
        questions = case["questions"]
        expected = case.get("expected") or {}
        gold = case.get("gold") or {}
        for qid, question in questions.items():
            target = target_vector(question, expected=expected.get(qid), gold=gold.get(qid))
            items.append((state, qid, question, target))
    return items


def _batch_items(tok, chunk, state_ids_cache):
    """Encode and collate one chunk of ``(state, qid, question, target)`` tuples."""
    encoded, targets = [], []
    for state, qid, question, target in chunk:
        ids = state_ids_cache.get(state)
        if ids is None:
            ids = state_ids_cache[state] = tok(strip_mask(tok, state), add_special_tokens=False)["input_ids"]
        encoded.append(build_sequence(tok, state, question, qid=qid, state_ids=ids))
        targets.append(target)

    batch = collate(encoded, tok.pad_token_id)
    target_matrix = torch.zeros(len(targets), batch["marker_mask"].size(1))
    for row, t in enumerate(targets):
        target_matrix[row, : len(t)] = torch.tensor(t)
    return batch, target_matrix


@torch.no_grad()
def evaluate(model, tok, cases, batch_size: int = 16) -> dict:
    """Run ``model`` over ``cases`` and report accuracy, calibration and latency.

    Applies the model's fitted ``temperature`` buffer — the whole point is to
    measure the model a reader will actually get probabilities from, not the
    unscaled logits. ``within_one_level`` only ever looks at ``score``
    questions, since only those have an ordering where "one level off" means
    anything; ``choice`` has no such notion of nearly right.
    """
    items = _cases_to_items(cases)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    confidences, corrects, type_names = [], [], []
    brier_terms = []
    within_one_correct, within_one_total = 0, 0
    latencies_ms = []
    state_ids_cache: dict[str, list[int]] = {}

    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            batch, target_matrix = _batch_items(tok, chunk, state_ids_cache)
            batch = {k: v.to(device) for k, v in batch.items()}

            start = time.perf_counter()
            logits = model(**batch)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            probs = compute_probabilities(
                logits, batch["marker_mask"], qtype=batch["qtype"], temperature=model.temperature
            ).cpu()

            pred_idx = probs.argmax(-1)
            target_idx = target_matrix.argmax(-1)
            top_conf = probs.gather(1, pred_idx[:, None]).squeeze(1)
            correct = pred_idx == target_idx
            qtype_cpu = batch["qtype"].cpu()

            confidences.append(top_conf.numpy())
            corrects.append(correct.numpy())
            type_names.extend(_QTYPE_NAMES[int(t)] for t in qtype_cpu.tolist())
            brier_terms.append(((probs - target_matrix) ** 2).sum(-1).numpy())

            is_score = qtype_cpu == QTYPES["score"]
            if is_score.any():
                diff = (pred_idx[is_score] - target_idx[is_score]).abs()
                within_one_total += int(is_score.sum())
                within_one_correct += int((diff <= 1).sum())
    finally:
        model.train(was_training)

    n = len(items)
    confidences = np.concatenate(confidences) if confidences else np.array([])
    corrects = np.concatenate(corrects) if corrects else np.array([])
    brier_terms = np.concatenate(brier_terms) if brier_terms else np.array([])
    type_names = np.array(type_names)

    accuracy_by_type = {}
    ece_by_type = {}
    for name in QTYPES:
        mask = type_names == name
        if not mask.any():
            continue
        accuracy_by_type[name] = float(corrects[mask].mean())
        ece_by_type[name] = ece(confidences[mask], corrects[mask])

    latencies_ms = np.asarray(latencies_ms, dtype=float)

    return {
        "accuracy": float(corrects.mean()) if n else 0.0,
        "accuracy_by_type": accuracy_by_type,
        "within_one_level": (within_one_correct / within_one_total) if within_one_total else 0.0,
        "ece": ece(confidences, corrects) if n else 0.0,
        "ece_by_type": ece_by_type,
        "brier": float(brier_terms.mean()) if n else 0.0,
        "latency_ms_median": float(np.median(latencies_ms)) if len(latencies_ms) else 0.0,
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)) if len(latencies_ms) else 0.0,
        "random_baseline": random_baseline(cases),
        "n": n,
    }
