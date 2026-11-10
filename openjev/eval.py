"""Evaluation metrics: the measurement that makes the calibration claim checkable.

Four numbers matter together: ``accuracy`` says how often the model is
right, ``ece`` (expected calibration error) says whether its confidence can
be trusted, and ``brier`` says the same thing from a different angle — ECE
only looks at the top prediction and depends on how it is binned, so a
confidently-wrong model with a lucky binning could otherwise hide. Both are
also reported per question type, since a binning that gets lucky overall can
still be wrong for one type and right for another. The ``random_baseline``
lets a reader judge accuracy without this paper in hand.

Only ``torch`` is used here, not ``numpy``: this module already holds torch
tensors end to end, and the project's dependency list stops at
torch/transformers/datasets/safetensors — a second numerical library bought
nothing a tensor's own ``.mean()``/``.quantile()`` doesn't already give.
"""

import time

import torch

from openjev.data import items_from_cases
from openjev.encode import HEAD_MAX_LEN, MAX_LEN, QTYPES, build_sequence, collate, render_options, strip_mask
from openjev.model import probabilities as compute_probabilities
from openjev.model import sync


def ece(confidence, correct, bins: int = 15) -> float:
    """Expected calibration error: the count-weighted mean |accuracy - confidence| per bin.

    Predictions are binned by confidence into ``bins`` equal-width bins over
    ``[0, 1]``. Empty bins are excluded rather than counted as a zero gap —
    an empty bin has no accuracy to compare against, so including it would
    silently dilute the score toward zero.
    """
    confidence = torch.as_tensor(confidence, dtype=torch.float32)
    correct = torch.as_tensor(correct, dtype=torch.float32)
    if confidence.numel() == 0:
        return 0.0

    edges = torch.linspace(0.0, 1.0, bins + 1)
    # torch.bucketize's "right=True" is np.digitize's "right=False" for an
    # increasing boundary array — both put a boundary value into the bin
    # that starts at it, which is what the hand-computed two-bin test pins.
    bin_idx = torch.bucketize(confidence, edges[1:-1], right=True)

    gap_sum = 0.0
    for b in range(bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        gap_sum += n * abs(correct[mask].mean().item() - confidence[mask].mean().item())
    return gap_sum / confidence.numel()


def brier(probs, targets) -> float:
    """Mean squared difference between the reported distribution and the target, over options."""
    probs = torch.as_tensor(probs, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    if probs.numel() == 0:
        return 0.0
    return ((probs - targets) ** 2).sum(-1).mean().item()


def random_baseline(cases) -> float:
    """Mean of ``1 / n_options`` across every question asked, across all cases."""
    reciprocals = [1.0 / len(render_options(q)) for case in cases for q in case["questions"].values()]
    if not reciprocals:
        return 0.0
    return sum(reciprocals) / len(reciprocals)


def _cases_to_items(cases):
    """Flatten cases into ``Item``s, one per question asked, targeted at the *truth*.

    ``smoothing=0.0``: label smoothing is a training device, and a smoothed
    eval target gives Brier a non-zero floor (0.015 for a perfectly right and
    perfectly confident 3-option prediction) that rewards underconfidence and
    is not comparable to any published Brier. Accuracy and ECE go through
    argmax and correctness, so neither is affected either way.

    ``items_from_cases`` is the one case-to-``Item`` converter, shared with
    the JSONL reader, so it also means a malformed case is skipped and counted
    here rather than raising mid-evaluation.
    """
    return items_from_cases(cases, smoothing=0.0)


def _batch_items(tok, chunk, state_ids_cache, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN):
    """Encode and collate one chunk of ``Item``s."""
    encoded, targets = [], []
    for item in chunk:
        state, qid, question, target = item.state, item.qid, item.question, item.target
        ids = state_ids_cache.get(state)
        if ids is None:
            ids = state_ids_cache[state] = tok(strip_mask(tok, state), add_special_tokens=False)["input_ids"]
        encoded.append(
            build_sequence(tok, state, question, qid=qid, max_len=max_len,
                           head_max_len=head_max_len, state_ids=ids)
        )
        targets.append(target)

    batch = collate(encoded, tok.pad_token_id)
    target_matrix = torch.zeros(len(targets), batch["marker_mask"].size(1))
    for row, t in enumerate(targets):
        target_matrix[row, : len(t)] = torch.tensor(t)
    return batch, target_matrix


def _median_p95(values: torch.Tensor) -> tuple[float, float]:
    """Median and 95th percentile of a 1-D tensor, or ``(0.0, 0.0)`` if it is empty."""
    if values.numel() == 0:
        return 0.0, 0.0
    return float(torch.quantile(values, 0.5)), float(torch.quantile(values, 0.95))


@torch.no_grad()
def evaluate(model, tok, cases, batch_size: int = 16, max_len: int = MAX_LEN,
             head_max_len: int = HEAD_MAX_LEN) -> dict:
    """Run ``model`` over ``cases`` and report accuracy, calibration and latency.

    Applies the model's fitted ``temperature`` buffer — the whole point is to
    measure the model a reader will actually get probabilities from, not the
    unscaled logits. ``within_one_level`` only ever looks at ``score``
    questions, since only those have an ordering where "one level off" means
    anything; ``choice`` has no such notion of nearly right.

    Latency is reported two ways. ``latency_ms_median``/``latency_ms_p95``
    time one forward call (only the ``model(**batch)`` line — not encoding,
    collating or softmax), with the device synchronised on both edges so a
    CUDA figure times the work rather than the kernel launch. That pool only ever holds full-size batches: a
    trailing partial batch is faster to run for the boring reason that it
    has fewer items in it, and letting it into the same percentile pool as
    full batches skews "how long does a call take" with a number that is
    really about batch size. (If every call this run made was partial — e.g.
    ``len(cases) < batch_size`` — the pool falls back to every call, since
    there is nothing homogeneous to exclude it in favour of.)
    ``latency_ms_per_decision_median``/``latency_ms_per_decision_p95`` instead
    divide each call's latency by that call's own item count — including the
    partial batch — so "time per decision" stays honest regardless of which
    batch happened to be shorter.
    """
    items, skipped = _cases_to_items(cases)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    confidences, corrects, qtypes, brier_terms = [], [], [], []
    within_one_correct, within_one_total = 0, 0
    call_latencies_ms, call_sizes = [], []
    state_ids_cache: dict[str, list[int]] = {}

    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            batch, target_matrix = _batch_items(tok, chunk, state_ids_cache, max_len, head_max_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            sync(device)  # a CUDA forward returns at kernel-launch time; time the work, not the launch
            start = time.perf_counter()
            logits = model(**batch)
            sync(device)
            call_latencies_ms.append((time.perf_counter() - start) * 1000.0)
            call_sizes.append(len(chunk))

            probs = compute_probabilities(
                logits, batch["marker_mask"], qtype=batch["qtype"], temperature=model.temperature
            ).cpu()

            pred_idx = probs.argmax(-1)
            target_idx = target_matrix.argmax(-1)
            top_conf = probs.gather(1, pred_idx[:, None]).squeeze(1)
            correct = pred_idx == target_idx
            qtype_cpu = batch["qtype"].cpu()

            confidences.append(top_conf)
            corrects.append(correct.float())
            qtypes.append(qtype_cpu)
            brier_terms.append(((probs - target_matrix) ** 2).sum(-1))

            is_score = qtype_cpu == QTYPES["score"]
            if is_score.any():
                diff = (pred_idx[is_score] - target_idx[is_score]).abs()
                within_one_total += int(is_score.sum())
                within_one_correct += int((diff <= 1).sum())
    finally:
        model.train(was_training)

    n = len(items)
    confidences = torch.cat(confidences) if confidences else torch.empty(0)
    corrects = torch.cat(corrects) if corrects else torch.empty(0)
    brier_terms = torch.cat(brier_terms) if brier_terms else torch.empty(0)
    qtypes = torch.cat(qtypes) if qtypes else torch.empty(0, dtype=torch.long)

    accuracy_by_type, ece_by_type, brier_by_type = {}, {}, {}
    for name, idx in QTYPES.items():
        mask = qtypes == idx
        if not mask.any():
            continue
        accuracy_by_type[name] = corrects[mask].mean().item()
        ece_by_type[name] = ece(confidences[mask], corrects[mask])
        brier_by_type[name] = brier_terms[mask].mean().item()

    call_latencies = torch.tensor(call_latencies_ms, dtype=torch.float32)
    call_sizes_t = torch.tensor(call_sizes, dtype=torch.float32)
    full_mask = call_sizes_t == batch_size
    per_call_pool = call_latencies[full_mask] if full_mask.any() else call_latencies
    per_decision_pool = call_latencies / call_sizes_t if call_sizes_t.numel() else torch.empty(0)

    latency_ms_median, latency_ms_p95 = _median_p95(per_call_pool)
    per_decision_median, per_decision_p95 = _median_p95(per_decision_pool)

    return {
        "accuracy": corrects.mean().item() if n else 0.0,
        "accuracy_by_type": accuracy_by_type,
        "within_one_level": (within_one_correct / within_one_total) if within_one_total else 0.0,
        "ece": ece(confidences, corrects),
        "ece_by_type": ece_by_type,
        "brier": brier_terms.mean().item() if n else 0.0,
        "brier_by_type": brier_by_type,
        "latency_ms_median": latency_ms_median,
        "latency_ms_p95": latency_ms_p95,
        "latency_ms_per_decision_median": per_decision_median,
        "latency_ms_per_decision_p95": per_decision_p95,
        "random_baseline": random_baseline(cases),
        "skipped": dict(skipped),
        "n": n,
    }
