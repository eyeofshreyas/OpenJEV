"""Build training targets from hard labels and from teacher (gold) distributions.

A strictly proper scoring rule is maximised by reporting your true belief
*when the target is itself a belief*. Against a one-hot label its optimum is
a one-hot output, which teaches the model to be falsely certain. So targets
built from a single ``expected`` answer get label smoothing: a hard label
becomes a blunt belief instead of a certainty. A ``gold`` distribution is
already a belief — it is used as given (renormalised over the question's
options), never smoothed.
"""

import math

SMOOTHING = 0.1


def _expected_labels(q: dict) -> list:
    """Option keys in render order, in the type the ``expected`` answer uses."""
    qtype = q["type"]
    if qtype == "choice":
        return list(q["criteria"].keys())
    if qtype == "score":
        return list(range(len(q["criteria"])))
    if qtype == "noul":
        return [False, True]
    raise ValueError(f"question {q!r}: unknown type {qtype!r}")


def _gold_keys(q: dict) -> list[str]:
    """Option keys in render order, in the string form a gold distribution uses."""
    qtype = q["type"]
    if qtype == "choice":
        return list(q["criteria"].keys())
    if qtype == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    if qtype == "noul":
        return ["false", "true"]
    raise ValueError(f"question {q!r}: unknown type {qtype!r}")


def _one_hot_smoothed(hot: int, n: int, smoothing: float) -> list[float]:
    other = smoothing / (n - 1) if n > 1 else 0.0
    return [1.0 - smoothing if i == hot else other for i in range(n)]


def _gold_vector(q: dict, gold: dict) -> list[float]:
    keys = _gold_keys(q)
    probabilities = gold["probabilities"]
    raw = [float(probabilities.get(key, 0.0)) for key in keys]
    total = sum(raw)
    if total <= 0:
        raise ValueError(f"question {q!r}: gold has no mass on this question's options")
    return [v / total for v in raw]


def _expected_vector(q: dict, expected, smoothing: float) -> list[float]:
    labels = _expected_labels(q)
    n = len(labels)

    if q["type"] == "score" and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        floor = math.floor(expected)
        frac = expected - floor
        if frac == 0:
            if floor not in labels:
                raise ValueError(f"question {q!r}: level {expected!r} is not one of this question's options")
            return _one_hot_smoothed(floor, n, smoothing)
        ceil = floor + 1
        if floor not in labels or ceil not in labels:
            raise ValueError(f"question {q!r}: level {expected!r} is not one of this question's options")
        # Already a distribution split between neighbouring levels — not smoothed.
        vector = [0.0] * n
        vector[floor] = 1.0 - frac
        vector[ceil] = frac
        return vector

    if expected not in labels:
        raise ValueError(f"question {q!r}: answer {expected!r} is not one of this question's options")
    return _one_hot_smoothed(labels.index(expected), n, smoothing)


def target_vector(q: dict, expected=None, gold: dict | None = None, smoothing: float = SMOOTHING) -> list[float]:
    """Build the target distribution a question's logits are scored against.

    ``gold`` wins when both are given. It is renormalised over the question's
    options; an option the gold omits gets 0. ``expected`` builds a one-hot
    (a fractional ``score`` level splits linearly between its two neighbours
    and is not smoothed, since it is already a distribution), then label
    smoothing moves ``smoothing`` of the mass uniformly across the other
    options. An answer outside the question's options raises, naming the
    question.
    """
    if gold is not None:
        return _gold_vector(q, gold)
    return _expected_vector(q, expected, smoothing)
