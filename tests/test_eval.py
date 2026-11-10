import copy

import numpy as np
import pytest
import torch

from openjev.encode import build_sequence, collate
from openjev.eval import brier, ece, evaluate, random_baseline

from tests.conftest import CHOICE, NOUL, _score_cases, _synthetic_cases


def test_ece_of_a_perfectly_calibrated_set_is_near_zero():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    correct = rng.uniform(size=20000) < conf
    assert ece(conf, correct, bins=15) < 0.02


def test_ece_of_a_confidently_wrong_set_is_near_one():
    conf = np.full(1000, 0.99)
    assert ece(conf, np.zeros(1000, dtype=bool)) == pytest.approx(0.99, abs=0.01)


def test_ece_is_hand_computable_on_two_bins():
    conf = np.array([0.1, 0.1, 0.9, 0.9])
    correct = np.array([False, False, True, False])
    assert ece(conf, correct, bins=10) == pytest.approx((0.1 + 0.4) / 2, abs=1e-6)


def test_brier_rewards_the_truth():
    t = np.array([[1.0, 0.0, 0.0]])
    assert brier(np.array([[0.8, 0.1, 0.1]]), t) < brier(np.array([[0.4, 0.3, 0.3]]), t)


def test_random_baseline_is_the_mean_reciprocal_option_count():
    cases = [{"questions": {"a": CHOICE, "b": NOUL}, "expected": {"a": "a", "b": True}}]
    assert random_baseline(cases) == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_within_one_level_is_looser_than_exact(tiny_model, tok):
    report = evaluate(tiny_model, tok, _score_cases(n=50))
    assert report["within_one_level"] >= report["accuracy_by_type"]["score"]


def test_report_carries_latency_and_n(tiny_model, tok):
    report = evaluate(tiny_model, tok, _synthetic_cases(n=20))
    assert report["n"] == 20 and report["latency_ms_median"] > 0
    assert report["latency_ms_p95"] >= report["latency_ms_median"]


def test_brier_by_type_matches_the_single_type_present(tiny_model, tok):
    # Every case here is ``noul``, so the per-type figure must equal the aggregate.
    report = evaluate(tiny_model, tok, _synthetic_cases(n=20))
    assert report["brier_by_type"]["noul"] == pytest.approx(report["brier"])


def test_per_decision_latency_is_reported_alongside_per_call(tiny_model, tok):
    report = evaluate(tiny_model, tok, _synthetic_cases(n=20))
    assert report["latency_ms_per_decision_median"] > 0
    assert report["latency_ms_per_decision_p95"] >= report["latency_ms_per_decision_median"]


def test_a_single_partial_batch_still_reports_latency(tiny_model, tok):
    # Fewer cases than the batch size means every call is partial; the per-call
    # pool must fall back to using it rather than reporting an empty-pool zero.
    report = evaluate(tiny_model, tok, _synthetic_cases(n=5), batch_size=16)
    assert report["latency_ms_median"] > 0
    assert report["latency_ms_per_decision_median"] > 0


def test_ece_of_no_predictions_is_zero():
    assert ece(np.array([]), np.array([])) == 0.0


def test_brier_of_no_predictions_is_zero():
    assert brier(np.empty((0, 3)), np.empty((0, 3))) == 0.0


def test_random_baseline_of_no_cases_is_zero():
    assert random_baseline([]) == 0.0


def test_a_perfectly_confident_correct_prediction_scores_brier_zero(tiny_model, tok):
    """Brier's floor has to be zero, which means the eval target is the truth.

    Label smoothing is a *training* device. Scored against a target smoothed
    at 0.1, a perfectly right and perfectly confident 3-option prediction
    scores 0.015 instead of 0 — a metric with a non-zero floor that rewards
    underconfidence and is comparable to no published Brier anywhere.
    """
    model = copy.deepcopy(tiny_model)
    with torch.no_grad():  # confident by construction, so the target is the only thing left to measure
        model.scorer[-1].weight *= 1e9
        model.scorer[-1].bias *= 1e9

    states = [f"case {i}: filler text" for i in range(12)]
    batch = collate([build_sequence(tok, s, CHOICE, qid="d") for s in states], tok.pad_token_id)
    with torch.no_grad():
        top = model(**batch).argmax(-1).tolist()

    labels = list(CHOICE["criteria"])
    # Each case is labelled with the answer the model actually gives it: right
    # every time and confident every time, so Brier can only read the target.
    cases = [{"state": s, "questions": {"d": CHOICE}, "expected": {"d": labels[t]}} for s, t in zip(states, top)]

    report = evaluate(model, tok, cases)
    assert report["accuracy"] == 1.0
    assert report["brier"] == pytest.approx(0.0, abs=1e-6)
