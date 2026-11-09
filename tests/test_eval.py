import numpy as np
import pytest

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
