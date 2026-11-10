import random

import pytest

from openjev.data import load_ag_news, load_boolq, load_massive, load_typed_decisions, sample_options


def test_sample_options_always_includes_gold_and_is_sized():
    rng = random.Random(0)
    labels = [f"l{i}" for i in range(60)]
    opts = sample_options(labels, "l42", 20, rng)
    assert "l42" in opts and len(opts) == 20 and len(set(opts)) == 20


def test_sample_options_is_deterministic_for_a_seed():
    labels = [f"l{i}" for i in range(60)]
    assert sample_options(labels, "l0", 20, random.Random(7)) == sample_options(labels, "l0", 20, random.Random(7))


def test_sample_options_handles_fewer_labels_than_requested():
    assert set(sample_options(["a", "b"], "a", 20, random.Random(0))) == {"a", "b"}


@pytest.mark.network
def test_typed_decisions_test_split_shape():
    cases = load_typed_decisions("test")
    assert len(cases) == 400
    assert sum(len(c["questions"]) for c in cases) == 2000
    assert all("gold" in c for c in cases)


@pytest.mark.network
def test_massive_cases_carry_twenty_options():
    cases = load_massive("validation", n_options=20, seed=0)
    q = next(iter(cases[0]["questions"].values()))
    assert len(q["criteria"]) == 20
    assert cases[0]["expected"][next(iter(cases[0]["questions"]))] in q["criteria"]


@pytest.mark.network
def test_boolq_validation_split_shape():
    cases = load_boolq("validation")
    assert len(cases) > 0
    c = cases[0]
    assert set(c["questions"]) == set(c["expected"])
    q = next(iter(c["questions"].values()))
    assert q["type"] == "noul"
    # A bool, not "true"/"false" strings — target_vector's noul path takes a bool
    # (or the normalized string form, but the producer should not rely on that).
    assert isinstance(c["expected"]["answer"], bool)


@pytest.mark.network
def test_ag_news_test_split_shape():
    cases = load_ag_news("test")
    assert len(cases) > 0
    c = cases[0]
    q = next(iter(c["questions"].values()))
    assert len(q["criteria"]) == 4
    assert c["expected"][next(iter(c["questions"]))] in q["criteria"]
