import pytest

from openjev.data import target_vector
from tests.conftest import CHOICE, NOUL, SCORE


def test_hard_choice_label_is_smoothed():
    v = target_vector(CHOICE, expected="b")
    assert v == pytest.approx([0.05, 0.9, 0.05])
    assert sum(v) == pytest.approx(1.0)


def test_smoothing_zero_gives_one_hot():
    assert target_vector(CHOICE, expected="a", smoothing=0.0) == pytest.approx([1.0, 0.0, 0.0])


def test_noul_true_is_the_second_slot():
    assert target_vector(NOUL, expected=True)[1] > target_vector(NOUL, expected=True)[0]
    assert target_vector(NOUL, expected=False)[0] > target_vector(NOUL, expected=False)[1]


def test_noul_accepts_the_string_forms_of_true_and_false():
    assert target_vector(NOUL, expected="true") == pytest.approx(target_vector(NOUL, expected=True))
    assert target_vector(NOUL, expected="false") == pytest.approx(target_vector(NOUL, expected=False))
    assert target_vector(NOUL, expected="True") == pytest.approx(target_vector(NOUL, expected=True))


def test_fractional_level_splits_between_neighbours():
    v = target_vector(SCORE, expected=1.5, smoothing=0.0)
    assert v == pytest.approx([0.0, 0.5, 0.5])


def test_integer_level_is_smoothed_like_a_label():
    assert target_vector(SCORE, expected=2) == pytest.approx([0.05, 0.05, 0.9])


def test_gold_distribution_passes_through_unsmoothed():
    g = {"probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}}
    assert target_vector(CHOICE, gold=g) == pytest.approx([0.7, 0.2, 0.1])


def test_gold_score_keys_are_level_index_strings():
    g = {"probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
    assert target_vector(SCORE, gold=g) == pytest.approx([0.1, 0.6, 0.3])


def test_gold_is_renormalised_and_missing_options_are_zero():
    assert target_vector(CHOICE, gold={"probabilities": {"a": 1.0, "b": 1.0}}) == pytest.approx([0.5, 0.5, 0.0])


@pytest.mark.parametrize("expected", ["nope", 7, -1, None])
def test_unknown_answers_rejected(expected):
    with pytest.raises(ValueError):
        target_vector(CHOICE, expected=expected)


def test_gold_wins_when_both_supplied():
    g = {"probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}}
    assert target_vector(CHOICE, expected="b", gold=g) == pytest.approx([0.7, 0.2, 0.1])
