import pytest
from openjev.encode import (
    QTYPES,
    OptionBudgetExceeded,
    OptionCollision,
    build_sequence,
    render_options,
    validate_question,
    validate_state,
)

def test_qtypes_ids():
    assert QTYPES == {"choice": 0, "score": 1, "noul": 2}

def test_choice_renders_label_and_description():
    q = {"type": "choice", "instructions": "i", "criteria": {"billing": "invoices, refunds", "tech": ""}}
    assert render_options(q) == ["billing: invoices, refunds", "tech"]

def test_score_renders_numbered_levels():
    q = {"type": "score", "instructions": "i", "criteria": ["low", "high"]}
    assert render_options(q) == ["level 0: low", "level 1: high"]

def test_noul_renders_false_then_true():
    opts = render_options({"type": "noul", "instructions": "i"})
    assert len(opts) == 2 and opts[0].startswith("false") and opts[1].startswith("true")

def test_structured_criterion_renders_as_json():
    q = {"type": "choice", "instructions": "i", "criteria": {"a": {"k": 1}, "b": "x"}}
    assert render_options(q)[0] == 'a: {"k": 1}'

@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "i", "criteria": {"only": "one"}},
    {"type": "choice", "instructions": "i", "criteria": {}},
    {"type": "score", "instructions": "i", "criteria": ["only"]},
    {"type": "nope", "instructions": "i", "criteria": {"a": "", "b": ""}},
    {"type": "choice", "instructions": "", "criteria": {"a": "", "b": ""}},
])
def test_invalid_questions_rejected(q):
    with pytest.raises(ValueError, match="department"):
        validate_question("department", q)

def test_choice_option_cap():
    q = {"type": "choice", "instructions": "i", "criteria": {str(i): "" for i in range(256)}}
    with pytest.raises(ValueError):
        validate_question("big", q)

@pytest.mark.parametrize("state", ["", "   \n\t ", None])
def test_empty_state_rejected(state):
    with pytest.raises(ValueError, match="state"):
        validate_state(state)

def test_state_accepts_str_dict_list():
    assert validate_state("hi") == "hi"
    assert "k" in validate_state({"k": "v"})
    assert "a" in validate_state(["a", "b"])


def test_one_marker_per_option(tok):
    q = {"type": "choice", "instructions": "which", "criteria": {"a": "x", "b": "y", "c": "z"}}
    enc = build_sequence(tok, "some state", q)
    assert len(enc.markers) == 3 == enc.n_options
    assert all(enc.ids[m] == tok.mask_token_id for m in enc.markers)


def test_markers_are_strictly_increasing_and_in_range(tok):
    enc = build_sequence(tok, "s", {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]})
    assert enc.markers == sorted(enc.markers)
    assert max(enc.markers) < len(enc.ids) <= 1024


def test_mask_token_in_state_is_stripped(tok):
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    enc = build_sequence(tok, f"before {tok.mask_token} after", q)
    assert len(enc.markers) == 2
    assert sum(1 for i in enc.ids if i == tok.mask_token_id) == 2


def test_mask_token_in_instructions_and_options_is_stripped(tok):
    q = {"type": "choice", "instructions": f"pick {tok.mask_token}",
         "criteria": {"a": f"x {tok.mask_token}", "b": "y"}}
    enc = build_sequence(tok, "s", q)
    assert sum(1 for i in enc.ids if i == tok.mask_token_id) == 2


def test_mask_token_in_nested_criteria_value_is_stripped(tok):
    q = {"type": "choice", "instructions": "i",
         "criteria": {"a": {"note": tok.mask_token}, "b": "y"}}
    enc = build_sequence(tok, "s", q)
    assert len(enc.markers) == 2 == enc.n_options
    assert sum(1 for i in enc.ids if i == tok.mask_token_id) == 2


def test_long_state_is_truncated_and_reported(tok):
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    enc = build_sequence(tok, "word " * 4000, q)
    assert len(enc.ids) <= 1024
    assert enc.truncated is True
    assert enc.state_tokens_used < enc.state_tokens


def test_short_state_is_not_reported_as_truncated(tok):
    enc = build_sequence(tok, "short", {"type": "noul", "instructions": "i"})
    assert enc.truncated is False
    assert enc.state_tokens_used == enc.state_tokens


def test_colliding_options_raise_naming_the_question(tok):
    shared = "identical prefix " * 40
    q = {"type": "choice", "instructions": "i", "criteria": {shared + "a": "", shared + "b": ""}}
    with pytest.raises(OptionCollision, match="dept"):
        build_sequence(tok, "s", q, qid="dept")


def test_too_many_options_raise_budget_exceeded_naming_the_question(tok):
    q = {"type": "choice", "instructions": "i", "criteria": {f"o{i}": "" for i in range(230)}}
    with pytest.raises(OptionBudgetExceeded, match="dept"):
        build_sequence(tok, "s", q, qid="dept")


def test_many_short_options_encode_without_raising(tok):
    q = {"type": "choice", "instructions": "i", "criteria": {f"o{i}": "" for i in range(52)}}
    enc = build_sequence(tok, "s", q)
    assert len(enc.markers) == 52 == enc.n_options
    assert enc.markers[-1] < 256


def test_head_budget_respected_with_many_options(tok):
    q = {"type": "choice", "instructions": "i",
         "criteria": {f"label{i}": "a description that is fairly long" for i in range(20)}}
    enc = build_sequence(tok, "s", q)
    assert len(enc.markers) == 20
    assert enc.markers[-1] < 256


def test_state_ids_reuse_gives_identical_sequence(tok):
    q = {"type": "noul", "instructions": "i"}
    state_ids = tok("shared state", add_special_tokens=False)["input_ids"]
    assert build_sequence(tok, "shared state", q).ids == build_sequence(tok, None, q, state_ids=state_ids).ids
