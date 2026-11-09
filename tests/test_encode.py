import pytest
from openjev.encode import QTYPES, render_options, validate_question, validate_state

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
