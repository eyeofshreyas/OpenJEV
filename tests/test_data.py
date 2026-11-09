import json

from openjev.data import items_from_jsonl, items_from_csv, update_budget, split_items


def test_jsonl_reads_both_label_shapes():
    items, skipped = items_from_jsonl("tests/fixtures/cases.jsonl")
    assert len(items) == 3
    assert {i.qid for i in items} == {"department", "urgent", "team"}
    assert all(abs(sum(i.target) - 1.0) < 1e-6 for i in items)


def test_jsonl_counts_skip_reasons():
    _, skipped = items_from_jsonl("tests/fixtures/cases.jsonl")
    assert skipped["empty_text"] == 1
    assert skipped["invalid_question"] == 1


def test_csv_builds_one_choice_question_from_the_label_column():
    items, question, skipped = items_from_csv("tests/fixtures/tickets.csv",
                                              text_column="body", label_column="department")
    assert question["type"] == "choice"
    assert list(question["criteria"]) == ["billing", "technical"]
    assert len(items) == 4 and skipped["empty_label"] == 1


def test_csv_tolerates_a_byte_order_mark():
    items, _, _ = items_from_csv("tests/fixtures/tickets_bom.csv",
                                 text_column="body", label_column="department")
    assert len(items) == 4


def test_update_budget_matches_the_spec_formula():
    assert update_budget(1100, 8, 8, 4) == 72
    assert update_budget(5600, 8, 8, 4) == 352


def test_split_is_deterministic_and_disjoint():
    items, _ = items_from_jsonl("tests/fixtures/cases.jsonl")
    a1, c1 = split_items(items, calib_frac=0.34, seed=0)
    a2, c2 = split_items(items, calib_frac=0.34, seed=0)
    assert [i.qid for i in a1] == [i.qid for i in a2] and [i.qid for i in c1] == [i.qid for i in c2]
    assert not ({id(i) for i in a1} & {id(i) for i in c1})
    assert len(a1) + len(c1) == len(items)


def test_jsonl_non_dict_case_is_skipped_not_raised(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("[1, 2, 3]\n")
    items, skipped = items_from_jsonl(str(p))
    assert items == []
    assert skipped["invalid_question"] == 1


def test_jsonl_non_dict_question_value_is_skipped_not_raised(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"state": "foo", "questions": {"q1": null}}\n')
    items, skipped = items_from_jsonl(str(p))
    assert items == []
    assert skipped["invalid_question"] == 1


def test_jsonl_non_dict_questions_value_is_skipped_not_raised(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"state": "foo", "questions": 7}\n')
    items, skipped = items_from_jsonl(str(p))
    assert items == []
    assert skipped["invalid_question"] == 1


def test_jsonl_non_dict_expected_is_skipped_not_raised(tmp_path):
    p = tmp_path / "bad.jsonl"
    case = {
        "state": "foo",
        "questions": {"q1": {"type": "noul", "instructions": "ok?"}},
        "expected": "not-a-dict",
    }
    p.write_text(json.dumps(case) + "\n")
    items, skipped = items_from_jsonl(str(p))
    assert items == []
    assert skipped["invalid_question"] == 1


def test_jsonl_consumes_a_case_shaped_like_the_public_loaders(tmp_path):
    """A case in the exact shape ``load_ag_news`` produces must not be skipped.

    This is the constraint Task 8's review flagged: the reader has to speak
    the same case shape the public loaders emit, not a shape invented for
    the on-disk fixture.
    """
    case = {
        "state": "Oil prices rose sharply after the embargo was announced.",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Which topic does this news text belong to?",
                "criteria": {"World": "", "Sports": "", "Business": "", "Sci/Tech": ""},
            }
        },
        "expected": {"topic": "Business"},
    }
    p = tmp_path / "loader_case.jsonl"
    p.write_text(json.dumps(case) + "\n")
    items, skipped = items_from_jsonl(str(p))
    assert len(items) == 1
    assert not skipped


def test_jsonl_accepts_a_string_noul_answer(tmp_path):
    """``load_boolq`` emits a bool, but a caller sending JSON over HTTP may
    naturally send the string form instead; the reader normalizes either.
    """
    case = {
        "state": "The report is final.",
        "questions": {"flag": {"type": "noul", "instructions": "is this final?"}},
        "expected": {"flag": "true"},
    }
    p = tmp_path / "noul_case.jsonl"
    p.write_text(json.dumps(case) + "\n")
    items, skipped = items_from_jsonl(str(p))
    assert len(items) == 1
    assert not skipped
