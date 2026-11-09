import pytest
from transformers import AutoTokenizer

import common


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(common.MODEL_ID)


def test_option_descriptions_reach_the_prompt(tok):
    schema = {"decision": {"type": "enum", "description": "Case decision",
                           "choices": ["FRAUD", "CLEAR"],
                           "descriptions": {"FRAUD": "confirmed fraudulent",
                                            "CLEAR": "legitimate activity"}}}
    prompt = common.parallel_prompt(tok, "a case", schema)
    assert "confirmed fraudulent" in prompt and "legitimate activity" in prompt


def test_score_field_renders_as_ordered_levels(tok):
    schema = {"urgency": {"type": "score", "description": "How urgent",
                          "choices": ["can wait", "today", "now"]}}
    assert common.choices_of(schema["urgency"]) == ["can wait", "today", "now"]
    trie = common.build_trie(tok, schema)
    assert [c for _, c in trie.labels] == ["can wait", "today", "now"]


def test_schema_without_descriptions_still_works(tok):
    trie = common.build_trie(tok, common.SCHEMA)
    assert len(trie.labels) == sum(len(common.choices_of(f)) for f in common.SCHEMA.values())
