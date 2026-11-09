"""Validate typed questions and render their options to text.

A question has one of three types:
- ``choice``: named options, each with a description (``criteria`` is a dict).
- ``score``: ordered levels, described (``criteria`` is a list).
- ``noul``: a yes/no statement; the two options are fixed.
"""

import json

QTYPES = {"choice": 0, "score": 1, "noul": 2}

_NOUL_FALSE = "false: no, the statement does not hold"
_NOUL_TRUE = "true: yes, the statement holds"


def _render_value(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict) -> list[str]:
    """Render a question's options to label-ordered option text."""
    qtype = q["type"]
    if qtype == "choice":
        options = []
        for label, description in q["criteria"].items():
            if description is None or description == "":
                options.append(label)
            else:
                options.append(f"{label}: {_render_value(description)}")
        return options
    if qtype == "score":
        return [f"level {i}: {_render_value(level)}" for i, level in enumerate(q["criteria"])]
    if qtype == "noul":
        return [_NOUL_FALSE, _NOUL_TRUE]
    raise ValueError(f"unknown question type {qtype!r}")


def validate_question(qid: str, q: dict) -> None:
    """Raise ValueError if the question definition is malformed."""
    qtype = q.get("type")
    if qtype not in QTYPES:
        raise ValueError(f"question {qid!r}: unknown type {qtype!r}")

    instructions = q.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"question {qid!r}: instructions must be a non-empty string")

    if qtype == "choice":
        criteria = q.get("criteria")
        if not isinstance(criteria, dict):
            raise ValueError(f"question {qid!r}: choice criteria must be a dict of label -> description")
        labels = list(criteria.keys())
    elif qtype == "score":
        criteria = q.get("criteria")
        if not isinstance(criteria, list):
            raise ValueError(f"question {qid!r}: score criteria must be a list of levels")
        labels = [str(i) for i in range(len(criteria))]
    else:  # noul
        labels = ["false", "true"]

    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(f"question {qid!r}: option labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError(f"question {qid!r}: option labels must be unique")
    if not (2 <= len(labels) <= 255):
        raise ValueError(f"question {qid!r}: must have between 2 and 255 options, got {len(labels)}")


def validate_state(state) -> str:
    """Serialize a str/dict/list state to str, rejecting an empty result."""
    if isinstance(state, str):
        text = state
    elif isinstance(state, (dict, list)):
        text = json.dumps(state, ensure_ascii=False)
    else:
        raise ValueError(f"state must be a str, dict, or list, got {type(state).__name__}")

    if not text.strip():
        raise ValueError("state must not be empty")
    return text
