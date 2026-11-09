"""Validate typed questions and render their options to text.

A question has one of three types:
- ``choice``: named options, each with a description (``criteria`` is a dict).
- ``score``: ordered levels, described (``criteria`` is a list).
- ``noul``: a yes/no statement; the two options are fixed.
"""

import json
from dataclasses import dataclass

QTYPES = {"choice": 0, "score": 1, "noul": 2}

MAX_LEN = 1024
HEAD_MAX_LEN = 256
OPTION_MAX_TOKENS = 48

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


class OptionCollision(ValueError):
    """Two options truncate to the same token span, making a question unanswerable."""


def strip_mask(tok, text: str) -> str:
    """Remove any literal mask-token text, so a caller can never forge a marker."""
    return text.replace(tok.mask_token, "")


def _clean_question(tok, q: dict) -> dict:
    """Return a copy of ``q`` with the mask token stripped from every string field."""
    clean = dict(q)
    clean["instructions"] = strip_mask(tok, q["instructions"])
    if q["type"] == "choice":
        clean["criteria"] = {
            strip_mask(tok, label): strip_mask(tok, value) if isinstance(value, str) else value
            for label, value in q["criteria"].items()
        }
    elif q["type"] == "score":
        clean["criteria"] = [strip_mask(tok, v) if isinstance(v, str) else v for v in q["criteria"]]
    return clean


@dataclass
class Encoded:
    ids: list[int]
    markers: list[int]
    qtype: int
    n_options: int
    state_tokens: int
    state_tokens_used: int

    @property
    def truncated(self) -> bool:
        return self.state_tokens_used < self.state_tokens


def build_head(tok, q: dict, head_max_len: int = HEAD_MAX_LEN, qid: str = "q") -> tuple[list[int], list[int]]:
    """Build ``[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]``.

    Returns the head's token ids and the index of each option's ``[MASK]`` marker.
    Each option is truncated at the tokenizer to ``OPTION_MAX_TOKENS``; if the options
    still overflow ``head_max_len``, each is cut further to an equal share (minimum 4
    tokens) and the instructions are clamped to whatever room remains, so the markers
    always survive. Raises ``OptionCollision`` if two options truncate to the same
    token span.
    """
    option_texts = render_options(q)
    n = len(option_texts)
    option_ids = [
        tok(text, add_special_tokens=False, truncation=True, max_length=OPTION_MAX_TOKENS)["input_ids"]
        for text in option_texts
    ]

    fixed = 3 + n  # [CLS] + [SEP] after instructions + [SEP] at the end, one [MASK] per option
    total_option_tokens = sum(len(ids) for ids in option_ids)
    if fixed + total_option_tokens > head_max_len:
        share = max(4, (head_max_len - fixed) // n)
        option_ids = [ids[:share] for ids in option_ids]
        total_option_tokens = sum(len(ids) for ids in option_ids)

    seen: dict[tuple[int, ...], int] = {}
    for i, ids in enumerate(option_ids):
        key = tuple(ids)
        if key in seen:
            raise OptionCollision(
                f"question {qid!r}: options {seen[key]} and {i} truncate to the same token span"
            )
        seen[key] = i

    instructions_ids = tok(f"{q['type']} question: {q['instructions']}", add_special_tokens=False)["input_ids"]
    instructions_budget = max(0, head_max_len - fixed - total_option_tokens)
    instructions_ids = instructions_ids[:instructions_budget]

    ids = [tok.cls_token_id] + instructions_ids + [tok.sep_token_id]
    markers = []
    for opt_ids in option_ids:
        markers.append(len(ids))
        ids.append(tok.mask_token_id)
        ids.extend(opt_ids)
    ids.append(tok.sep_token_id)

    return ids, markers


def build_sequence(
    tok,
    state,
    q: dict,
    qid: str = "q",
    max_len: int = MAX_LEN,
    head_max_len: int = HEAD_MAX_LEN,
    state_ids=None,
) -> Encoded:
    """Build the full ``[CLS] ... [SEP] <state> [SEP]`` sequence the encoder reads.

    When ``state_ids`` is supplied, the state is taken as already validated and
    tokenized (so N questions about one state can share a single tokenization);
    otherwise the state is validated and tokenized here. The state gets whatever
    room is left in ``max_len`` after the head and the closing ``[SEP]``, and the
    loss, if any, is reported on the returned ``Encoded`` rather than guessed at.
    """
    clean_q = _clean_question(tok, q)
    validate_question(qid, clean_q)

    if state_ids is None:
        state_text = strip_mask(tok, validate_state(state))
        state_ids = tok(state_text, add_special_tokens=False)["input_ids"]

    state_tokens = len(state_ids)

    head_ids, markers = build_head(tok, clean_q, head_max_len=head_max_len, qid=qid)

    room = max(0, max_len - len(head_ids) - 1)
    state_tokens_used = min(state_tokens, room)

    ids = head_ids + state_ids[:state_tokens_used] + [tok.sep_token_id]

    return Encoded(
        ids=ids,
        markers=markers,
        qtype=QTYPES[clean_q["type"]],
        n_options=len(markers),
        state_tokens=state_tokens,
        state_tokens_used=state_tokens_used,
    )
