"""Build training targets from hard labels and from teacher (gold) distributions.

A strictly proper scoring rule is maximised by reporting your true belief
*when the target is itself a belief*. Against a one-hot label its optimum is
a one-hot output, which teaches the model to be falsely certain. So targets
built from a single ``expected`` answer get label smoothing: a hard label
becomes a blunt belief instead of a certainty. A ``gold`` distribution is
already a belief — it is used as given (renormalised over the question's
options), never smoothed.
"""

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer

from openjev.encode import OptionBudgetExceeded, OptionCollision, build_head, validate_question, validate_state
from openjev.model import DEFAULT_BACKBONE

SMOOTHING = 0.1


def _expected_labels(q: dict) -> list:
    """Option keys in render order, in the type the ``expected`` answer uses."""
    qtype = q["type"]
    if qtype == "choice":
        return list(q["criteria"].keys())
    if qtype == "score":
        return list(range(len(q["criteria"])))
    if qtype == "noul":
        return [False, True]
    raise ValueError(f"question {q!r}: unknown type {qtype!r}")


def _gold_keys(q: dict) -> list[str]:
    """Option keys in render order, in the string form a gold distribution uses."""
    qtype = q["type"]
    if qtype == "choice":
        return list(q["criteria"].keys())
    if qtype == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    if qtype == "noul":
        return ["false", "true"]
    raise ValueError(f"question {q!r}: unknown type {qtype!r}")


def _one_hot_smoothed(hot: int, n: int, smoothing: float) -> list[float]:
    other = smoothing / (n - 1) if n > 1 else 0.0
    return [1.0 - smoothing if i == hot else other for i in range(n)]


def _gold_vector(q: dict, gold: dict) -> list[float]:
    keys = _gold_keys(q)
    probabilities = gold["probabilities"]
    raw = [float(probabilities.get(key, 0.0)) for key in keys]
    total = sum(raw)
    if total <= 0:
        raise ValueError(f"question {q!r}: gold has no mass on this question's options")
    return [v / total for v in raw]


def _expected_vector(q: dict, expected, smoothing: float) -> list[float]:
    labels = _expected_labels(q)
    n = len(labels)

    if q["type"] == "noul" and isinstance(expected, str) and expected.strip().lower() in ("true", "false"):
        # A producer sending JSON over HTTP (or hand-writing a fixture) may naturally
        # reach for the string form; both ends of this project's own case shape only
        # ever carry a bool here, but a caller outside our control is not guaranteed to.
        expected = expected.strip().lower() == "true"

    if q["type"] == "score" and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        floor = math.floor(expected)
        frac = expected - floor
        if frac == 0:
            if floor not in labels:
                raise ValueError(f"question {q!r}: level {expected!r} is not one of this question's options")
            return _one_hot_smoothed(floor, n, smoothing)
        ceil = floor + 1
        if floor not in labels or ceil not in labels:
            raise ValueError(f"question {q!r}: level {expected!r} is not one of this question's options")
        # Already a distribution split between neighbouring levels — not smoothed.
        vector = [0.0] * n
        vector[floor] = 1.0 - frac
        vector[ceil] = frac
        return vector

    if expected not in labels:
        raise ValueError(f"question {q!r}: answer {expected!r} is not one of this question's options")
    return _one_hot_smoothed(labels.index(expected), n, smoothing)


def target_vector(q: dict, expected=None, gold: dict | None = None, smoothing: float = SMOOTHING) -> list[float]:
    """Build the target distribution a question's logits are scored against.

    ``gold`` wins when both are given. It is renormalised over the question's
    options; an option the gold omits gets 0. ``expected`` builds a one-hot
    (a fractional ``score`` level splits linearly between its two neighbours
    and is not smoothed, since it is already a distribution), then label
    smoothing moves ``smoothing`` of the mass uniformly across the other
    options. An answer outside the question's options raises, naming the
    question.
    """
    if gold is not None:
        return _gold_vector(q, gold)
    return _expected_vector(q, expected, smoothing)


@dataclass
class Item:
    state: str
    qid: str
    question: dict
    target: list[float]


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)


def _build_item(state, qid, question, expected, gold, smoothing, skipped: Counter) -> Item | None:
    """Turn one (state, question, answer) triple into an ``Item``, or count why not.

    Never raises: every way a row can be malformed — an empty state, a question
    with a bad or unknown type, an answer outside the question's options, or
    options that can't be encoded — is caught here and tallied by reason.
    """
    try:
        text = validate_state(state)
    except (ValueError, TypeError):
        skipped["empty_text"] += 1
        return None

    try:
        validate_question(qid, question)
    except (ValueError, TypeError, AttributeError):
        skipped["invalid_question"] += 1
        return None

    try:
        target = target_vector(question, expected=expected, gold=gold, smoothing=smoothing)
    except (ValueError, TypeError, KeyError):
        skipped["invalid_target"] += 1
        return None

    try:
        build_head(_tokenizer(), question, qid=qid)
    except OptionCollision:
        skipped["options_collapsed"] += 1
        return None
    except OptionBudgetExceeded:
        skipped["options_beyond_max_len"] += 1
        return None

    return Item(state=text, qid=qid, question=question, target=target)


def items_from_cases(cases, smoothing: float = SMOOTHING) -> tuple[list[Item], Counter]:
    """Turn in-memory cases into ``Item``s, one per (case, question).

    A case is ``{"state": ..., "questions": {qid: question, ...},
    "expected"|"gold": {qid: answer, ...}}`` — the shape every public loader in
    this module emits (``load_typed_decisions``, ``load_massive``,
    ``load_boolq``, ``load_ag_news``) and the shape ``items_from_jsonl`` reads
    off disk. This is where that conversion lives, so calibrating or
    evaluating on a loader's output needs no round trip through a file, and so
    the file reader and the in-memory reader can never disagree about which
    rows are valid.

    Nothing here raises: every malformed shape is counted by reason and the
    rest of the data is still read.
    """
    items: list[Item] = []
    skipped: Counter = Counter()

    for case in cases:
        if not isinstance(case, dict):
            skipped["invalid_question"] += 1
            continue

        state = case.get("state")
        questions = case.get("questions", {})
        if not isinstance(questions, dict):
            skipped["invalid_question"] += 1
            continue

        expected_raw = case.get("expected")
        gold_raw = case.get("gold")
        # "expected"/"gold" are each optional, but if present must be a
        # qid -> answer dict; a present-but-wrong-shaped one invalidates
        # every question in the case, each counted individually below.
        answers_malformed = (expected_raw is not None and not isinstance(expected_raw, dict)) or (
            gold_raw is not None and not isinstance(gold_raw, dict)
        )
        expected_map = expected_raw if isinstance(expected_raw, dict) else {}
        gold_map = gold_raw if isinstance(gold_raw, dict) else {}

        for qid, question in questions.items():
            if not isinstance(question, dict):
                skipped["invalid_question"] += 1
                continue
            if answers_malformed:
                skipped["invalid_question"] += 1
                continue
            item = _build_item(
                state,
                qid,
                question,
                expected_map.get(qid),
                gold_map.get(qid),
                smoothing,
                skipped,
            )
            if item is not None:
                items.append(item)

    return items, skipped


def items_from_jsonl(path, smoothing: float = SMOOTHING) -> tuple[list[Item], Counter]:
    """Read labelled decisions from a JSON-lines file of cases.

    Each line is one case in ``items_from_cases``' shape, which does the
    actual conversion; this function only turns the file into cases. A line
    that is not JSON at all never raises past here — it is counted under
    ``invalid_question`` and reading continues.
    """
    cases = []
    skipped: Counter = Counter()

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                skipped["invalid_question"] += 1

    items, case_skipped = items_from_cases(cases, smoothing)
    return items, skipped + case_skipped


def items_from_csv(
    path,
    text_column: str,
    label_column: str,
    question_id: str = "label",
    instructions: str | None = None,
    smoothing: float = SMOOTHING,
) -> tuple[list[Item], dict, Counter]:
    """Read labelled decisions from a CSV, deriving one choice question from a label column.

    The question's options are the label column's distinct non-empty values,
    in first-appearance order (not sorted). A row with an empty label or an
    empty text cell is skipped and counted (``empty_label``/``empty_text``);
    everything else routes through the same checks ``items_from_jsonl`` uses.
    """
    if instructions is None:
        instructions = f"choose the {label_column} that best matches the text"

    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = []
    seen = set()
    for row in rows:
        label = (row.get(label_column) or "").strip()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)

    question = {"type": "choice", "instructions": instructions, "criteria": {label: "" for label in labels}}

    items: list[Item] = []
    skipped: Counter = Counter()

    for row in rows:
        label = (row.get(label_column) or "").strip()
        if not label:
            skipped["empty_label"] += 1
            continue
        text = row.get(text_column)
        item = _build_item(text, question_id, question, label, None, smoothing, skipped)
        if item is not None:
            items.append(item)

    return items, question, skipped


# Commit SHAs on each repo's auto-generated ``refs/convert/parquet`` branch, resolved
# once and pinned here. That branch is mutable — the Hub can regenerate it — and this
# project publishes benchmark numbers against this data, so a fetch has to name an
# immutable commit rather than float with whatever the branch currently points at.
_PARQUET_REVISIONS = {
    "LocalLLaMA/typed-decisions": "9529ecb7b0e283199a7889022761ede040d0e636",
    "AmazonScience/massive": "ed58ac423a2f4121720918bf5301577edce4ffd3",
    "google/boolq": "d89fc6ef4da46f056176dcf941ba670524ab1c87",
    "fancyzhx/ag_news": "4efd5262b756aee7a12710ca78c74aa258660063",
}


def _hf_parquet_split(repo_id: str, config: str, split: str):
    """Load one split's auto-converted parquet shard, without preparing its siblings.

    ``datasets.load_dataset(repo_id, config, split=...)`` still downloads and
    builds every split of the config on first use. Pointing the ``parquet``
    builder directly at a pinned commit on the Hub's parquet mirror for just
    the requested split avoids paying for the splits nobody asked for, and
    keeps the fetch reproducible. The split is globbed (``*.parquet``), not a
    literal ``0000.parquet``, so a split that grows past one shard is still
    read in full instead of silently truncated.
    """
    from datasets import load_dataset

    revision = _PARQUET_REVISIONS[repo_id]
    path = f"hf://datasets/{repo_id}@{revision}/{config}/{split}/*.parquet"
    return load_dataset("parquet", data_files=path, split="train")


def sample_options(labels: list[str], gold: str, n: int, rng: random.Random) -> list[str]:
    """Gold plus up to ``n - 1`` distractors drawn without replacement, then shuffled.

    Shuffling means an option's position never carries information about
    whether it is correct. If fewer than ``n`` labels exist, every label is
    returned.
    """
    pool = [label for label in labels if label != gold]
    options = [gold] + rng.sample(pool, min(n - 1, len(pool)))
    rng.shuffle(options)
    return options


def load_typed_decisions(split: str) -> list[dict]:
    """Load LocalLLaMA/typed-decisions: hand-labelled cases with gold belief distributions.

    This is the project's only source of probability-distribution targets —
    everything else loaded here is a hard label and gets smoothed instead.
    """
    ds = _hf_parquet_split("LocalLLaMA/typed-decisions", "all", split)
    return [
        {
            "state": json.loads(row["state"]),
            "questions": json.loads(row["questions"]),
            "gold": json.loads(row["gold"]),
        }
        for row in ds
    ]


def load_massive(split: str, n_options: int = 20, seed: int = 0) -> list[dict]:
    """Load MASSIVE (en-US) at the reference implementation's 20-option protocol.

    MASSIVE's full intent set is 60 labels: more than the published
    protocol uses, and more than the encoder's 256-token head budget can
    fit. Each case keeps the gold intent plus ``n_options - 1`` sampled
    distractors, shuffled so position carries no information about
    correctness, seeded by ``seed``.
    """
    ds = _hf_parquet_split("AmazonScience/massive", "en-US", split)
    intents = ds.features["intent"].names
    rng = random.Random(seed)
    qid = "intent"
    cases = []
    for row in ds:
        gold_label = intents[row["intent"]]
        options = sample_options(intents, gold_label, n_options, rng)
        question = {
            "type": "choice",
            "instructions": "Which intent best matches the utterance?",
            "criteria": {label: "" for label in options},
        }
        cases.append({"state": row["utt"], "questions": {qid: question}, "expected": {qid: gold_label}})
    return cases


def load_boolq(split: str) -> list[dict]:
    """Load BoolQ as a yes/no (``noul``) question over a passage and a question."""
    ds = _hf_parquet_split("google/boolq", "default", split)
    qid = "answer"
    question = {"type": "noul", "instructions": "Does the passage support a yes answer to the question?"}
    cases = []
    for row in ds:
        state = f"{row['passage']}\n\nQuestion: {row['question']}"
        cases.append({"state": state, "questions": {qid: question}, "expected": {qid: bool(row["answer"])}})
    return cases


def load_ag_news(split: str) -> list[dict]:
    """Load AG News as a 4-way topic-choice question."""
    ds = _hf_parquet_split("fancyzhx/ag_news", "default", split)
    names = ds.features["label"].names
    qid = "topic"
    question = {
        "type": "choice",
        "instructions": "Which topic does this news text belong to?",
        "criteria": {name: "" for name in names},
    }
    cases = []
    for row in ds:
        gold_label = names[row["label"]]
        cases.append({"state": row["text"], "questions": {qid: question}, "expected": {qid: gold_label}})
    return cases


def update_budget(n_items: int, micro_batch: int, grad_accum: int, epochs: int) -> int:
    """Total optimizer updates over training: ``epochs`` times the per-epoch update count.

    Each epoch's items are grouped into micro-batches, which are themselves
    grouped into optimizer updates by gradient accumulation; both groupings
    round up, so a partial trailing group still counts as one more step.
    """
    steps_per_epoch = -(-n_items // micro_batch)
    updates_per_epoch = -(-steps_per_epoch // grad_accum)
    return epochs * updates_per_epoch


def split_items(items: list[Item], calib_frac: float = 0.1, seed: int = 0) -> tuple[list[Item], list[Item]]:
    """Split items into a training set and a disjoint calibration set.

    The split is deterministic for a given ``seed``: the same items and seed
    always produce the same two lists, partitioned by a shuffled index set
    rather than copied, so each returned ``Item`` is the same object.
    """
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_calib = round(len(items) * calib_frac)
    calib_idx = set(order[:n_calib])

    train = [item for i, item in enumerate(items) if i not in calib_idx]
    calib = [item for i, item in enumerate(items) if i in calib_idx]
    return train, calib
