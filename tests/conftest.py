import copy
import random

import torch
import pytest
from transformers import AutoTokenizer

import openjev
import openjev.encode
from openjev.data import Item, target_vector
from openjev.model import build_model, probabilities, tiny_config
from openjev.train import TrainConfig, _batch

CHOICE = {"type": "choice", "instructions": "i", "criteria": {"a": "", "b": "", "c": ""}}
SCORE = {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}
NOUL = {"type": "noul", "instructions": "i"}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


@pytest.fixture(scope="module")
def tiny_model(tok):
    torch.manual_seed(0)  # deterministic weights: a test whose outcome depends on OS-entropy is worse than none
    return build_model(config=tiny_config(vocab_size=len(tok))).eval()


@pytest.fixture
def tiny_agent(tok, tiny_model):
    """An ``Agent`` wrapping the real tokenizer and a tiny model, so tests never download weights.

    Bypasses ``Agent.__init__`` (which only knows how to build a fresh backbone or
    load a saved checkpoint from disk) and wires the already-built fixtures in directly.
    """
    agent = openjev.Agent.__new__(openjev.Agent)
    agent.tok = tok
    agent.model = tiny_model
    agent.device = torch.device("cpu")
    agent.max_len = openjev.encode.MAX_LEN
    agent.head_max_len = openjev.encode.HEAD_MAX_LEN
    return agent


_SYNTHETIC = {"type": "noul", "instructions": "does the report say the statement holds?"}


def _synthetic_items(n):
    """A separable two-label task: the answer word is present in the state.

    Deliberately trivial — several tests need the tiny model to be *able* to
    learn this, so that a failure to learn points at the loop, not the task.

    The question is ``noul`` rather than ``choice`` on purpose. A ``choice``
    question's options are permuted every epoch (see ``train.shuffle_options``),
    so its answer can only be found by matching the option text against the
    state — attention a freshly initialized tiny encoder does not have and
    cannot acquire in a few hundred CPU steps (measured: it parks at the
    uniform plateau for 300 updates). A ``noul`` question's two options are
    fixed and ordered, so the task stays learnable.
    """
    items = []
    for i in range(n):
        holds = i % 2 == 0
        items.append(
            Item(
                state=f"report {i}: the statement is {'true' if holds else 'false'}",
                qid="holds",
                question=_SYNTHETIC,
                target=target_vector(_SYNTHETIC, expected=holds),
            )
        )
    return items


def _synthetic_cases(n):
    """``_synthetic_items``, wrapped in the public case shape ``evaluate`` consumes.

    One ``noul`` question per case, so ``n`` cases is also ``n`` evaluated items.
    """
    cases = []
    for i in range(n):
        holds = i % 2 == 0
        cases.append(
            {
                "state": f"report {i}: the statement is {'true' if holds else 'false'}",
                "questions": {"holds": _SYNTHETIC},
                "expected": {"holds": holds},
            }
        )
    return cases


def _score_cases(n):
    """``n`` cases, each a single ``score`` question cycling through its levels."""
    cases = []
    for i in range(n):
        level = i % len(SCORE["criteria"])
        cases.append(
            {
                "state": f"case {i}: level is {level}",
                "questions": {"q": SCORE},
                "expected": {"q": level},
            }
        )
    return cases


def _miscalibrated_items(n, model, tok):
    """Build a confidently-miscalibrated model and a fifth-wrong target set for it.

    A freshly-built, untrained marker-based model is no help here: every
    option's marker is the literal same ``[MASK]`` token, so its hidden state
    barely moves with surrounding content (measured across 8 random seeds and
    content from one-word hints to 300-token contexts: marker hidden states
    agree to within ~0.01-1% of each other regardless of what the state
    says). Its logits are then ~equal, its softmax ~uniform — an
    *under*confident model, not an overconfident one, and no amount of noisy
    training reliably flips that without a margin that has to be hand-tuned
    between "no effect" and "the model just memorizes the noise" (measured
    both failure modes directly).

    So instead of training a signal in, this works on its *own copy* of
    ``model`` (the caller's model is never mutated — a fixture that scales a
    shared, module-scoped model in place would make this file's tests order-
    and count-dependent) and scales that copy's scorer's final ``Linear`` by
    a constant large enough to dominate the ~1e-6 natural logit gap, turning
    it into one comfortably big enough to produce confident — but not
    saturated-against-the-fitter's-own-search-bound — probabilities.
    Pointing ~20% of each item's target at a *different* option (one the
    copy is not favouring) then makes the copy wrong exactly that often
    while still confident every time: confidently right most of the time,
    confidently wrong a fifth of the time, by construction rather than by
    hoping a training run lands in a narrow window.

    Returns ``(items, model)`` — the copy, not the original — since the
    caller needs the exact model the items were built against.
    """
    model = copy.deepcopy(model)
    # Chosen by measurement, not guesswork: scaled linearly from the ~1e-6 natural
    # gap, this lands the fitted temperature around 3-4, comfortably inside
    # [0.25, 10] rather than pinned to either end (a fit pinned to the upper
    # bound can't be told apart from a fitter that just always returns the max).
    with torch.no_grad():
        final = model.scorer[-1]  # the Linear(d, 1) that turns a marker's hidden state into its logit
        final.weight *= 5e6
        final.bias *= 5e6

    labels = list(CHOICE["criteria"])
    items = [
        Item(state=f"case {i}: filler text", qid="q", question=CHOICE, target=target_vector(CHOICE, expected="a"))
        for i in range(n)
    ]

    was_training = model.training
    model.eval()
    try:
        batch, _ = _batch(tok, items, TrainConfig(), None, {})
        with torch.no_grad():
            top = model(**batch).argmax(-1).tolist()
    finally:
        model.train(was_training)

    rng = random.Random(0)
    out = []
    for item, top_idx in zip(items, top):
        top_label = labels[top_idx]
        wrong = rng.random() < 0.2
        target_label = rng.choice([o for o in labels if o != top_label]) if wrong else top_label
        out.append(Item(state=item.state, qid=item.qid, question=CHOICE,
                         target=target_vector(CHOICE, expected=target_label)))
    return out, model


@torch.no_grad()
def _accuracy(model, items, tok, temperature=None):
    """The share of ``items`` whose top option matches the target's, optionally through ``temperature``.

    Dividing logits by a positive scalar never moves the argmax, so this is
    the check that pins ``fit_temperatures`` to that fact: accuracy computed
    with the fitted temperatures must equal accuracy computed without them.
    """
    was_training = model.training
    model.eval()
    cfg = TrainConfig()
    state_ids: dict[str, list[int]] = {}
    temperature_t = torch.tensor(temperature) if temperature is not None else None
    correct = 0
    try:
        for i in range(0, len(items), cfg.micro_batch):
            batch, target = _batch(tok, items[i : i + cfg.micro_batch], cfg, None, state_ids)
            logits = model(**batch)
            if temperature_t is not None:
                pred = probabilities(logits, batch["marker_mask"], qtype=batch["qtype"], temperature=temperature_t)
            else:
                pred = logits
            correct += int((pred.argmax(-1) == target.argmax(-1)).sum())
    finally:
        model.train(was_training)
    return correct / len(items)


def _fake_batch(b, length, k, counts=None):
    """A random batch shaped like ``collate``'s output, for model-only tests.

    ``counts`` (default: all ``k``) sets each row's real option count; the rest
    of each row's marker slots are masked out, pointing at position 0.
    """
    if counts is None:
        counts = [k] * b

    input_ids = torch.randint(0, 1000, (b, length))
    attention_mask = torch.ones(b, length, dtype=torch.long)
    marker_pos = torch.zeros(b, k, dtype=torch.long)
    marker_mask = torch.zeros(b, k, dtype=torch.bool)
    qtype = torch.zeros(b, dtype=torch.long)

    for i, n in enumerate(counts):
        marker_pos[i, :n] = torch.randperm(length)[:n]
        marker_mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }
