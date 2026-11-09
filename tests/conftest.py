import torch
import pytest
from transformers import AutoTokenizer

from openjev.data import Item, target_vector
from openjev.model import build_model, tiny_config

CHOICE = {"type": "choice", "instructions": "i", "criteria": {"a": "", "b": "", "c": ""}}
SCORE = {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}
NOUL = {"type": "noul", "instructions": "i"}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


@pytest.fixture(scope="module")
def tiny_model(tok):
    return build_model(config=tiny_config(vocab_size=len(tok))).eval()


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
