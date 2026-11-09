import torch
import pytest
from transformers import AutoTokenizer

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
