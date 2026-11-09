import math
import random

import torch
import pytest

from openjev.data import Item, update_budget
from openjev.train import NonFiniteLoss, TrainConfig, autocast_dtype, shuffle_options, tag_state, train

from tests.conftest import CHOICE, SCORE, _synthetic_items


def test_autocast_dtype_is_fp16_below_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    assert autocast_dtype(torch.device("cuda")) == torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 6))
    assert autocast_dtype(torch.device("cuda")) == torch.bfloat16


def test_cpu_gets_no_autocast():
    assert autocast_dtype(torch.device("cpu")) is None


def test_tag_state_prepends_the_answer():
    item = Item(state="s", qid="q", question=CHOICE, target=[0.0, 1.0, 0.0])
    assert "b" in tag_state(item).state and tag_state(item).target == item.target


# The config's default learning rates (2e-5 / 1e-4) are sized for a pretrained
# ModernBERT-base. The tiny test model is randomly initialized and 64-dimensional,
# and AdamW's step size is set by the learning rate, so at 1e-4 it needs thousands
# of updates to move at all (measured: 540 updates, still at the uniform plateau).
# These two tests raise it so that a *working* loop can clear the bar they assert
# within a few dozen updates; the bars themselves are the brief's.
TINY_LR = dict(lr_encoder=1e-3, lr_head=1e-3)


def test_positive_control_is_learned(tiny_model, tok, tmp_path):
    items = [tag_state(i) for i in _synthetic_items(n=40)]
    report = train(items, tiny_model, tok, out_dir=tmp_path,
                   cfg=TrainConfig(epochs=8, micro_batch=4, grad_accum=1, **TINY_LR))
    assert report["train_accuracy"] > 0.95


def test_overfits_a_small_set(tiny_model, tok, tmp_path):
    report = train(_synthetic_items(n=60), tiny_model, tok, out_dir=tmp_path,
                   cfg=TrainConfig(epochs=30, micro_batch=4, grad_accum=1, **TINY_LR))
    assert report["train_accuracy"] > 0.9


def test_choice_questions_train_with_permuted_options(tiny_model, tok, tmp_path):
    """The per-epoch re-encode of a shuffled ``choice`` question has to survive a run."""
    items = [
        Item(state=f"report {i}", qid="d", question=CHOICE, target=[0.1, 0.8, 0.1])
        for i in range(8)
    ]
    report = train(items, tiny_model, tok, out_dir=tmp_path, cfg=TrainConfig(epochs=2, micro_batch=4))
    assert len(report["epoch_loss"]) == 2 and all(math.isfinite(x) for x in report["epoch_loss"])


def test_report_states_the_optimizer_budget(tiny_model, tok, tmp_path):
    report = train(_synthetic_items(n=40), tiny_model, tok, out_dir=tmp_path,
                   cfg=TrainConfig(epochs=2, micro_batch=4, grad_accum=2))
    assert report["optimizer_updates"] == update_budget(report["train_items"], 4, 2, 2)


def test_training_refuses_an_empty_training_set(tiny_model, tok, tmp_path):
    with pytest.raises(ValueError, match="no training items"):
        train([], tiny_model, tok, out_dir=tmp_path, cfg=TrainConfig(epochs=1, micro_batch=4))


def test_non_finite_loss_aborts(tiny_model, tok, monkeypatch, tmp_path):
    monkeypatch.setattr("openjev.train.decision_loss", lambda *a, **k: torch.tensor(float("nan")))
    with pytest.raises(NonFiniteLoss):
        train(_synthetic_items(n=8), tiny_model, tok, out_dir=tmp_path, cfg=TrainConfig(epochs=1, micro_batch=4))


def test_rolling_checkpoint_written_each_epoch(tiny_model, tok, tmp_path):
    train(_synthetic_items(n=8), tiny_model, tok, out_dir=tmp_path, cfg=TrainConfig(epochs=2, micro_batch=4))
    assert (tmp_path / "checkpoint_latest" / "model.safetensors").exists()


def test_shuffle_options_keeps_the_answer_aligned(tok):
    item = Item(state="s", qid="d", question=CHOICE, target=[0.0, 1.0, 0.0])
    for seed in range(5):
        shuffled, target = shuffle_options(item, random.Random(seed))
        labels = list(shuffled.question["criteria"])
        assert labels[target.index(max(target))] == "b"
        assert sorted(labels) == ["a", "b", "c"]


def test_shuffle_options_leaves_score_levels_alone(tok):
    item = Item(state="s", qid="u", question=SCORE, target=[0.0, 0.0, 1.0])
    shuffled, target = shuffle_options(item, random.Random(3))
    assert list(shuffled.question["criteria"]) == list(SCORE["criteria"])
    assert target == item.target
