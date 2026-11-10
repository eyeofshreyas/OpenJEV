import math
import random
from dataclasses import replace

import torch
import pytest

from openjev.data import Item, target_vector, update_budget
from openjev.model import build_model, tiny_config
from openjev.score import decision_loss
from openjev.train import (
    NonFiniteLoss,
    TrainConfig,
    _batch,
    autocast_dtype,
    fit_temperatures,
    load_checkpoint,
    save_checkpoint,
    shuffle_options,
    tag_state,
    train,
)

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


def test_permuted_choice_batch_keeps_the_target_on_its_own_option(tok):
    """The target must still sit on its option after the whole encode path, not just after the permutation.

    Read back from the tensors the model is actually handed: decode the option
    behind each marker, and the target's argmax has to land on ``b`` whichever
    slot ``b`` was permuted into. This covers the chain the unit test above
    cannot — ``shuffle_options`` into ``build_sequence`` into ``collate`` into
    the padded target matrix — and it is the check that catches a mis-permuted
    target, which the learning test below cannot (see its docstring).
    """
    item = Item(state="the correct option is b", qid="d", question=CHOICE,
                target=target_vector(CHOICE, expected="b"))
    rng = random.Random(0)
    slots = set()
    for _ in range(8):
        batch, target = _batch(tok, [item], TrainConfig(), rng, {})
        options = [tok.decode(batch["input_ids"][0, pos + 1]).strip() for pos in batch["marker_pos"][0]]
        assert sorted(options) == ["a", "b", "c"]
        assert options[int(target[0].argmax())] == "b"
        slots.add(options.index("b"))
    assert len(slots) > 1, "the options were never actually permuted"


def test_choice_questions_learn_under_permuted_options(tok, tmp_path):
    """Training a ``choice`` question whose options move every epoch has to get somewhere.

    Not a misalignment detector, and deliberately not claimed as one: measured,
    a deliberately mis-permuted target reaches the same loss (-0.0567 against
    this test's -0.0568), because leaving the target on a fixed index just
    teaches the model that index instead. What this does pin is that the
    per-epoch re-encode path can still learn — a run that cannot get off the
    uniform plateau here has lost the signal somewhere between the permutation
    and the loss.
    """
    torch.manual_seed(0)
    model = build_model(config=tiny_config(vocab_size=len(tok)))
    items = [
        Item(state=f"report {i}: the correct option is b", qid="d", question=CHOICE,
             target=target_vector(CHOICE, expected="b"))
        for i in range(2)
    ]
    report = train(items, model, tok, out_dir=tmp_path,
                   cfg=TrainConfig(epochs=80, micro_batch=2, grad_accum=1, calib_frac=0.0, **TINY_LR))

    target = torch.tensor([item.target for item in items])
    plateau = float(decision_loss(torch.zeros(2, 3), target, torch.zeros(2, dtype=torch.long),
                                  torch.ones(2, 3, dtype=torch.bool)))
    assert all(math.isfinite(x) for x in report["epoch_loss"])
    assert min(report["epoch_loss"]) < plateau - 0.2, f"parked at the K=3 plateau ({plateau:.4f})"


def test_trailing_accumulation_group_is_weighted_by_its_own_size(tok, tmp_path, monkeypatch):
    """An epoch's short last group must step as hard as a full one.

    12 items at ``micro_batch=4`` is 3 micro-batches, which ``grad_accum=2``
    splits into a group of 2 and a trailing group of 1. The items are identical
    and both learning rates are 0, so the weights never move and every
    micro-batch produces the same gradient: both updates must then present the
    same gradient to the clip. Dividing the trailing group by ``grad_accum``
    instead of by its own size halves the second one.
    """
    model = build_model(config=tiny_config(vocab_size=len(tok)))
    for module in model.modules():  # dropout would make the two gradients differ by noise
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0

    clipped = []
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda params, clip: clipped.append(float(clip_grad_norm_(params, clip))),
    )

    train(_synthetic_items(n=1) * 12, model, tok, out_dir=tmp_path,
          cfg=TrainConfig(epochs=1, micro_batch=4, grad_accum=2, lr_encoder=0.0, lr_head=0.0, calib_frac=0.0))

    assert len(clipped) == 2
    assert clipped[1] == pytest.approx(clipped[0], rel=1e-3)


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


def test_rolling_checkpoint_carries_resume_state(tiny_model, tok, tmp_path):
    """Weights alone would make a restart a new run from a warm start, not a resume."""
    train(_synthetic_items(n=8), tiny_model, tok, out_dir=tmp_path, cfg=TrainConfig(epochs=2, micro_batch=4))
    state = torch.load(tmp_path / "checkpoint_latest" / "trainer_state.pt", weights_only=True)
    assert state["epoch"] == 2 and state["updates"] == 2 and len(state["epoch_loss"]) == 2
    assert state["optimizer"]["state"], "Adam moments must come back warm"
    assert state["scheduler"]["last_epoch"] == 2
    # A disabled ``GradScaler`` has an empty state dict by design, so on CPU this only
    # pins that the slot is written; the loss scale itself rides in it on the fp16 path.
    assert state["scaler"] == {}


def test_resume_continues_the_schedule_instead_of_restarting(tok, tmp_path):
    model = build_model(config=tiny_config(vocab_size=len(tok)))
    items = _synthetic_items(n=8)
    cfg = TrainConfig(epochs=4, micro_batch=4, grad_accum=1)

    interrupted = train(items, model, tok, out_dir=tmp_path, cfg=replace(cfg, epochs=2))
    resumed = train(items, model, tok, out_dir=tmp_path, cfg=cfg, resume=True)

    budget = update_budget(resumed["train_items"], 4, 1, 4)
    assert interrupted["optimizer_updates"] == budget // 2
    # The updates already spent are counted, the loss history is continuous, and the
    # cosine schedule picks up mid-flight: a restart would sit at update 4 of 8, in warmup.
    assert resumed["optimizer_updates"] == budget
    assert len(resumed["epoch_loss"]) == 4
    state = torch.load(tmp_path / "checkpoint_latest" / "trainer_state.pt", weights_only=True)
    assert state["scheduler"]["last_epoch"] == budget and state["epoch"] == 4


@pytest.mark.network
def test_checkpoint_round_trips_the_real_backbone(tmp_path):
    """``save_model`` raises on shared tensors, and the real backbone is the one that counts.

    The tiny test config cannot answer this: tied or shared weights are a
    property of ModernBERT-base. Finding out at the end of the first real epoch
    would cost the GPU hours already spent.
    """
    model = build_model()
    save_checkpoint(model, tmp_path, state={"epoch": 1, "updates": 7})
    assert (tmp_path / "checkpoint_latest" / "model.safetensors").exists()
    assert load_checkpoint(model, tmp_path)["updates"] == 7


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


def test_report_hands_back_the_calibration_slice_itself(tiny_model, tok, tmp_path):
    """The fitter needs this exact slice; re-deriving it from the seed is drift waiting to happen."""
    items = _synthetic_items(n=40)
    report = train(items, tiny_model, tok, out_dir=tmp_path,
                   cfg=TrainConfig(epochs=1, micro_batch=4, calib_frac=0.25))

    calib = report["calibration_set"]
    assert len(calib) == report["calibration_items"] == 10
    assert all(any(c is i for i in items) for c in calib), "not the held-out items themselves"
    # And it feeds the fitter directly, which is the whole point of returning it.
    _, notes = fit_temperatures(tiny_model, calib, tok, TrainConfig())
    assert notes["noul"]["fitted"] is True


def test_gradient_checkpointing_still_trains(tok, tmp_path):
    """The spec's Kaggle recipe runs with checkpointing on, so the loop has to work with it on.

    Its ~12 GB/rank estimate assumes it: micro_batch=8 at max_len=1024 on
    ModernBERT-large does not otherwise fit a 16 GB T4.
    """
    model = build_model(config=tiny_config(vocab_size=len(tok)))
    before = model.scorer[-1].weight.clone()

    report = train(_synthetic_items(n=8), model, tok, out_dir=tmp_path,
                   cfg=TrainConfig(epochs=1, micro_batch=4, grad_accum=1,
                                   gradient_checkpointing=True, **TINY_LR))

    assert model.encoder.gradient_checkpointing is True
    assert all(math.isfinite(x) for x in report["epoch_loss"])
    # Gradients reached the head through the checkpointed graph, not just the encoder.
    assert not torch.equal(model.scorer[-1].weight, before)
