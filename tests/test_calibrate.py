import pytest
import torch
from safetensors.torch import load_file, save_file

import openjev.train
from openjev.encode import QTYPES
from openjev.train import TrainConfig, fit_temperatures, load_checkpoint, save_checkpoint

from tests.conftest import CHOICE, _accuracy, _fake_batch, _miscalibrated_items, _synthetic_items


def test_temperature_reduces_nll_of_an_overconfident_model(tiny_model, tok):
    original_weight = tiny_model.scorer[-1].weight.clone()
    items, model = _miscalibrated_items(n=200, model=tiny_model, tok=tok)
    temps, notes = fit_temperatures(model, items, tok, TrainConfig())
    assert temps[QTYPES["choice"]] > 1.0
    # Strictly inside the search bounds: a fitter that just always returned the
    # grid's own ceiling would otherwise pass this test identically to a real fit.
    assert 0.25 < temps[QTYPES["choice"]] < 10.0
    assert notes["choice"]["nll_after"] < notes["choice"]["nll_before"]
    # The fixture must not have mutated the shared, module-scoped fixture model.
    assert torch.equal(tiny_model.scorer[-1].weight, original_weight)


def test_temperature_never_changes_accuracy(tiny_model, tok):
    items, model = _miscalibrated_items(n=200, model=tiny_model, tok=tok)
    before = _accuracy(model, items, tok, temperature=None)
    temps, _ = fit_temperatures(model, items, tok, TrainConfig())
    assert _accuracy(model, items, tok, temperature=temps) == before


def test_too_few_items_leaves_temperature_unfitted(tiny_model, tok):
    # The five items are ``noul``, so ``noul`` is the type that tests the
    # 5 < MIN_CALIBRATION_ITEMS boundary; ``choice`` has no items at all here.
    temps, notes = fit_temperatures(tiny_model, _synthetic_items(n=5), tok, TrainConfig())
    assert temps[QTYPES["noul"]] == 1.0
    assert notes["noul"]["fitted"] is False
    assert notes["noul"]["n_items"] == 5


def test_calibration_forwards_in_micro_batches(tiny_model, tok, monkeypatch):
    """A real calibration slice is thousands of sequences: one forward over all of them OOMs.

    And it would OOM *after* training finished, before any publishable number
    exists, since this is the only path that fits the temperature.
    """
    sizes = []
    real = openjev.train._batch
    monkeypatch.setattr(
        openjev.train,
        "_batch",
        lambda tok_, items_, *rest: (sizes.append(len(items_)), real(tok_, items_, *rest))[1],
    )
    fit_temperatures(tiny_model, _synthetic_items(n=20), tok, TrainConfig(micro_batch=4))
    assert sizes, "no batching happened at all"
    assert max(sizes) <= 4


def test_checkpoint_round_trip_preserves_logits(tiny_model, tok, tmp_path):
    batch = _fake_batch(b=2, length=32, k=3)
    before = tiny_model(**batch)
    save_checkpoint(tiny_model, tok, tmp_path, questions={"q": CHOICE})
    model2, tok2, cfg = load_checkpoint(tmp_path)
    assert torch.allclose(before, model2(**batch), atol=1e-5)


def test_checkpoint_layout_uses_layas_filenames(tiny_model, tok, tmp_path):
    save_checkpoint(tiny_model, tok, tmp_path, questions={"q": CHOICE})
    for name in ["model.safetensors", "rl_agent_config.json", "questions.json"]:
        assert (tmp_path / name).exists()
    assert (tmp_path / "encoder").is_dir() and (tmp_path / "tokenizer").is_dir()


def test_saved_temperature_survives_the_round_trip(tiny_model, tok, tmp_path):
    tiny_model.temperature.copy_(torch.tensor([1.5, 2.0, 0.8]))
    save_checkpoint(tiny_model, tok, tmp_path)
    model2, _, _ = load_checkpoint(tmp_path)
    assert torch.allclose(model2.temperature, torch.tensor([1.5, 2.0, 0.8]))


def test_a_checkpoint_missing_head_weights_is_refused(tiny_model, tok, tmp_path):
    """A truncated model.safetensors must not load a randomly initialised head.

    ``strict=False`` is there because every ``encoder.*`` key legitimately
    lives in the encoder directory instead. Any *other* missing key means the
    model serves confident garbage while calling itself a trained checkpoint.
    """
    save_checkpoint(tiny_model, tok, tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    truncated = {k: v for k, v in state.items() if not k.startswith("scorer.")}
    assert len(truncated) < len(state), "the fixture dropped nothing"
    save_file(truncated, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="scorer"):
        load_checkpoint(tmp_path)
