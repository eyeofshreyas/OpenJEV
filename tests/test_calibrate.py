import torch

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
    temps, notes = fit_temperatures(tiny_model, _synthetic_items(n=5), tok, TrainConfig())
    assert temps[QTYPES["choice"]] == 1.0
    assert notes["choice"]["fitted"] is False


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
