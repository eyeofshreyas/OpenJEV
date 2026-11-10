import torch

from openjev.encode import build_sequence, collate
from openjev.model import probabilities, sync
from tests.conftest import _fake_batch


def test_logits_shape_and_dtype(tiny_model):
    logits = tiny_model(**_fake_batch(b=2, length=32, k=3))
    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32


def test_probabilities_sum_to_one_over_real_options(tiny_model):
    batch = _fake_batch(b=2, length=32, k=3, counts=[2, 3])
    p = probabilities(tiny_model(**batch), batch["marker_mask"])
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)


def test_masked_options_get_zero_probability(tiny_model):
    batch = _fake_batch(b=1, length=32, k=4, counts=[2])
    p = probabilities(tiny_model(**batch), batch["marker_mask"])
    assert p[0, 2:].max().item() < 1e-6


def test_temperature_softens_without_changing_argmax(tiny_model):
    batch = _fake_batch(b=1, length=32, k=3)
    logits = tiny_model(**batch)
    cold = probabilities(logits, batch["marker_mask"])
    warm = probabilities(logits, batch["marker_mask"], qtype=batch["qtype"],
                         temperature=torch.tensor([5.0, 5.0, 5.0]))
    assert cold.argmax(-1).tolist() == warm.argmax(-1).tolist()
    assert warm.max().item() < cold.max().item()


def test_question_type_changes_the_logits(tiny_model):
    batch = _fake_batch(b=1, length=32, k=3)
    a = tiny_model(**batch)
    batch["qtype"] = batch["qtype"] + 1
    assert not torch.allclose(a, tiny_model(**batch))


def test_gradients_reach_encoder_and_scorer(tiny_model):
    batch = _fake_batch(b=2, length=32, k=3)
    tiny_model.train()
    tiny_model(**batch).sum().backward()
    assert tiny_model.scorer[-1].weight.grad is not None
    assert next(tiny_model.encoder.parameters()).grad is not None
    tiny_model.eval()


def test_identical_options_spread_is_measured(tiny_model, tok, capsys):
    # The spec defers the parallel option layout to v2 "with a measurement attached".
    # This is that measurement: five identical options, so any spread is position alone.
    q = {"type": "choice", "instructions": "pick one",
         "criteria": {f"opt{i}": "exactly the same description" for i in range(5)}}
    enc = build_sequence(tok, "a state", q)
    logits = tiny_model(**collate([enc], tok.pad_token_id))
    spread = (logits.max() - logits.min()).item()
    print(f"identical-option logit spread: {spread:.3f}")
    assert spread >= 0.0  # recorded, not gated: laya measured 3.59 on a trained checkpoint


def test_sync_waits_only_on_cuda(monkeypatch):
    """Latency timing brackets a forward call, which on CUDA returns at kernel-launch time.

    Without this the published latency figures would read roughly an order of
    magnitude too fast on the one device the spec asks them to be measured on.
    """
    waited = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: waited.append(device))

    sync(torch.device("cpu"))
    assert waited == []

    cuda = torch.device("cuda")
    sync(cuda)
    assert waited == [cuda]
