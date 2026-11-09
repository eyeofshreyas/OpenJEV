import torch
import pytest

from openjev.score import proper_reward, decision_loss
from openjev.model import build_model, tiny_config
from transformers import AutoTokenizer


def test_strictly_proper_truth_scores_best():
    truth = torch.tensor([[0.6, 0.3, 0.1]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    qtype = torch.zeros(1, dtype=torch.long)
    honest = proper_reward(truth, truth, qtype, mask).item()
    for other in [[0.34, 0.33, 0.33], [0.9, 0.05, 0.05], [0.1, 0.3, 0.6], [0.6, 0.1, 0.3]]:
        assert proper_reward(torch.tensor([other]), truth, qtype, mask).item() < honest


def test_rps_prefers_a_near_miss_for_score_questions():
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    near = torch.tensor([[0.0, 0.1, 0.8, 0.1]])
    far = torch.tensor([[0.1, 0.1, 0.8, 0.0]])
    score_t = torch.tensor([1]); choice_t = torch.tensor([0])
    assert proper_reward(near, target, score_t, mask) > proper_reward(far, target, score_t, mask)
    assert torch.allclose(proper_reward(near, target, choice_t, mask),
                          proper_reward(far, target, choice_t, mask), atol=1e-6)


def test_confident_and_wrong_is_floored_not_infinite():
    p = torch.tensor([[1e-30, 1.0 - 1e-30]])
    target = torch.tensor([[1.0, 0.0]])
    r = proper_reward(p, target, torch.zeros(1, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
    assert torch.isfinite(r).all() and r.item() >= -9.21 - 1.0


def test_masked_options_are_ignored():
    mask = torch.tensor([[True, True, False]])
    qtype = torch.zeros(1, dtype=torch.long)
    target = torch.tensor([[0.7, 0.3, 0.0]])
    a = proper_reward(torch.tensor([[0.7, 0.3, 0.0]]), target, qtype, mask)
    b = proper_reward(torch.tensor([[0.7, 0.3, 0.9]]), target, qtype, mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_loss_is_fp32_and_decreases_toward_the_target():
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool); qtype = torch.zeros(1, dtype=torch.long)
    first = decision_loss(logits, target, qtype, mask)
    assert first.dtype == torch.float32
    first.backward()
    better = decision_loss((logits - 0.5 * logits.grad).detach(), target, qtype, mask)
    assert better.item() < first.item()
