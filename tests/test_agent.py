import json

import pytest
import torch

import openjev
import openjev.encode
from openjev import load
from openjev.train import TrainConfig, save_checkpoint
from tests.conftest import CHOICE, NOUL, SCORE


def test_three_questions_answered_in_one_forward_pass(tiny_agent):
    out = tiny_agent.predict("billed twice", {"department": CHOICE, "urgency": SCORE, "churn_risk": NOUL})
    assert set(out["answers"]) == {"department", "urgency", "churn_risk"}
    assert out["usage"]["forward_passes"] == 1
    assert out["usage"]["questions"] == 3


def test_each_type_reports_its_own_key(tiny_agent):
    out = tiny_agent.predict("s", {"d": CHOICE, "u": SCORE, "c": NOUL})
    assert out["answers"]["d"]["choice"] in CHOICE["criteria"]
    assert 0 <= out["answers"]["u"]["score"] <= 2
    assert 0.0 <= out["answers"]["c"]["noul"] <= 1.0


def test_probabilities_are_keyed_by_label_and_sum_to_one(tiny_agent):
    p = tiny_agent.predict("s", {"d": CHOICE})["answers"]["d"]["probabilities"]
    assert set(p) == set(CHOICE["criteria"])
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-5)


def test_truncation_is_surfaced_in_usage(tiny_agent):
    out = tiny_agent.predict("word " * 4000, {"d": CHOICE})
    assert out["usage"]["truncated"] is True
    assert out["usage"]["state_tokens_used"] < out["usage"]["state_tokens"]


def test_empty_state_is_rejected(tiny_agent):
    with pytest.raises(ValueError, match="state"):
        tiny_agent.predict("   ", {"d": CHOICE})


def test_state_is_tokenized_once_for_all_questions(tiny_agent, monkeypatch):
    seen = []
    real = openjev.encode.build_sequence
    monkeypatch.setattr(openjev.encode, "build_sequence",
                        lambda *a, **kw: (seen.append(kw.get("state_ids")), real(*a, **kw))[1])
    tiny_agent.predict("shared state", {"a": CHOICE, "b": CHOICE, "c": NOUL})
    assert len(seen) == 3
    assert all(s is not None and s is seen[0] for s in seen)


def test_temperature_from_the_checkpoint_is_applied(tmp_path, tiny_model, tok):
    save_checkpoint(tiny_model, tok, tmp_path / "cold")
    tiny_model.temperature.copy_(torch.tensor([6.0, 6.0, 6.0]))
    save_checkpoint(tiny_model, tok, tmp_path / "warm")
    cold = load(tmp_path / "cold").predict("s", {"d": CHOICE})["answers"]["d"]
    warm = load(tmp_path / "warm").predict("s", {"d": CHOICE})["answers"]["d"]
    assert warm["answer_confidence"] < cold["answer_confidence"]
    assert warm["choice"] == cold["choice"]


def test_checkpoint_is_served_at_the_budget_it_was_trained_at(tiny_model, tok, tmp_path):
    """A checkpoint trained at a non-default token budget must be served at it.

    The spec contemplates ``head_max_len=512`` for MASSIVE's 60-label setting;
    recording the module default and then serving at the module default would
    publish a number measured at a budget the weights never saw.
    """
    save_checkpoint(tiny_model, tok, tmp_path, cfg=TrainConfig(max_len=300, head_max_len=64))
    config = json.loads((tmp_path / "rl_agent_config.json").read_text())
    assert (config["max_len"], config["head_max_len"]) == (300, 64)

    usage = load(tmp_path).predict("word " * 500, {"d": CHOICE})["usage"]
    assert usage["state_tokens"] >= 500  # the state itself is far longer than the budget
    assert usage["state_tokens_used"] < 300, "served at the module default, not the checkpoint's budget"
