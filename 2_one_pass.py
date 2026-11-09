"""The same case, through both engines.

Part 1 showed why a generated token's probability isn't a decision probability.
``common.decide`` fixes the scoring (a trie over one prefill, normalised over the
allowed answers, every option scored). That's still a prompted causal LM reading
a frozen 0.6B backbone - the thing this project's success criterion says we have
to beat. So it has to see what our model sees: the baseline's ``SCHEMA`` now
carries per-option descriptions (``common.py``'s ``parallel_prompt``), and the
questions we hand ``Agent.predict`` below carry the exact same text, so neither
engine is answering with less information than the other.

Both engines are untrained here (``TINY=1`` builds a randomly initialised tiny
config for each, same convention as part 1 - no 0.6B download, no checkpoint to
load). Their distributions will be close to uniform; that's the smoke test
working, not a result. What this script proves is the plumbing and the pass
counts, side by side.
"""
import os
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoTokenizer

import common
import openjev
from openjev.model import DEFAULT_BACKBONE, build_model, tiny_config


def to_questions(schema):
    """The same fields, as the ``Agent`` sees them - same instructions, same per-option text."""
    questions = {}
    for name, field in schema.items():
        if field["type"] == "boolean":
            questions[name] = {"type": "noul", "instructions": field["description"]}
        else:
            descriptions = field.get("descriptions", {})
            criteria = {c: descriptions.get(c, "") for c in field["choices"]}
            questions[name] = {"type": "choice", "instructions": field["description"], "criteria": criteria}
    return questions


def build_agent():
    if os.environ.get("TINY"):
        # Same smoke path as common.load(): a randomly initialised tiny config, no download.
        tok = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)
        torch.manual_seed(0)
        model = build_model(config=tiny_config(vocab_size=len(tok))).eval()
        agent = openjev.Agent.__new__(openjev.Agent)
        agent.tok, agent.model, agent.device = tok, model, torch.device("cpu")
        return agent
    return openjev.Agent()  # fresh ModernBERT-base backbone; swap for openjev.load(path) once a checkpoint exists


print("CONTEXT:", common.CONTEXT)

print("\n--- engine 1: prompted Qwen3-0.6B, scored by common.decide's trie ---")
tok, model = common.load()
t0 = time.perf_counter()
baseline = common.decide(tok, model, common.CONTEXT, common.SCHEMA)
baseline_ms = (time.perf_counter() - t0) * 1000
for name, answer in baseline.answers.items():
    ranked = sorted(zip(answer.choices, answer.probs), key=lambda cp: -cp[1])
    rendered = "  ".join(f"{c}={p:.3f}" for c, p in ranked)
    print(f"  {name:<9} {rendered}")
print(f"  forward_passes={baseline.forward_passes}  nodes={baseline.nodes}  ms={baseline_ms:.0f}")

print("\n--- engine 2: openjev.Agent, one forward pass ---")
agent = build_agent()
out = agent.predict(common.CONTEXT, to_questions(common.SCHEMA))
for qid, answer in out["answers"].items():
    rendered = "  ".join(f"{label}={p:.3f}" for label, p in answer["probabilities"].items())
    print(f"  {qid:<9} {rendered}")
usage = out["usage"]
print(f"  forward_passes={usage['forward_passes']}  ms={usage['ms']:.0f}")

print("\n--- the claim, watched rather than read ---")
print(f"  baseline : forward_passes={baseline.forward_passes}  latency_ms={baseline_ms:.0f}")
print(f"  openjev  : forward_passes={usage['forward_passes']}  latency_ms={usage['ms']:.0f}")
