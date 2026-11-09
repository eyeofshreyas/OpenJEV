"""The same case, through both engines.

Part 1 showed why a generated token's probability isn't a decision probability.
``common.decide`` fixes the scoring (a trie over one prefill, normalised over the
allowed answers, every option scored). That's still a prompted causal LM reading
a frozen 0.6B backbone - the thing this project's success criterion says we have
to beat. So it has to see what our model sees: the baseline's ``SCHEMA`` now
carries per-option descriptions (``common.py``'s ``parallel_prompt``), and the
questions we hand ``Agent.predict`` below carry the exact same text, so neither
engine is answering with less information than the other.

Under ``TINY=1`` (part 1's convention: a randomly initialised tiny config, no
0.6B download) both engines are untrained, and the output says so - a demo that
prints a uniform distribution for an obviously fraudulent case without saying
why would be exactly the kind of unearned-looking number this project argues
against. Point ``OPENJEV_CHECKPOINT`` at a trained checkpoint to get the real
comparison instead; the caveat below adjusts itself to whichever is true.
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


def build_agent(tiny, checkpoint_path):
    if tiny:
        # Same smoke path as common.load(): a randomly initialised tiny config, no download.
        tok = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)
        torch.manual_seed(0)
        model = build_model(config=tiny_config(vocab_size=len(tok))).eval()
        agent = openjev.Agent.__new__(openjev.Agent)
        agent.tok, agent.model, agent.device = tok, model, torch.device("cpu")
        return agent
    if checkpoint_path:
        return openjev.load(checkpoint_path)
    return openjev.Agent()  # fresh ModernBERT-base backbone, decision head still untrained


def print_caveat(tiny, checkpoint_path):
    """Say, out loud and unmissably, what the probabilities above do and don't mean.

    A reader runs this script; they don't read the task report. Printing a
    uniform distribution for an obviously fraudulent case with no caveat reads
    as "the model has no opinion" - the project's own anti-goal, pointed at
    itself. So this is conditional on what's actually running, not a canned
    disclaimer that would go stale (and start lying) the day a real checkpoint
    shows up.
    """
    print("\n--- what these numbers do (and don't) mean ---")
    if tiny:
        print("  Both engines are running untrained weights under TINY=1; no checkpoint exists")
        print("  yet (that is milestone 5). The probabilities above demonstrate the mechanism")
        print("  and the forward-pass counts, and say nothing about accuracy.")
        print("  openjev's output is *exactly* uniform (0.500, 0.333) rather than merely close,")
        print("  because every option marker is the same [MASK] token and an untrained model's")
        print("  per-marker hidden states differ by only ~1.4e-6. That exactness is evidence the")
        print("  read-out mechanism works as designed, not evidence that it fails.")
    elif checkpoint_path is None:
        print("  The baseline is running real pretrained Qwen3-0.6B weights, zero-shot and not")
        print("  fine-tuned on this task. openjev has no checkpoint (OPENJEV_CHECKPOINT is")
        print("  unset), so its decision head is untrained. Neither distribution reflects task")
        print("  accuracy yet - this still only demonstrates the mechanism and the pass counts.")
    else:
        print("  openjev is running the trained checkpoint at OPENJEV_CHECKPOINT; the baseline")
        print("  is the zero-shot prompted Qwen3-0.6B from the spec. This is the real comparison.")


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
tiny = bool(os.environ.get("TINY"))
checkpoint_path = os.environ.get("OPENJEV_CHECKPOINT")
agent = build_agent(tiny, checkpoint_path)
out = agent.predict(common.CONTEXT, to_questions(common.SCHEMA))
for qid, answer in out["answers"].items():
    rendered = "  ".join(f"{label}={p:.3f}" for label, p in answer["probabilities"].items())
    print(f"  {qid:<9} {rendered}")
usage = out["usage"]
print(f"  forward_passes={usage['forward_passes']}  ms={usage['ms']:.0f}")

print_caveat(tiny, checkpoint_path)

print("\n--- the claim, watched rather than read ---")
print(f"  baseline : forward_passes={baseline.forward_passes}  latency_ms={baseline_ms:.0f}")
print(f"  openjev  : forward_passes={usage['forward_passes']}  latency_ms={usage['ms']:.0f}")
