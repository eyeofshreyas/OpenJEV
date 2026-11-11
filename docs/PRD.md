# OpenJEV — Product Requirements Document

Status: **in development.** A first base-size checkpoint (milestone 5,
local run) is trained and published — see `docs/MODEL_CARD.md` for its
real numbers and limits. The large-size Kaggle run and the broader
training set (MASSIVE, BoolQ, AG News) are still ahead. This document
describes what OpenJEV is for and what "done" looks like, not a finished
product. See `docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md`
for the binding technical design and `docs/CONCEPTS.md` for the reasoning
behind it.

## 1. Problem

Software increasingly asks a language model to make small, high-volume
judgment calls — *which team handles this ticket, is this spam, how urgent
is this, does this look like fraud* — and then prompts a chat-style model
for JSON and reads a value out of it. That approach has four structural
problems, demonstrated in this repo's `1_vanilla_decoding.py`:

1. The model's token probability is normalized over its entire vocabulary,
   not over the handful of options you actually offered it.
2. Multi-token labels look less likely than short ones, for reasons of
   spelling, not belief.
3. Fields written later in the JSON are conditioned on fields written
   earlier, so the reported confidence depends on field order.
4. The alternatives the model didn't pick are never scored at all — there
   is no distribution, only the one path greedy decoding happened to walk.

The result is a number that *looks* like a probability but isn't one, and
can't safely be used to gate automation ("answer automatically above 0.9,
otherwise send to a human") or compared against anything.

## 2. Product vision

A small, open, from-scratch **decision model**: given a short text
(`state`) and a set of typed questions about it, return a calibrated
probability distribution over each question's allowed answers — in one
batched forward pass, with no text generated. This category was named
"System One models" by TypeSafe AI's Jev; the open reference implementation
is [laya](https://github.com/NandhaKishorM/laya). OpenJEV is our own
version: same idea, our own weights, small enough that one person can read
all of it.

This is **not** an attempt to beat Jev or laya on accuracy. The explicit
target is laya's base-size, raw-start specialist score (0.646 on their
typed-decisions benchmark), not their best fine-tune (0.766) or Jev's
published number (0.727) — see §5 for why that bar was chosen honestly
rather than aspirationally.

## 3. Target users

- Developers building routing/triage/moderation features who currently
  coerce a chat LLM into returning structured judgments and want a number
  they can actually threshold on.
- Anyone evaluating whether a decision model is worth adopting, who wants
  to see the failure mode of the prompt-and-decode approach next to the
  calibrated alternative, side by side, on the same input.
- Contributors extending this project's training data, backbone, or
  question types.

## 4. Scope

### In scope (v1)

- Three question types: `choice` (named options), `score` (ordered levels,
  partial credit for near misses), `noul` (yes/no).
- Any number of questions about one state, answered in a single batched
  forward pass, with the state tokenized once and reused.
- Options and their descriptions supplied by the caller at inference time —
  no fixed label set baked into the weights.
- Calibrated probabilities: a strictly proper scoring rule as the training
  objective, plus post-hoc per-question-type temperature scaling, with
  calibration measured honestly (ECE, Brier) on a held-out test slice.
- A baseline (`common.py`) that answers the same questions via a frozen
  causal LM and a trie scorer, kept apart as the thing being compared
  against, not replaced.
- A trainable pipeline (`openjev.train`, `openjev.eval`) usable today as
  library functions, driven by a script rather than a one-command CLI.
- A `playground/` web app (FastAPI + React) to try `Agent.predict`
  interactively and compare it against the baseline.

### Out of scope (v1 — deliberate, not forgotten)

- An MCP server, LangChain/LlamaIndex integrations, or a CLI beyond the
  training entry point.
- ONNX export, quantization, or any non-PyTorch runtime.
- Multilingual support (v1 is English-only; a multilingual backbone swap is
  a later, separate decision).
- A router across multiple checkpoints.
- Non-text modalities (images, audio, video) or very long documents.
- Conformal prediction sets, abstain/route flags, out-of-distribution
  detection.
- Shortlisting/tournament selection for very large option sets.
- SDKs in languages other than Python.
- A paid-API dependency to produce training labels.

## 5. Success criteria

1. A trained checkpoint answers `choice`, `score`, and `noul` questions,
   each in one forward pass, with N questions about one state answered in
   one batched call. **Met** — `checkpoints/seed{0,1,2}/published`.
2. On a held-out test slice, it beats the random baseline by a wide margin
   and beats the prompted-LLM baseline (`common.py`) on the same task.
   **Partially met:** all three seeds clear the 0.3175 random baseline by
   a wide margin (0.610–0.678 test accuracy); head-to-head comparison
   against the `common.py` baseline is demonstrated case-by-case in
   `2_one_pass.py` but not yet measured in aggregate over a test set.
3. Its probabilities are calibrated: ECE reported per question type, with
   the temperature fitted on a calibration slice disjoint from the test
   slice that produces every published number. **Met, with a caveat** —
   ECE 0.108–0.141 across seeds, a real improvement over uncalibrated
   output but not yet low by production standards; see `docs/MODEL_CARD.md`
   for a flagged anomaly in the `choice`/`score` calibration diagnostics.
4. The whole pipeline trains for **zero dollars** — development runs on a
   4 GB laptop GPU, the real run on Kaggle's free 2×T4. **Met for the
   base-size run** (trained locally on the 4 GB laptop GPU); the
   large-size Kaggle run hasn't happened yet.

Accuracy target, stated honestly rather than aspirationally: this project
trains a fresh head on a raw backbone, which in laya's own numbers is their
**0.646** row (a base-size specialist), not their 0.766 row (a fine-tune of
an already decision-pretrained checkpoint). Clearing 0.646 is the bar;
quoting 0.766 or Jev's 0.727 as the target would misrepresent what a raw
start can be expected to reach. **Current result: seed 2 reaches 0.678,
clearing the target; seeds 0 and 1 (0.610, 0.641) sit at or just below it**
— three seeds of one run on one dataset, not a settled claim.

## 6. What failure looks like

Not "a bug" — these are the ways the project could technically run and
still not be worth using:

- **A confident liar.** A model whose top probability reads 0.95 and is
  right 60% of the time is worse than no model.
- **Accuracy claimed off data the model or its calibration already saw.**
  Every published number must come from a test slice untouched by both.
- **An unreadable codebase.** The reference implementation's core package
  is ~21,000 lines; this project exists partly to prove the same idea fits
  in far less, readable by one person.

## 7. Milestones

See the design spec for the full breakdown; at a high level:

1. Encode + forward pass on a tiny random config (CPU-only sanity check).
2. Data loaders for the public datasets and the labelled-file formats.
3. Loss correctness, verified with a positive control before anything else.
4. Calibration and evaluation metrics, with the baseline made fair.
5. A real trained checkpoint (base-size locally, then large-size on
   Kaggle), multiple seeds, published numbers. **Base-size step done** —
   see `docs/MODEL_CARD.md`; large-size Kaggle run still ahead.
6. The `Agent` API and the side-by-side demo script.
7. The playground web app.

## 8. Open questions

- Whether the 2-layer scoring head is the right size (one ablation planned
  at milestone 3).
- The exact distractor-sampling rule for large option sets beyond "seeded".
- Whether a training CLI (vs. scripted calls to library functions) is worth
  building before or after the first real checkpoint lands.
