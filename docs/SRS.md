# OpenJEV — Software Requirements Specification

Companion to `docs/PRD.md` (what/why) and
`docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md` (the
binding design this implements). This document specifies *what the software
must do*, at the level of its actual modules and interfaces.

## 1. Introduction

### 1.1 Purpose

Specify the functional and non-functional requirements of OpenJEV: a Python
library that answers typed questions about a text input with calibrated
probability distributions, plus a reference baseline and a web playground
for comparing the two.

### 1.2 Scope

Covers `openjev/` (the decision model library), `common.py` (the baseline),
and `playground/` (the FastAPI + React demo app). Excludes items listed as
non-goals in the design spec (§4 of this document restates the relevant
ones as explicit exclusions).

### 1.3 Definitions

| Term | Meaning |
|---|---|
| **State** | The unstructured text (or structured text) describing the situation being judged. |
| **Question** | A typed request for a judgment about the state: `choice`, `score`, or `noul`. |
| **Marker** | A `[MASK]` token placed immediately before an option's text; its hidden state is read out as that option's logit. |
| **Calibration** | Agreement between a reported probability and the observed frequency of being correct at that probability. |
| **ECE** | Expected Calibration Error — the standard metric for calibration quality. |
| **Temperature** | A single fitted scalar divisor applied to logits before softmax, to correct over/under-confidence without changing accuracy. |
| **Agent** | The public Python interface (`openjev.Agent`) that answers questions about a state. |

## 2. Overall description

### 2.1 Product perspective

OpenJEV is a standalone library plus a thin demo app. It depends on
`torch`, `transformers`, `datasets`, and `safetensors`, and optionally
`fastapi`/`uvicorn` for serving (kept out of the training/eval import path).
It is not a service with accounts, persistence, or multi-tenant state; the
playground's run history lives in browser `localStorage` only.

### 2.2 Operating environment

- Python 3.12, PyTorch ≥2.2, Transformers ≥5.0.
- Runs on CPU or a single GPU for inference; training targets either a 4 GB
  laptop GPU (development, bf16 on Ampere) or Kaggle's free dual-T4
  environment (the real run, fp16 + GradScaler, since T4 has no bf16).
- No external network access is required at inference time once a
  checkpoint and tokenizer are local; the default backbone
  (`answerdotai/ModernBERT-base`) is fetched from the Hugging Face Hub on
  first use otherwise.

### 2.3 Constraints

- No paid API may be a dependency for producing training labels.
- Total training spend across both environments: $0.
- The core decision-model package must stay small enough for one person to
  read in full — an explicit anti-goal is matching the ~21,000-line scale
  of the reference implementation.

## 3. Functional requirements

### FR-1 — Question validation

The system **shall** validate every question before use (`openjev.encode`):

- FR-1.1: Each question **shall** declare a `type` of `choice`, `score`, or
  `noul`.
- FR-1.2: `choice` and `score` questions **shall** supply named options
  (`criteria`) with descriptive text; `noul` questions **shall not** require
  options.
- FR-1.3: Two options that collide into the same token span after
  truncation **shall** be rejected with a raised error
  (`OptionCollision`), not silently scored.
- FR-1.4: Exceeding the option/token budget **shall** raise
  `OptionBudgetExceeded` rather than truncating silently.

### FR-2 — Sequence construction

- FR-2.1: The system **shall** build one sequence per (state, question) in
  the form `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 ...
  [SEP] <state> [SEP]`, with exactly one `[MASK]` marker immediately before
  each option's text (`build_sequence`, `build_head`).
- FR-2.2: The head (question + options) **shall** be budgeted at
  `head_max_len` tokens and the full sequence at `max_len` tokens, with the
  state receiving whatever room remains.
- FR-2.3: Any truncation of the state **shall** be reported in the
  prediction's `usage` field, not merely absorbed.
- FR-2.4: When N questions share one state, the state **shall** be
  tokenized once and spliced into each question's sequence (`collate`).

### FR-3 — Model inference

- FR-3.1: The model (`openjev.model.DecisionModel`) **shall** run a
  bidirectional encoder over the full sequence, add a learned embedding for
  the question type, refine hidden states with a small transformer head,
  and score each marker to one logit via a scorer MLP
  (`LayerNorm → Linear → GELU → Linear`).
- FR-3.2: The system **shall** compute a softmax over each question's
  markers, masked to that question's real option count
  (`model.probabilities`).
- FR-3.3: For `noul`, the system **shall** report `P(true)` as the
  two-option case. For `score`, the system **shall** report the expected
  level (probability-weighted sum over level indices), not just the argmax
  level.
- FR-3.4: A single call **shall** answer any number of questions about one
  state in one batched forward pass; `usage.forward_passes` **shall**
  report the actual count.

### FR-4 — Calibration

- FR-4.1: The system **shall** support one temperature per question type,
  stored as a checkpoint buffer, applied to logits at inference
  (`model.probabilities`, `temperature` argument).
- FR-4.2: Temperature fitting (`train.fit_temperatures`) **shall** use a
  calibration slice disjoint from both the training data and the slice used
  to report final metrics.
- FR-4.3: If fewer than ~10 items of a question type are available for
  fitting, the system **shall** leave that type's temperature unfitted
  (1.0) rather than fit it on too little data.

### FR-5 — Training

- FR-5.1: The training loss (`openjev.score`) **shall** implement the
  reward `log_score + 0.5 * spherical_score − 1.0 * ranked_probability_score`
  (RPS applied only to `score` questions), maximized directly by gradient
  descent on the reported distribution.
- FR-5.2: Training targets **shall** support two shapes: a full gold
  probability distribution (`gold`), and a single expected answer with
  label smoothing applied (`expected`) — never a bare one-hot target.
- FR-5.3: The scoring rule's arithmetic **shall** run in fp32 regardless of
  the surrounding autocast precision, to avoid overflow in the log and
  spherical terms under fp16.
- FR-5.4: A run **shall** abort on a non-finite loss (`NonFiniteLoss`)
  rather than continuing to train through it.
- FR-5.5: `choice` questions **shall** have their option order shuffled
  during training (with the target permuted to match) to reduce
  position-dependent bias in the logits.
- FR-5.6: The system **shall** write a checkpoint after every epoch
  (`save_checkpoint`), so an interruption costs at most one epoch.
- FR-5.7: Training **shall** support resuming from a saved checkpoint
  (`load_checkpoint`), restoring model, optimizer, and scheduler state.

### FR-6 — Data loading

- FR-6.1: The system **shall** load labelled items from JSONL and CSV files
  (`items_from_jsonl`, `items_from_csv`), reporting a count of skipped rows
  and why each was skipped.
- FR-6.2: The system **shall** load the public benchmark sets used for
  training and evaluation — `typed-decisions`, MASSIVE, BoolQ, AG News —
  through dedicated loaders, each producing the same `{state, questions,
  gold/expected}` item shape.
- FR-6.3: `split_items` **shall** partition items into train/calibration
  slices by a seeded, reproducible split.

### FR-7 — Evaluation

- FR-7.1: `openjev.eval.evaluate` **shall** report, per question type:
  accuracy (and within-one-level accuracy for `score`), ECE (15 bins),
  Brier score, and the random baseline for that question mix.
- FR-7.2: The system **shall** report latency per decision and per call at
  N questions, both median and p95.
- FR-7.3: Every reported number **shall** state which data slice (train,
  calibration, or test) produced it.

### FR-8 — Public API (`Agent`)

- FR-8.1: `openjev.load(path)` **shall** load a trained checkpoint and
  return a ready-to-use `Agent`.
- FR-8.2: `Agent()` **shall** construct an agent on a fresh, untrained
  backbone for exploring the API without a checkpoint.
- FR-8.3: `Agent.predict(state, questions)` **shall** return, per question:
  the probability distribution over its answers, the chosen answer
  (`choice`/`score`/`true`-`false` as appropriate), and an
  `answer_confidence`. The call **shall** also return a top-level `usage`
  object with `forward_passes`, elapsed time, and truncation status.

### FR-9 — Baseline (`common.py`)

- FR-9.1: The baseline **shall** answer the same typed questions by
  prompting a frozen causal LM and scoring its generated tokens via a trie
  over the allowed answers, so it can be directly compared to `Agent` on
  identical input.
- FR-9.2: The baseline's schema **shall** carry per-option descriptions and
  render `score` levels, so it is not evaluated on strictly less
  information than the decision model receives (baseline parity).

### FR-10 — Playground

- FR-10.1: The backend (`playground/backend/app.py`, FastAPI) **shall**
  expose an endpoint that accepts a state and a set of questions and
  returns the same shape `Agent.predict` returns.
- FR-10.2: The backend **shall** expose a status endpoint reporting model
  identity and readiness.
- FR-10.3: The frontend (React, Vite) **shall** display, for every
  question, the full probability distribution (not just the top answer) as
  a bar per option, and **shall** surface a truncation warning when the
  state lost tokens to the head budget.
- FR-10.4: Run history **shall** be stored in browser `localStorage` only
  — no server-side persistence, no accounts.

## 4. Explicit exclusions

The following are out of scope for this specification and **shall not** be
assumed present: an MCP server; LangChain/LlamaIndex integrations; a
training CLI beyond scripted library calls; ONNX export or quantization;
multilingual support; a router across checkpoints; non-text modalities;
conformal prediction / abstain flags; tournament selection over large label
spaces; non-Python SDKs.

## 5. Non-functional requirements

### 5.1 Performance

- NFR-1: Answering N questions about one state **shall** cost exactly one
  batched forward pass, independent of N (not N forward passes).
- NFR-2: Reported latency numbers **shall** be measured, not asserted —
  every evaluation run **shall** produce its own median/p95 figures on the
  hardware it ran on.

### 5.2 Reliability / correctness

- NFR-3: A question whose options cannot be distinguished after truncation
  **shall** fail loudly (raise), never silently return a meaningless
  answer.
- NFR-4: Calibration metrics **shall never** be computed on a data slice
  that the temperature fitting or the model training already touched.

### 5.3 Calibration quality

- NFR-5: The system is considered to have failed its calibration goal if it
  reports confident, miscalibrated probabilities (a "confident liar") —
  this is treated as a shipped failure even if all other requirements pass.

### 5.4 Maintainability

- NFR-6: The `openjev/` package **shall** remain small enough to be read in
  full by one engineer; this is an explicit design constraint, not an
  aspiration.

### 5.5 Cost

- NFR-7: Training, end to end across development and the real run, **shall**
  cost $0 in compute and **shall not** depend on a paid labeling API.

### 5.6 Licensing

- NFR-8: The project **shall** remain Apache-2.0, with `NOTICE` crediting
  laya for the sequence format, checkpoint layout, and data format it
  reuses.

## 6. External interfaces

### 6.1 Python API

```
openjev.load(path, device=None) -> Agent
openjev.Agent().predict(state: str, questions: dict) -> dict
```

### 6.2 HTTP API (playground backend, as implemented)

```
GET  /api/status    -> model readiness / identity
GET  /api/example    -> a sample state + questions payload
POST /api/predict    -> {"state": ..., "questions": ...} -> Agent.predict's result shape
```

(The original design doc specified `POST /v1/decide` / `GET /health`; the
implemented playground uses the routes above — this document reflects the
code as built.)

## 7. Verification

Each functional requirement above is covered by an existing test module:
`test_encode.py` (FR-1, FR-2), `test_collate.py` (FR-2.4), `test_model.py`
(FR-3), `test_calibrate.py` (FR-4), `test_score.py`/`test_targets.py`/
`test_train.py` (FR-5), `test_data.py`/`test_datasets.py` (FR-6),
`test_eval.py` (FR-7), `test_agent.py` (FR-8), `test_baseline.py` (FR-9),
`test_one_pass.py` (end-to-end demo). Run with:

```bash
pytest -q -W error
```
