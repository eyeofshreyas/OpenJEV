# OpenJEV — Architecture

How the system is actually put together. For *why* these choices were made,
see `docs/CONCEPTS.md`; for the original design intent, see
`docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md`. This
document follows the code as it exists.

## 1. System overview

```
                         ┌─────────────────────────┐
                         │   playground/frontend    │  React + Vite
                         │  (state + question form,  │
                         │   probability bars)        │
                         └────────────┬─────────────┘
                                      │ HTTP (/api/*)
                         ┌────────────▼─────────────┐
                         │   playground/backend      │  FastAPI
                         │     (app.py)               │
                         └────────────┬─────────────┘
                                      │ calls
                         ┌────────────▼─────────────┐
                         │   openjev.Agent            │  Public Python API
                         │   (__init__.py)            │
                         └────────────┬─────────────┘
                                      │ uses
        ┌─────────────┬──────────────┼──────────────┬─────────────┐
        ▼             ▼              ▼              ▼             ▼
   encode.py      model.py       score.py        data.py      train.py / eval.py
  (sequence,     (encoder +     (proper        (datasets,    (training loop,
   budgets,       head +         scoring         JSONL/CSV      checkpoints,
   collate)       scorer)        rule)           loaders)       metrics)

                         ┌─────────────────────────┐
                         │        common.py           │  Baseline: frozen
                         │  (prompt + trie scorer)    │  causal LM, compared
                         └─────────────────────────┘  against, not reused
```

Two independent engines answer the same typed-question interface:
`openjev.Agent` (the decision model this project builds) and `common.py`
(a prompted-LLM baseline it's measured against). `2_one_pass.py` runs a
case through both side by side; `1_vanilla_decoding.py` is a standalone
illustration of why naive token-probability decoding fails, and is not
wired into either engine.

## 2. Request flow (inference)

1. A caller provides a `state` (text) and a `questions` dict, each question
   typed `choice` / `score` / `noul` with instructions and (for `choice`/
   `score`) labeled options.
2. `encode.validate_question` / `validate_state` check the input; invalid
   shapes raise before any tokenization happens.
3. For each question, `encode.build_head` renders its options into text
   with one `[MASK]` marker placed immediately before each option, and
   budgets it against `head_max_len`.
4. `encode.build_sequence` assembles the full sequence —
   `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 ... [SEP]
   <state> [SEP]` — tokenizing the state once and reusing it across every
   question that shares it. Truncation, if any, is recorded rather than
   silently absorbed.
5. `encode.collate` batches all N question-sequences for this state (and,
   in evaluation, across states) into padded tensors in one call.
6. `model.DecisionModel.forward` runs the batch once:
   - The backbone (bidirectional encoder, default `ModernBERT-base`) reads
     every sequence in full, in both directions.
   - A learned embedding for the question's type (`choice`/`score`/`noul`)
     is added to every hidden state.
   - A small transformer head (2 layers, no positional encoding) refines
     the encoder output.
   - A scorer MLP (`LayerNorm → Linear(d,d) → GELU → Linear(d,1)`) reduces
     each marker position's hidden state to one logit.
7. `model.probabilities` takes the per-marker logits, applies that
   question's fitted temperature (if any), masks to the real option count,
   and softmaxes — yielding a distribution per question.
8. `openjev.Agent._answer` turns each distribution into the public answer
   shape: `probabilities`, the picked `choice`/`score`/boolean, and
   `answer_confidence`.
9. `Agent.predict` assembles `usage` (`forward_passes`, elapsed ms,
   `truncated`) alongside the per-question answers and returns the whole
   result.

One batched forward pass answers every question about one state — N
questions cost N sequences' worth of encoder compute, but exactly one
model call, not N decode loops.

## 3. Model architecture

Sequence layout (one sequence per state×question pair):

```
[CLS] <type> question: <instructions> [SEP]
      [MASK] option_0_text [MASK] option_1_text ... [SEP]
      <state text> [SEP]
```

- **Markers are the mechanism.** A `[MASK]` immediately before each
  option's text is the only read-out position; everything else exists to
  give that position context. A variable number of options, or new option
  text, needs no architecture change — an option is just text with a
  marker in front of it.
- **Bidirectional attention** lets every marker see the instructions, its
  own option, every other option, and the full state simultaneously. This
  is the capability a causal LM lacks in one pass, and the reason
  `common.py`'s trie scorer exists as a workaround for a model that can't
  do this natively.
- **Type embedding**: the question's type is added to every hidden state
  so the same backbone and head serve all three question types.
- **Head**: 2 transformer layers, no positional encoding (laya's
  configuration; untested whether 2 is optimal for this project's scale —
  flagged as an open ablation).
- **Scorer**: `LayerNorm → Linear(d, d) → GELU → Linear(d, 1)` per marker,
  producing one logit per option.
- **Read-out**: softmax over a question's markers, masked to its real
  option count. `noul` is the two-option case reporting `P(true)`; `score`
  reports the probability-weighted expected level rather than a bare
  argmax, so a near-miss is visible in the number itself.
- **Calibration**: one temperature scalar per question type, stored as a
  checkpoint buffer, dividing logits before the final softmax at inference.

### Known architectural defect, inherited and mitigated

In this sequential layout, an option's *position* in the sequence measurably
shifts its logit (laya measured identical options spreading by 3.59 logits
by position alone). This project's v1 mitigation is training-time option
order shuffling (re-encoding `choice` questions with a randomized option
order each epoch, permuting the target to match) rather than laya's
parallel, non-attending layout — cheaper to implement, with the residual
spread measured rather than assumed fixed. A parallel layout remains a v2
option if the measurement says shuffling wasn't enough.

## 4. Training architecture

```
Items (state, questions, gold/expected)
        │
        ▼
 target_vector()  →  gold distribution, or smoothed expected-answer vector
        │
        ▼
  encode.collate() →  batched sequences, with per-epoch option shuffling
        │
        ▼
  DecisionModel.forward() → per-marker logits
        │
        ▼
  score.py: reward = log_score + 0.5·spherical − 1.0·RPS (score questions only)
        │            maximized directly via gradient descent (no sampling,
        │            no policy gradient — see docs/CONCEPTS.md §6)
        ▼
  AdamW (lr 2e-5 encoder / 1e-4 head, cosine schedule, 10% warmup)
        │
        ▼
  save_checkpoint() every epoch  →  resumable via load_checkpoint()
```

Precision is environment-dependent: bf16 autocast on the Ampere laptop GPU
(development), fp16 autocast + `GradScaler` on Kaggle's T4s (compute
capability 7.5 has no bf16) for the real run. Regardless of autocast, the
scoring rule itself computes in fp32 with gradient-norm clipping at 1.0,
because its log and spherical terms are overflow-prone under fp16. A
non-finite loss aborts the run (`NonFiniteLoss`) rather than training
through corrupted gradients.

Two safeguards against a raw backbone's head getting stuck at uniform
output (the `ln K` plateau, documented by laya across multiple seeds):
every milestone runs a **positive control** (a state tagged with the
answer, which must train to near-perfect accuracy) before trusting any
result on real data, and every real run tracks **multiple seeds** and how
many escaped the plateau.

## 5. Data architecture

`openjev/data.py` normalizes every source into one `Item` shape before
anything downstream sees it:

- **File-based**: `items_from_jsonl`, `items_from_csv` — labelled data in
  laya's JSONL shape, read from either format, with skip reasons counted
  rather than swallowed.
- **Public datasets**, each wrapped in a loader returning the same item
  shape: `load_typed_decisions` (the only source of real teacher
  probability distributions — the primary calibration benchmark),
  `load_massive` (large option sets, sampled to a fixed protocol for
  comparability with laya's published numbers), `load_boolq` (`noul`),
  `load_ag_news` (`choice`, few options).
- **Two target shapes**, computed by `target_vector`: a full `gold`
  distribution where available, or a label-smoothed `expected`-answer
  vector otherwise — never a bare one-hot target (training against one
  would push confidence to the ceiling and defeat the calibration goal by
  construction).
- **Splitting**: `split_items` carves a calibration slice out of the
  training side, keeping the published test slice untouched by both
  training and temperature fitting.

## 6. Evaluation architecture

`openjev/eval.py` computes, per question type, on the test slice only:
accuracy (plus within-one-level accuracy for `score`), 15-bin ECE, Brier
score, the random baseline for that question mix, and latency (median/p95,
per-decision and per-call). `train.fit_temperatures` produces the
temperatures these numbers are reported *with*, fitted on the calibration
slice — never the slice the numbers come from.

## 7. Serving architecture (playground)

```
playground/frontend (React + Vite)  ──HTTP──▶  playground/backend (FastAPI)
  state + question builder form                   GET  /api/status
  probability bars, per option                     GET  /api/example
  truncation warning                                POST /api/predict
  localStorage run history (client-only)                   │
                                                             ▼
                                                      openjev.Agent
```

The backend loads one model at startup and holds no per-request state; the
`/api/predict` route returns exactly the shape `Agent.predict` returns, so
the UI and the library can never disagree about what an answer looks like.
Run history lives in the browser only — no database, no accounts, matching
the project's deliberate non-goals for the playground. In development, Vite
proxies API calls to the backend; `playground/README.md` has the exact run
commands.

## 8. Checkpoint layout

Inherited from laya, deliberately, so existing tooling built against laya's
format isn't a reason to diverge for its own sake: `model.safetensors`,
`encoder/`, `tokenizer/`, `rl_agent_config.json`, `questions.json`,
`checkpoint_latest/`.

## 9. Repository layout

| Path | Role |
|---|---|
| `openjev/encode.py` | Question validation, sequence/marker layout, budgets, collation |
| `openjev/model.py` | `DecisionModel`: encoder + type embedding + head + scorer + temperature |
| `openjev/score.py` | The strictly proper scoring rule, as the training loss |
| `openjev/data.py` | Target construction, file loaders, public dataset loaders, splitting |
| `openjev/train.py` | Training loop, temperature fitting, checkpoint save/load, seeds |
| `openjev/eval.py` | Accuracy, ECE, Brier, latency, baselines |
| `openjev/__init__.py` | `Agent` — the public `predict`/`load` interface |
| `common.py` | Baseline: frozen causal LM + trie scorer over generated tokens |
| `1_vanilla_decoding.py` | Standalone illustration of why naive decoding fails (not wired to either engine) |
| `2_one_pass.py` | Runs one case through both engines side by side |
| `playground/backend/` | FastAPI app exposing `Agent.predict` over HTTP |
| `playground/frontend/` | React + Vite UI for the playground |
| `docs/CONCEPTS.md` | First-principles explanation of the ideas this project rests on |
| `docs/superpowers/specs/` | The binding design document |
| `docs/superpowers/plans/` | The implementation plan this was built from |
| `tests/` | One test module per `openjev` module, plus baseline/one-pass coverage |

## 10. Deployment environments

| | Development | Real training run |
|---|---|---|
| Hardware | RTX 3050 Laptop, 4 GB usable ~3.8 GB, Ampere | Kaggle free 2×T4, 16 GB each |
| Backbone | `ModernBERT-base`, 149M | `ModernBERT-large`, 421M |
| Precision | bf16 autocast | fp16 autocast + GradScaler |
| Gradient checkpointing | off (fallback if OOM) | on (encoder + head) |
| Expected memory | ~2.8 GB | ~12 GB per rank |

No production deployment environment exists yet — there is no published
checkpoint to deploy. The playground is a local/dev-only demo.
