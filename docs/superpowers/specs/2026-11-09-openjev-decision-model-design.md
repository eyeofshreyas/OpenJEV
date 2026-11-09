# OpenJEV: a System-1 decision model

Design document. Written 2026-11-09.

## Intent

Build our own trained decision model — a model that takes unstructured text plus a set of
typed questions and returns calibrated probability distributions over the allowed answers, in
one forward pass, with no text generation anywhere.

TypeSafe AI's Jev (September 2026) named the category "System One models". Laya
(`NandhaKishorM/laya`, Apache-2.0) is the open implementation we are following: a bidirectional
encoder with a typed decision head, trained with RLCD, ~33 ms a decision. We want the same thing
under our own name, with our own weights, small enough that one person can read all of it.

### Why now

This repo currently contains the *argument* for the idea and none of the thing itself:
`1_vanilla_decoding.py` generates JSON token by token from Qwen3-0.6B and then shows that the
resulting `p(path)` is not a decision probability — it normalises over the vocabulary instead of
the allowed answers, multiplies without length normalisation, depends on field order, and never
scores the alternatives at all. `common.py` already fixes the scoring with a trie over a frozen
causal LM. That is still the wrong shape: a decoder-only model reading a prompt left-to-right,
borrowed for a job it was not trained for.

A decision model is the right shape, and the measured gap is large. Laya's own numbers: base
checkpoints sit near chance on its typed-decisions benchmark (0.36 and 0.35 against a 0.318
random baseline) and the fine-tuned checkpoint reaches 0.766 on the same 2,000 decisions — above
Jev's published 0.727. **The training is the product.** An untrained engine is a demo.

### Success

1. A checkpoint of ours answers `choice`, `score` and `noul` questions, each in one forward pass,
   with N questions about one state answered in one batched call.
2. On a held-out test slice it beats the random baseline by a wide margin and beats a prompted
   Qwen3-0.6B doing the same task through `common.py`'s trie scorer.
3. Its probabilities are calibrated: ECE reported per question type, with the temperature fitted
   on a calibration slice and the metrics measured on a test slice that is neither.
4. The whole thing trains on the 4 GB laptop GPU for development runs and on Kaggle's free
   2xT4 for the real one. Total spend: zero.

Target accuracy band, stated honestly up front: we are training a fresh head on a raw backbone,
which in laya's own head-to-head is the **0.646** row (their 149M specialist), not the 0.766 row
(a fine-tune of their already decision-pretrained checkpoint). Clearing 0.646 on typed-decisions
with a base-size model is a good result. Quoting 0.766 as our target would be dishonest.

## Goals

- Typed questions: `choice` (named options with descriptions), `score` (ordered levels),
  `noul` (Bernoulli, returns P(true)).
- Any number of questions against one state, answered in a single batched call with the state
  tokenized once and reused across them.
- Option text supplied by the caller at inference. No fixed label set compiled into the weights.
- Calibrated probabilities, trained against a strictly proper scoring rule.
- Trains from a labelled CSV or JSONL in one command.
- Honest evaluation: accuracy, ECE, Brier, latency, against a stated baseline.

## Non-goals

Deliberately out of v1. Each is a later decision, not a gap:

- HTTP server, MCP server, LangChain/LlamaIndex integrations, CLI beyond the training entry point.
- ONNX export, quantisation, TileLang kernels, any non-PyTorch runtime.
- Multilingual. v1 is English; mmBERT is a backbone swap once the loop works.
- A router across checkpoints.
- Images, audio, video, 73k-token documents.
- Conformal prediction sets, abstain/route flags, out-of-distribution flags.
- Shortlisting and tournament selection for large label spaces.
- SDKs in other languages.

## Anti-goals

Things that would make this a failure even if it technically runs:

- **A confident liar.** A model whose top probability reads 0.95 and is right 60% of the time is
  worse than no model. If calibration cannot be demonstrated, the project has not shipped.
- **Accuracy claimed off the training data**, or off the slice the temperature was fitted on.
  Every number we quote comes from a test slice that neither training nor calibration touched,
  and we say which slice.
- **21,000 lines.** Laya's core package is that big. We are building the part that decides, and
  we expect to read our own code a year from now.
- **A dependency on a paid API** to produce labels.

## Architecture

Taken from laya (`laya/common.py`), simplified. One sequence per (state, question):

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

- One `[MASK]` marker immediately before each option's text. The markers are the read-out
  positions — that is the whole mechanism. A variable number of options and arbitrary option
  text need no architectural change, because an option is just text with a marker in front of it.
- The encoder is **bidirectional**, so every marker sees the instructions, its own option, the
  other options and the entire state at once. This is what a causal LM cannot do and why the
  trie scorer in `common.py` is a workaround rather than a design.
- Question type is added as a learned embedding to every hidden state.
- A small transformer head (2 layers, **no positional encoding**) runs over the encoder output,
  then a scorer MLP (`LayerNorm -> Linear(d,d) -> GELU -> Linear(d,1)`) maps each marker's hidden
  state to **one logit per option**.
- Softmax over that question's markers, masked to its real option count, is the answer.
  `noul` is the two-option case and reports P(true). `score` softmaxes over levels; the reported
  score is the expected level.
- One temperature per question type, a buffer in the checkpoint, applied to the logits at
  inference.

**N questions, one call.** Each question is its own sequence, so N questions about one state are
a batch of N sequences in one forward pass — not one sequence sharing a single encoding of the
state. The state is tokenized once and spliced into each sequence, so the cost paid per question
is encoder compute, not tokenization. Worth being precise about, because `common.py` currently
prefills the state once and shares its KV cache, which on that one axis shares *more* than this
design does. What we are replacing is autoregression, not state sharing, and the latency table
reports both per decision and per call at N questions so the claim is checkable.

Budgets, which exist because a long state must not silently eat the options: `max_len` 1024 and
`head_max_len` 256 (laya's notebook figures, not the 512/192 library defaults — 192 cannot hold a
large option set), each option truncated at the tokenizer (48 tokens), and the state gets
whatever room is left. Truncation is reported to the caller, not guessed at — a caller cannot
reconstruct it from the outside because the budget is in tokens and moves per question. Two
options that collide into the same token span after truncation make the question unanswerable
and are rejected loudly rather than scored.

### Known defect we inherit, and what we do about it

In this sequential layout an option's position changes its logit: laya measures five *identical*
options spreading by 3.59 logits by position alone. Laya's fix is a parallel layout where every
option starts at the same position and options cannot attend to each other.

v1 ships the sequential layout and **option-order shuffling during training** (re-encode choice
questions with a random option order each epoch, moving the answer to match), which is the cheap
half of the fix and trains the position effect down instead of engineering around it. The
parallel layout is a v2 item with a measurement attached: re-run the identical-options test and
report the spread. We will know whether we need it.

## Training

RLCD: reinforcement learning against strictly proper scoring rules. The reward, per laya's
`proper_reward`:

```
reward = log_score + 0.5 * spherical_score   (every type)
       - 1.0 * ranked_probability_score      (score questions only, because levels are ordered)
```

A strictly proper scoring rule is maximised only by reporting your true belief, which is exactly
why the resulting probabilities mean something.

**Our v1 maximises this reward directly as a loss, by gradient descent on the reported
distribution, rather than by a GRPO-style policy gradient over sampled decisions.** Same
objective, same optimum, much lower gradient variance, and the code is one function. The
sampling-based policy gradient buys reward shaping we do not need while the reward is a pure
scoring rule of the label. This is a deliberate simplification with a known ceiling: it is marked
in the code and revisited if we add rewards that are not differentiable in the reported
distribution (a cost matrix over confusions, a downstream outcome). Laya measured their
`soft-ce` arm at 0.790 against their `rlcd` arm at 0.7715 on typed-decisions, and our loss sits
between the two — it is soft cross-entropy plus the spherical and RPS terms — so the
simplification points at the arm that won.

### Targets: a belief, not a one-hot

This is the load-bearing detail. A proper scoring rule is maximised by reporting your true
belief *when the target is a belief*. Against one-hot labels its optimum is a one-hot output, so
training on hard labels alone drives confidence to the ceiling and leaves the whole calibration
claim resting on post-hoc temperature — the first anti-goal above. Laya says the same thing of
their own loop: it trains on gold distributions, not hard labels, and "the loop is only as good
as the targets".

So the loss reads two target shapes, and the data plan supplies both:

- `gold` — a full probability distribution per question. Used wherever we have one.
- `expected` — one answer per question: a choice label, `true`/`false`, or a level index
  (a fractional level such as `1.5` splits weight across levels 1 and 2). **Label smoothing 0.1**
  applied, so a hard label enters as a blunt belief rather than a certainty.

Both shapes are laya's JSONL format, read from the same file, so our data is interchangeable
with theirs and with their evals.

### Data

Decided: **typed-decisions for the gold distributions, plus the public sets for breadth.**

- **`LocalLLaMA/typed-decisions`** (config `all`) — 1,200 train cases / 6,000 decisions and
  400 test / 2,000 decisions, all three question types, across workflows including Customer
  Service and Security Incidents. Free, and already in the `{state, questions, gold}` shape we
  adopt. This is the only source of real distributional targets, the only usable body of `score`
  data, and the only benchmark on which laya's 0.766 and Jev's 0.727 are comparable to anything
  we produce. Its golds come from a teacher model, so on this slice we are partly imitating that
  teacher; that is stated in the model card rather than hidden.
- **MASSIVE** (English intents) — `choice` with a large option set, the hard case. Measured on
  laya's published protocol: **gold plus 19 sampled distractors, 20 options, seeded sampling**,
  so the number is comparable to theirs. The full 60-label setting is reported separately as a
  known limit, because 60 `[MASK]`-prefixed options do not fit even a 256-token head — laya
  reports 0.515 at their default budget and 0.622 only at `head_max_len` 512. We report what we
  measure and name the budget.
- **BoolQ** — `noul`. **AG News** — `choice`, few options. Both hard-labelled, so both enter
  smoothed.

First domain, my call since it was not specified: **support and incident triage** — route a
message to a team, rate its urgency, flag whether the sender is threatening to leave. It is what
the existing fraud-triage example in `common.py` is already reaching for, it is what laya's demo
does so we can compare directly, and typed-decisions covers it.

### Warm start: none

Decided: **raw `answerdotai/ModernBERT`, fresh head.** Not a fine-tune of
`convaiinnovations/laya`, although that is Apache-2.0 and would be permitted, faster and about
0.12 more accurate. The point of the project is our own weights from the backbone up.

The cost of that choice is a real training risk, and it changes the milestones rather than just
the hours. Laya documents an `ln K` plateau on raw starts (their issue #963): on identical data
and configuration, **9 of 15 seeds escaped it and 6 never did within 32 epochs**. The head sits
at uniform output and the loss sits at `ln K`. So:

- Every training milestone runs **multiple seeds** and reports how many escaped. A single failed
  seed is not evidence of a bug.
- Every training milestone runs a **positive control** first: prepend a tag to the state that
  spells the answer, and confirm the model reaches near-perfect accuracy. That separates "the
  budget cannot fit the signal" and "the plumbing is broken" from "the task is hard", which is
  otherwise unanswerable when everything sits at `ln K`.

### The recipe

| | Development (this laptop) | The real run (Kaggle) |
|---|---|---|
| Backbone | `answerdotai/ModernBERT-base`, 149M | `answerdotai/ModernBERT-large`, 421M |
| Device | RTX 3050 Laptop, 4 GB (3.8 usable), Ampere | free 2xT4, 16 GB each, Turing, DDP |
| Precision | **bf16** autocast | **fp16** autocast + GradScaler |
| Micro-batch / accum | 2 / 32 (effective 64) | 8 / 4 per rank (effective 64) |
| `max_len` / `head_max_len` | 1024 / 256 | 1024 / 256 |
| Gradient checkpointing | off, fallback if OOM | on, encoder and head |
| Expected footprint | ~2.8 GB | ~12 GB per rank |
| Runtime | minutes on a small set | ~5–7 h, 4 epochs, ~30k questions |

**A T4 has no bf16** (compute capability 7.5), so the run that produces every published number
is fp16, and fp16 plus a log- and norm-heavy objective is the overflow-prone combination: the
loss takes `log` of clamped probabilities and the spherical term divides by `‖q‖`. Therefore
the scoring rule computes in **fp32 regardless of autocast**, with a GradScaler, grad-norm
clipping at 1.0, and a run that aborts on a non-finite loss rather than training through it.
The laptop is Ampere and gets bf16, so a loss curve that only misbehaves on Kaggle is a
precision bug, not a data bug — worth knowing which is which before spending five hours.

A 4 GB card cannot full-fine-tune the large backbone (~7 GB of weights, grads and Adam state).
That is why development is base-size and the real run is Kaggle's. If the large run ever has to
happen locally, the fallbacks in order are: freeze the bottom encoder layers, 8-bit Adam, LoRA.

Optimizer: AdamW, lr 2e-5 on the encoder and 1e-4 on the head, cosine schedule, warmup 10%.
Report the optimizer-update budget before a run starts (`epochs * ceil(ceil(items / micro_batch)
/ grad_accum)`), because that number and not the epoch count predicts whether a run converges;
a small dataset needs more epochs to reach a comparable count. Write a rolling checkpoint after
every epoch so a crash or a Kaggle timeout costs one epoch.

## Evaluation

Three slices, not two: **train / calibration / test**. The temperature is fitted on the
calibration slice, every published number is measured on the test slice, and a run's report says
which slice produced which figure. typed-decisions ships its own train/test split, so the
calibration slice is held out of its train side. Note that a temperature fitted on fewer than
~10 items of a type is meaningless; if a type ever falls below that the run reports the
temperature as unfitted (1.0) rather than quietly fitting noise.

Every run reports, on the test slice:

- accuracy per question, and within-one-level for `score`
- **ECE** (15 bins) and Brier score, per question type — the calibration claim
- latency per decision and per call at N questions, median and p95, on this laptop's GPU and CPU
- the random baseline for the question mix, so a number can be read without the paper
- seeds run and seeds escaped the `ln K` plateau

Baselines to beat, in order: random; prompted Qwen3-0.6B through `common.py`'s trie scorer; our
own untrained head (near chance — if we score near chance after training, training is broken,
and this is how we find out).

**Baseline parity.** `common.py`'s `SCHEMA` has a description per field but none per option,
while our `choice` questions put each option's criteria text into the sequence — so as it stands
the baseline sees strictly less than we do, which flatters us. We extend `common.py` to carry
per-option descriptions and to render `score` levels, so the comparison is fair.
`1_vanilla_decoding.py` stays untouched; it is the illustration, not the baseline.

## Shape of the code

A small package beside the existing scripts, which stay as the explanation of why it exists:

```
openjev/
  encode.py     render options, build the head and the sequence, budgets, collate
  model.py      DecisionModel (encoder + type embedding + head + scorer + temperature)
  score.py      the proper scoring rule, as reward and as loss; both target shapes
  data.py       CSV and JSONL -> training items; typed-decisions and the public loaders; dry run
  train.py      train loop, temperature fitting, checkpoint save/load, seeds, positive control
  eval.py       accuracy, ECE, Brier, latency, baselines
  __init__.py   Agent.predict(state, questions) -> answers with probabilities
1_vanilla_decoding.py   unchanged: why generated JSON is not a decision
2_one_pass.py           the same questions through our model, with the numbers side by side
```

Dependencies: `torch`, `transformers` (already present), plus `datasets` for the public sets and
`safetensors` for checkpoints. Nothing else without a reason.

Checkpoint layout: `model.safetensors`, `encoder/`, `tokenizer/`, **`rl_agent_config.json`**,
`questions.json`, `checkpoint_latest/`. Laya's layout and laya's config filename, because
`laya.load` keys off that name and "their tooling can read our output" should not fail on a
filename.

**Licensing:** Apache-2.0, with a `NOTICE` crediting laya, since this is an acknowledged
simplification of Apache-2.0 code and reuses its sequence format, checkpoint layout and data
format. Any published weights carry a model card naming the training sets, their licences, and
the fact that the typed-decisions targets are teacher-generated.

## Milestones

Each one ends with a runnable check; the next does not start until it passes.

1. **Encode and forward.** `encode.py` + `model.py` on a tiny random config, CPU. Asserts: one
   marker per option, masked softmax sums to 1, option-count masking, head and state budgets
   respected, colliding options rejected, truncation reported.
2. **Data.** typed-decisions, the public sets, CSV and JSONL to items — both target shapes, with
   skip reasons counted and a dry run that prints the item and optimizer-update budget without
   loading weights.
3. **The loss works.** Positive control first (tagged state, near-perfect accuracy). Then overfit
   60 rows to near-perfect training accuracy on the laptop, across 3 seeds. If this does not
   happen in minutes, nothing later is worth running.
4. **Calibration and evaluation.** Per-type temperature on the calibration slice; the full metric
   report on the test slice; the baseline comparison with a parity-fixed `common.py`.
5. **A real checkpoint.** typed-decisions + MASSIVE (20-option) + BoolQ + AG News. Base-size
   locally first, then large on Kaggle 2xT4, multiple seeds, plateau escapes reported. Published
   numbers come from here. The bar is laya's 0.646 specialist row, not their 0.766.
6. **The API and the demo.** `Agent.predict`, save/load, and `2_one_pass.py` next to
   `1_vanilla_decoding.py`.

## Left to the builder

- Exact tokenizer and padding mechanics, the collate function, and how questions sharing a state
  reuse one tokenization.
- Whether the 2-layer head is right; it is laya's number and worth one ablation at milestone 3.
- The distractor sampling rule for MASSIVE's 20-option protocol, beyond "seeded".
- Learning-rate specifics beyond the starting point above, and the seed count per milestone.
- Whether milestone 5's local base-size run is worth the hours before moving to Kaggle.
