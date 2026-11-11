# Model card — base-size, seed-averaged checkpoints

Covers the checkpoints in `published_checkpoints.zip` (extract to
`checkpoints/`). These are the first real trained checkpoints this project
has produced — milestone 5's base-size, local run, not yet the large-size
Kaggle run the design spec also calls for.

## What it is

- **Backbone:** `answerdotai/ModernBERT-base` (149M parameters), raw —
  no warm start from a decision-pretrained checkpoint.
- **Head:** 2-layer transformer + scorer MLP, trained from scratch
  (`openjev/model.py`).
- **Training data:** `LocalLLaMA/typed-decisions` (config `all`) only —
  1,200 training cases / 6,000 decisions, evaluated on its 400-case / 2,000
  decision test split. **Not yet trained on** MASSIVE, BoolQ, or AG News;
  those are implemented loaders (`openjev/data.py`) the design spec calls
  for in a broader run that hasn't happened yet.
- **Important caveat on the data:** `typed-decisions`' gold distributions
  come from a teacher model, not human annotators. Training and evaluating
  on this set means partly imitating that teacher — including its
  mistakes. There is no independent human-labelled ground truth here.
- **Training run:** `scripts/train_base.py` — 4 epochs, batch size and
  gradient checkpointing auto-sized to the GPU found (this run used the
  4 GB-card config: micro-batch 4, grad-accum 16, gradient checkpointing
  on). A positive control (a tagged state that spells its own answer) was
  trained and passed before any of the three seeds below ran, confirming
  the training loop and token budget aren't broken.
- **Three seeds (0, 1, 2), each independently trained and calibrated.**
  All three escaped the raw-start `ln K` uniform-output plateau this
  project's design doc warns about. `checkpoints/report.json` has the raw
  numbers; summarized below.
- **License:** Apache-2.0, same as the rest of the repo. See root `NOTICE`
  for the laya attribution this project's architecture and data format
  follow.

## Results (test split, held out from both training and calibration)

Random baseline on this question mix: **0.3175**.

| Seed | Test accuracy | `choice` | `score` | `noul` |
|---|---|---|---|---|
| 0 | 0.610 | 0.575 | 0.546 | 0.730 |
| 1 | 0.641 | 0.595 | 0.581 | 0.768 |
| 2 | **0.678** | 0.625 | 0.660 | 0.755 |

Accuracy is unaffected by the temperature-fitting bug below (temperature
scaling can only reshape probabilities, never change which option is
largest, so the argmax — and therefore accuracy — is identical before and
after the fix). **ECE and Brier are not** — they depend on the actual
probability values, which the bug corrupted for `choice`/`score`. The
original published ECE/Brier numbers for those two types are retracted;
see the next section for why, and this card will carry the recomputed
numbers once the post-fix evaluation run finishes.

Seed 2 is the best on test accuracy and is the one referenced as the
default in the README (`checkpoints/seed2/published`). All three are kept,
not just the best, per this project's stated anti-goal of not quietly
discarding seeds that ran worse.

**Against the design spec's own target band:** the stated bar was laya's
0.646 base-size, raw-start specialist row — explicitly *not* laya's 0.766
fine-tune or Jev's published 0.727, because those start from an
already-decision-pretrained checkpoint this run doesn't have. Seed 2
(0.678) clears that bar; seeds 0 and 1 (0.610, 0.641) sit at or just below
it. This is three seeds of one run, not a claim that the architecture
reliably clears 0.646.

## What "calibrated" means here, honestly

**Fixed, not just flagged.** The first version of this card noted that
`choice`/`score` calibration showed `NaN` in their NLL diagnostics while
`noul` didn't, and that their fitted temperature landed exactly on the
search grid's floor (0.25) — suspicious rather than a real fit. That's
root-caused now: `openjev.train._type_nll` computed
`target * torch.log_softmax(scaled_logits)`, and padded/invalid option
slots have `target == 0` there but `log_softmax == -inf` (they're masked
out before the softmax) — and `0 * -inf` is `nan` in IEEE float, not 0.
`noul` questions always have exactly 2 options, so they never hit a padded
slot and never showed the bug; `choice` and `score` items have varying
option/level counts, so a batch almost always mixes items with different
real counts, padded to the widest one — which is every real `choice`/
`score` batch. The `NaN` poisoned the whole NLL, which poisoned the grid
search's `argmin`, which landed on index 0 of the search grid (0.25) every
time — a coincidence that looked like a fit.

Fix (`openjev/train.py`, commit `46ca86a`): zero the log-softmax at masked
slots before multiplying by target, instead of relying on `0 * -inf` to
cancel out. Covered by a new regression test
(`tests/test_calibrate.py::test_temperature_fitting_handles_mixed_option_counts`)
using items with mixed option counts — every prior calibration test
fixture happened to use one fixed option count per question, which is why
this slipped through.

All three published checkpoints were **refitted in place** with the fixed
code, reusing the exact same calibration split each was originally trained
with (verified: the recomputed slice sizes matched the original notes
exactly for every seed and type — 181/237/182, 202/222/176, 191/218/191).
Fitted temperatures moved from the 0.25 floor to real values:

| Seed | `choice` | `score` | `noul` (unaffected by the bug) |
|---|---|---|---|
| 0 | 1.102 (was 0.25) | 1.029 (was 0.25) | 0.848 |
| 1 | 1.014 (was 0.25) | 1.000 (was 0.25) | 0.786 |
| 2 | 1.123 (was 0.25) | 1.027 (was 0.25) | 1.007 |

`nll_after ≤ nll_before` for every type on every seed now (previously
`NaN` for `choice`/`score`), confirming the fit is real. `noul`'s
temperatures are unchanged from the original run, as expected — it was
never affected.

With real temperatures now in place, expect ECE to land in a believable
range (not yet measured here — see the retraction note above) rather than
whatever a search floor happened to produce. **Treat these checkpoints'
probabilities as directionally meaningful, not as precise enough to gate
fully automated decisions on without a human-reviewed threshold**, pending
the recomputed ECE/Brier numbers.

## What this is not

- Not the large-size (`ModernBERT-large`) checkpoint the design spec's
  "real run" targets for Kaggle's 2×T4 — that hasn't been trained yet.
- Not trained on MASSIVE's large option sets, BoolQ, or AG News.
- Not a production-ready fraud/support-triage model. The side-by-side demo
  (`2_one_pass.py`) will sometimes show this checkpoint getting an
  unambiguous example wrong — that's the measured ~61–68% accuracy showing
  up in a single example, not a bug.

## How to use it

```bash
# extract published_checkpoints.zip to checkpoints/ first
OPENJEV_CHECKPOINT=checkpoints/seed2/published python 2_one_pass.py
```

```python
import openjev
agent = openjev.load("checkpoints/seed2/published")
```
