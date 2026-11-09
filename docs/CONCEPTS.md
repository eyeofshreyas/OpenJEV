# Decision models, from first principles

Study notes for OpenJEV. What we are building, why, and the ideas it rests on.
Read top to bottom; the interview questions at the end are the same material backwards.

---

## 1. The problem

You have a fraud case and you need three answers out of a model:

```
priority: HIGH | LOW
decision: FRAUD | FRAUD_REVIEW | CLEAR
escalate: true | false
```

The ordinary way is to ask a language model for JSON. It writes `{`, then `"`, then `prior`,
then `ity`, one token at a time, and you hope the whole thing parses. `1_vanilla_decoding.py`
in this repo does exactly that and then shows you the four things wrong with it.

Say you keep each token's probability as it comes out and multiply the ones that wrote a value.
It looks like a confidence. It is not, for four separate reasons:

1. **It normalises over the wrong set.** The softmax that produced `"HIGH"` ran over the whole
   vocabulary — 150,000 tokens — not over `{HIGH, LOW}`. The leftover probability mass went to
   ` HIGH`, `High`, `HI`, `"`, a newline, a Chinese character. You wanted P(HIGH | case) and you
   got P(this exact spelling next | text so far).
2. **It is not length-normalised.** `FRAUD_REVIEW` is more tokens than `CLEAR`, so it
   multiplies more numbers below 1 and looks less likely for reasons of spelling.
3. **It depends on field order.** `escalate` was written after `decision`, so it is conditioned
   on it. Swap the field order in the schema and the number changes.
4. **The alternatives were never scored.** `CLEAR` was never evaluated. You have no distribution,
   just the path the greedy decode happened to walk.

Add the practical failures: the output can be invalid JSON, it can contain a label that is not
in your enum, and it costs one forward pass per token.

**A probability you cannot compare to anything is not a decision.** That is the whole motivation.

---

## 2. System 1 and System 2

Borrowed from Kahneman. System 2 is slow, deliberate, verbal reasoning. System 1 is fast,
automatic judgement — you see a face and know it is angry before you could explain why.

A chat LLM is a System 2 machine pressed into System 1 work. Most production "AI" calls are not
essays; they are judgement calls: *which team handles this ticket, is this spam, how urgent is
this, does this answer the question, should a human look at it.* High volume, tiny output, and
you need the answer and how sure the model is.

A **decision model** (TypeSafe AI's Jev, September 2026, named the category "System One models")
is built for that job only. Unstructured state in, typed probabilistic decisions out. No prose,
ever.

The consequences are structural, not incremental:

| | Chat LLM asked for JSON | Decision model |
|---|---|---|
| Forward passes | one per output token | one per decision |
| Invalid output | possible | impossible — the output space *is* your option set |
| Label outside your enum | possible | impossible |
| Probability normalised over | the vocabulary | the allowed answers |
| Alternatives scored | no | all of them |
| Output cost | you pay per token | nothing is generated, so output is free |
| Speed | baseline | 40–200x faster (Jev's published figure); laya reports ~33 ms |

"Never hallucinates" is not a training claim. It is a claim about the shape of the output space:
there is no way to emit a token that is not one of your options, because it does not emit tokens.

---

## 3. The interface

Two things go in: a **state** (the text, or JSON, or name-value pairs — whatever the case is) and
a set of **questions**. Every question has a type.

```python
state = "Billed twice for March. Refund the duplicate today or we cancel."

questions = {
  "department":  {"type": "choice", "instructions": "Which department handles this?",
                  "criteria": {"billing": "invoices, payments, refunds",
                               "technical": "bugs, outages, errors",
                               "other": "everything else"}},
  "urgency":     {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["not urgent", "soon", "blocking"]},
  "churn_risk":  {"type": "noul", "instructions": "Does the sender threaten to leave?"},
}
```

Out comes, per question, a distribution over that question's allowed answers:

```python
{"choice": "billing",
 "probabilities": {"billing": 0.91, "technical": 0.04, "other": 0.05}}
```

**The three types:**

- **`choice`** — named options, each with a description. Softmax over the options.
- **`score`** — *ordered* levels, described. Softmax over the levels; the reported score is the
  expected level, so you get 1.7 rather than a bare "level 2". Order matters, and the loss knows
  it (see §7).
- **`noul`** — Bernoulli. A statement, and P(true). It is the two-option case of `choice` with a
  different read-out convention.

Two things to notice, because they are the design:

- **The options are supplied at call time**, as text with descriptions. They are not a fixed
  label set baked into the weights. That is what makes the model general: a new question with new
  options needs no retraining.
- **Many questions, one state.** The state is read once; the questions are answered off it.

---

## 4. The architecture

One sequence per (state, question):

```
[CLS] choice question: Which department handles this? [SEP]
      [MASK] billing: invoices, payments, refunds
      [MASK] technical: bugs, outages, errors
      [MASK] other: everything else [SEP]
      Billed twice for March. Refund the duplicate today or we cancel. [SEP]
```

- A **`[MASK]` marker immediately before each option's text.** These are the read-out positions.
- A **bidirectional encoder** (ModernBERT) runs over the whole sequence.
- The **question type** is added as a learned embedding to every hidden state.
- A small **transformer head** (2 layers, no positional encoding) refines the hidden states.
- A **scorer MLP** — `LayerNorm → Linear(d,d) → GELU → Linear(d,1)` — maps each marker's hidden
  state to **one scalar logit**.
- **Softmax over that question's markers** is the answer, masked to the real option count.

That is the whole mechanism. Three things fall out of it, and this is the part worth being able
to explain:

**Why a marker per option, instead of a classification head?** A classification head has a fixed
output width, so it can only ever answer one question with one label set. A marker reads *a span
of text*, so the number of options and their wording are inputs, not architecture. One model,
any question. It is also why the option descriptions do real work — the marker is attending to
them, so `"billing: invoices, payments, refunds"` is more informative than `"billing"`.

**Why bidirectional?** A marker must see the state to judge its option — and in a causal model
the markers come before the state in the sequence, or the options come before each other, so
something is always invisible. A bidirectional encoder lets every marker attend to the
instructions, its own option, all the other options, and the entire state, simultaneously. The
answer is a comparison, and you cannot compare against what you cannot see. A left-to-right
model fundamentally cannot do this in one pass; the trie scorer in this repo's `common.py` is the
workaround you need when you only have a causal LM, not a design you would choose.

**Why the probabilities mean something.** The softmax runs over the markers — the allowed answers
— so defect 1 from §1 is gone by construction. There is no spelling, so defect 2 is gone. Every
question reads the same shared state with no ordering between them, so defect 3 is gone. Every
option gets a logit, so defect 4 is gone. All four defects were artefacts of generating text.
Stop generating text and they disappear.

---

## 5. Proper scoring rules

This is the idea that makes the probabilities trustworthy, and the most likely thing to be asked
about.

A **scoring rule** scores a reported distribution `q` against what actually happened, `y`. A rule
is **proper** if reporting your honest belief maximises your expected score, and **strictly
proper** if honesty is the *only* way to maximise it.

Why that matters: it removes the incentive to game. Under a strictly proper rule, a model cannot
improve its score by being more confident than it is, or by hedging. The optimum is the truth.
A model trained to maximise one is trained to report calibrated beliefs — calibration is the
objective, not a post-processing step.

The ones in play:

| Rule | Formula | Note |
|---|---|---|
| Log score | `log q_y` | strictly proper; infinite penalty for confident and wrong, hence the clamp |
| Brier score | `−Σ_k (q_k − y_k)²` | strictly proper, bounded, gentler |
| Spherical | `q_y / ‖q‖` | strictly proper; bounded, stabilises the log term |
| Ranked probability (RPS) | `Σ_k (CDF_q(k) − CDF_y(k))² / (K−1)` | for **ordered** outcomes |

A counter-example to keep handy: **accuracy is not a proper scoring rule.** It only reads the
argmax, so it cannot distinguish 0.51 from 0.99 and gives you no reason to be honest about either.
Neither is "distance to the right answer". That is why you train against a scoring rule and
*report* accuracy, not the other way round.

Our reward, following laya's `proper_reward`:

```
reward = log_score + 0.5 · spherical            (all types)
       − 1.0 · RPS                              (score questions only)
```

**Why RPS only for `score`:** `score` levels are ordered. If the answer is level 3, predicting
level 2 should cost less than predicting level 0. Plain log score treats every wrong level as
equally wrong, because softmax has no idea its classes have an order. RPS compares cumulative
distributions, so being nearly right scores nearly well. For `choice`, options are unordered and
RPS would be nonsense — there is no "nearly billing".

---

## 6. RLCD, and the simplification we make

**RLCD** — reinforcement learning for calibrated decisions. Treat a decision as an action, the
proper scoring rule as the reward, and push up the probability of actions that score well, with a
GRPO-style policy gradient over sampled decisions.

OpenJEV does something simpler on purpose and you should be able to defend it: **we maximise the
same reward directly by gradient descent on the reported distribution, instead of sampling
decisions and estimating a policy gradient.**

The argument: the reward here is a *closed-form, differentiable function of the distribution we
output*. Sampling from the distribution to estimate the gradient of a function you can already
differentiate exactly throws away information and adds variance for nothing. Same objective, same
optimum, lower variance, one function instead of a sampling loop.

When the policy-gradient version earns its place: when the reward is **not** differentiable in
the reported distribution. A cost matrix over confusions where mislabelling fraud as clear costs
50x the reverse; a reward that depends on what happened downstream after the decision was acted
on; a multi-turn trajectory. Then you need the RL machinery, because you can no longer write the
reward as a function of `q` you can backprop through.

Supporting evidence: laya measured their soft-cross-entropy arm at **0.790** against their RLCD
arm at **0.7715** on the same benchmark. The simpler objective won. Our loss sits between the two
— it is soft cross-entropy plus the spherical and RPS terms.

---

## 7. Targets: a belief, not a one-hot

The subtlest trap in the whole project.

A strictly proper rule is maximised by reporting your true belief **when the target is a belief.**
Train against one-hot labels and the optimum is a one-hot output: the model is rewarded for
pushing confidence to 1.0 on the training answer. You get a confident liar — exactly the thing
the proper scoring rule was supposed to prevent. The rule is still proper; your target just
asserts certainty you do not have.

Three ways out, in increasing quality:

1. **Label smoothing** — move 10% of the weight off the answer. A blunt belief instead of a
   certainty. Cheap, and all you can do with ordinary labelled data.
2. **Soft targets from annotator disagreement** — if 7 of 10 humans said billing, the target
   *is* 0.7. The best targets, and the rarest.
3. **Teacher distributions** — a strong model's full output distribution, not its argmax. This is
   what the `typed-decisions` dataset ships, and what we train on where we have it. The honest
   caveat: on that data we are partly imitating the teacher's beliefs, including its mistakes.

Laya's own phrasing: the loop is only as good as the targets.

---

## 8. Calibration

**Calibrated** means: of all the times the model says 0.8, it is right about 80% of the time.
Independent of accuracy. A model can be accurate and badly calibrated (right 90% of the time,
always says 0.99) or inaccurate and well calibrated (right 60% of the time, and says 0.6).

For routing work, calibration is the valuable property, because the whole point is to gate on it:
*answer automatically above 0.9, send the rest to a human.* That gate is only as good as the
number it reads.

**ECE — expected calibration error.** Bin the predictions by confidence (15 bins), and in each
bin compare average confidence to actual accuracy. ECE is the weighted mean of the gaps. 0 is
perfect. It is the standard number, and it has known holes — it depends on the binning and only
looks at the top prediction — so report Brier alongside it.

**Temperature scaling.** One scalar `T`, divide the logits by it before the softmax. `T > 1`
softens an overconfident model, `T < 1` sharpens an underconfident one. Fit it by minimising
negative log-likelihood on held-out data. Two facts to know:

- It **cannot change accuracy at all**. Dividing every logit by the same positive number does not
  change which is largest. It only moves the probabilities. So it is free in accuracy terms.
- It must be fitted on data the training never saw, and **the metrics must come from a third
  slice that neither training nor temperature fitting touched.** Fit the temperature and report
  ECE on the same rows and you have measured how well you fitted that slice.

Hence **train / calibration / test**, and we fit one temperature per question type, because
`choice`, `score` and `noul` are miscalibrated differently.

---

## 9. The traps

Things that go wrong in practice, all of them documented by laya at real cost.

**Option position bias.** In this sequential layout, *where* an option is listed changes its
logit. Five identical options spread by 3.59 logits. The fixes: a parallel layout where every
option starts at the same position and options cannot attend to each other (laya's answer), or
shuffle option order during training so the model learns not to care (ours, for v1, with the
spread measured so we know whether it was enough).

**The head budget.** The options compete with the state for a fixed token budget. 60 labels do
not fit in 192 tokens, so each label gets ~3 tokens and labels become indistinguishable — laya
scores 0.425 on Banking77's 77 labels for exactly this reason while Jev, which takes 255 options,
scores 0.870. Two options truncated into the *same* token span make a question unanswerable while
looking fine from outside, so that is detected and rejected loudly. A long state silently losing
tokens is the same class of bug, so truncation is reported rather than inferred.

**The `ln K` plateau.** Train a fresh head from a raw backbone and it can sit at uniform output
over K options — loss exactly `ln K` — and never leave. Laya measured 9 of 15 seeds escaping and
6 never escaping within 32 epochs, on identical data and config. Two consequences: run multiple
seeds and report how many escaped, and run a **positive control** first — prepend a tag to the
state that spells out the answer and confirm the model nails it. If the control fails, the
plumbing or the budget is broken. If the control passes and the task does not move, the task is
hard or the targets are bad. Without the control, everything sitting at `ln K` looks identical.

**Optimizer updates, not epochs.** `updates = epochs × ceil(ceil(items / micro_batch) / grad_accum)`.
A small dataset at 4 epochs may spend 72 updates where a large one spends 352. Converging is
about updates, so a small dataset needs more epochs, not the same number.

**bf16 is not available below compute capability 8.** A T4 is 7.5, so Kaggle runs fp16, which
with a log- and norm-heavy loss overflows. The scoring rule computes in fp32 regardless of
autocast, with a gradient scaler and grad-norm clipping.

---

## 10. Numbers worth remembering

| | |
|---|---|
| Jev input price | $0.042 / M tokens, output free (nothing is generated) |
| Jev speed | 40–200x frontier LLMs |
| laya latency | ~33 ms a decision, T4 |
| laya backbone | ModernBERT-large 421M (English), mmBERT-base ~150M (multilingual) |
| typed-decisions, random baseline | 0.318 |
| laya base, zero-shot | 0.36 — **near chance** |
| laya fine-tuned | 0.766 |
| Jev published | 0.727 |
| teacher self-agreement ceiling | 0.735 |
| laya 149M specialist, raw start | 0.646 ← **our target band** |
| soft-CE vs RLCD arm | 0.790 vs 0.7715 |

The single most important line: **0.36 zero-shot, 0.766 fine-tuned.** The architecture is not
the product. The training is the product. An untrained decision model is near chance.

---

## 11. Interview questions

**Why is a generated-JSON probability not a decision probability?**
Four reasons: it normalises over the vocabulary instead of the allowed answers; it multiplies
per-token probabilities without length normalisation so long labels look unlikely; it is
conditioned on the fields already written so field order changes it; and the alternatives are
never scored, so there is no distribution. All four are artefacts of generating text.

**How can one model handle option sets it has never seen?**
The options are text in the input, and each gets a `[MASK]` marker whose hidden state is scored
to one logit. The output width is the number of markers, so it varies per call. A fixed
classification head cannot do this; it has one output layer for one label set.

**Why does the encoder have to be bidirectional?**
A marker's logit is a judgement of its option against the state and against the other options.
In a causal model something is always behind the cursor and therefore invisible. Bidirectional
attention lets every marker see the instructions, every option and the whole state in one pass.

**What is a strictly proper scoring rule and why train against one?**
A scoring rule whose expected value is maximised *only* by reporting your true belief. It removes
the incentive to be more or less confident than you are, which makes calibration the training
objective rather than a post-hoc fix. Log, Brier and spherical are strictly proper; accuracy is
not, because it only reads the argmax.

**Why RPS for `score` and not for `choice`?**
`score` levels are ordered, so a near miss should cost less than a far one. RPS compares
cumulative distributions and does exactly that. `choice` options are unordered — there is no
"nearly billing" — so a CDF over them is meaningless.

**You train on one-hot labels. What breaks?**
The optimum of a proper scoring rule against a one-hot target is a one-hot output, so training
drives confidence to the ceiling and calibration has to be rescued entirely by post-hoc
temperature. Fixes, in order of quality: label smoothing, annotator-disagreement soft targets,
teacher distributions.

**What is ECE and what does temperature scaling do to it?**
ECE bins predictions by confidence and averages the gap between confidence and accuracy per bin.
Temperature scaling divides the logits by one fitted scalar: it can substantially reduce ECE and
it cannot change accuracy at all, because it preserves the argmax. It must be fitted on held-out
data, and the reported ECE must come from a third slice.

**Why three data splits instead of two?**
The temperature is a parameter fitted on data. Fitting it on the test slice and reporting ECE
from that slice measures how well you fitted it, not how calibrated the model is. Train fits the
weights, calibration fits the temperature, test produces every published number.

**Your loss isn't really RL. Defend that.**
The reward is a closed-form differentiable function of the distribution we output, so a sampled
policy gradient estimates a gradient we can compute exactly — strictly more variance for no
information. Same objective, same optimum, far simpler. The policy-gradient version earns its
place when the reward is not differentiable in the reported distribution: a confusion cost
matrix, a downstream outcome, a multi-turn trajectory. Also, the simpler arm measured better:
0.790 soft-CE against 0.7715 RLCD.

**Why is output free?**
Because nothing is generated. The model does one encoder pass and reads logits off existing
positions; there is no decode loop, so there are no output tokens to bill. The price list is a
consequence of the architecture.

**How would you know your training loop was broken rather than the task being hard?**
Positive control: prepend a tag to the state that spells the answer, and check the model reaches
near-perfect accuracy. If it cannot learn a trivially present signal, the plumbing or the token
budget is broken. Then overfit ~60 rows deliberately. Then multiple seeds, because a raw head can
sit at `ln K` forever and one dead seed is not evidence of a bug.

**When would you not use a decision model?**
When you need prose, open-ended extraction, or an answer whose option set you cannot enumerate.
Decision models also struggle with numbers, dates and adversarial text, and offer no
interpretability beyond the score. The common production pattern is both: a decision model gates,
and routes the genuinely hard cases to an expensive LLM.

---

## Glossary

**System 1 model / decision model** — unstructured state in, typed probabilistic decisions out,
no text generation. **Non-autoregressive** — output does not condition on previously generated
output; no decode loop. **Marker** — the `[MASK]` token before an option whose hidden state is
scored. **`noul`** — Bernoulli question, returns P(true). **Proper scoring rule** — a scoring rule
maximised by honest reporting; strictly proper if honesty is the unique maximum. **RPS** — ranked
probability score, a proper rule for ordered outcomes. **RLCD** — reinforcement learning for
calibrated decisions. **Calibration** — agreement between stated confidence and observed
accuracy. **ECE** — expected calibration error. **Temperature scaling** — one fitted scalar
divisor on the logits; changes probabilities, never the argmax. **`ln K` plateau** — a training
failure where the head outputs uniform over K options and the loss sits at `ln K`.
**Head budget** — the token allowance the question and its options get before the state takes
the rest.

---

Sources: [Jev / System One announcement](https://simonwillison.net/2026/Sep/21/jev/) ·
[Jev explained](https://www.datacamp.com/blog/system-one-models-jev) ·
[laya](https://github.com/NandhaKishorM/laya) (Apache-2.0; architecture, benchmarks and failure
modes in `laya/common.py`, `docs/finetune.md`, `BENCHMARKS.md`) ·
`docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md` for what we are building.
