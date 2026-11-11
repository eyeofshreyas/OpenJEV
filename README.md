# OpenJEV

OpenJEV answers yes/no and multiple-choice questions about a piece of text,
and tells you how confident it is — as real probabilities, not a guess
dressed up to look like one.

You give it a **state** (a short description of a situation) and one or more
**typed questions** about that state. It gives back a probability for each
possible answer. For example:

- State: *"Customer requests a refund to a different card than the one charged."*
- Question: *"Does this show signs of card-testing fraud?"* (yes/no)
- Answer: `{"true": 0.87, "false": 0.13}`

All questions about one state are answered together in a single pass through
the model — no matter how many questions you ask.

## Why this exists

The usual way to get an AI model to answer a yes/no question is to prompt a
chat-style language model and read off the one word it generates ("yes" or
"no"). You can turn the model's internal scores for those words into
something that *looks* like a probability, but it usually isn't one — it's
rarely reflective of the model's actual confidence, just a number pulled
from a process that wasn't built to produce one.

OpenJEV does it differently: instead of generating text, it feeds the
question directly into a *bidirectional encoder* (a type of model that reads
the entire input at once, rather than one word at a time) and reads the
answer straight out of its internal state through a small dedicated scoring
head. That head is trained specifically to produce calibrated probabilities
— using a *strictly proper scoring rule* (a training objective that only
rewards the model for reporting its honest confidence, never for
overstating or underselling it) and *temperature scaling* (a calibration
step that corrects how over- or under-confident the raw output is). The
result is a number meant to be checked against real outcomes, not taken on
faith.

This is inspired by TypeSafe AI's Jev, and follows the design of the
open-source project [laya](https://github.com/NandhaKishorM/laya) (sequence
format, checkpoint layout, data format — credited in `NOTICE`).

The repo also ships a baseline that answers the same questions the old way
— by prompting a frozen chat model and scoring the text it generates — so
you can compare the two approaches directly. `common.py` is that baseline;
`openjev/` is the calibrated decision model.

For the full design, including what this project deliberately avoids doing,
see `docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

This installs `torch`, `transformers`, `datasets`, `safetensors`, and
`pytest`. The model defaults to `answerdotai/ModernBERT-base`, a small
bidirectional encoder downloaded automatically the first time you run it
(unless you point it at your own trained checkpoint instead).

## Try it

```bash
TINY=1 python 2_one_pass.py
```

This runs one example case through both engines — the old prompt-and-decode
baseline, and OpenJEV's `Agent` — using a tiny, randomly-initialized model so
nothing needs to download. Because the model hasn't been trained, its
answers will be meaningless, but the script explains that in its output, so
you can see *what a trained model's answer would look like* without waiting
on a download.

- Drop `TINY=1` to use the real `ModernBERT-base` model (still untrained on
  its own).
- A real trained checkpoint now exists: `checkpoints/seed2/published` (see
  `docs/MODEL_CARD.md` for what it was trained on and how accurate it
  actually is — 67.8% on held-out test cases, well above the 31.75% random
  baseline, but still wrong often enough that this is a research checkpoint,
  not a production one). Point at it with:

  ```bash
  OPENJEV_CHECKPOINT=checkpoints/seed2/published python 2_one_pass.py
  ```

  The checkpoint isn't in git (binary weights, ~650 MB per seed) — extract
  it from `published_checkpoints.zip` into `checkpoints/` first, or point
  `OPENJEV_CHECKPOINT` at your own trained checkpoint directory.

Prefer a visual interface? `playground/` is a small web app (FastAPI
backend, React frontend) for the same thing — type in a state and its
questions, click ask, and see the probabilities as bars. See
`playground/README.md` to run it.

`1_vanilla_decoding.py` demonstrates the problem this project exists to fix:
why the probability of a generated word isn't the same thing as a decision
probability.

## Use it

```python
import openjev

# Point at a trained checkpoint, or call Agent() for an untrained one to explore the API
agent = openjev.load("checkpoints/seed2/published")

state = "Customer requests a refund to a different card than the one charged."

questions = {
    "is_fraud": {
        "type": "noul",  # yes/no question
        "instructions": "Does this request show signs of card-testing fraud?",
    },
    "priority": {
        "type": "choice",  # pick one labeled option
        "instructions": "How urgently should this be reviewed?",
        "criteria": {"low": "", "medium": "", "high": ""},
    },
}

result = agent.predict(state, questions)
result["answers"]["is_fraud"]["probabilities"]   # {"true": 0.87, "false": 0.13}
result["usage"]["forward_passes"]                # how many model passes this cost — 1, regardless of question count
```

There are three question types:

- **`noul`** — a yes/no question.
- **`choice`** — pick one of several labeled options.
- **`score`** — an ordered scale (e.g. low/medium/high), where being one
  level off still counts as partially correct.

You can ask any number of questions about one state, and it still costs a
single forward pass. `result["usage"]` reports the real pass count and
timing, rather than just claiming "one pass" without proof.

## Train and evaluate

The building blocks for training exist as library functions —
`openjev.train.train`, `fit_temperatures`, `save_checkpoint`, and
`openjev.eval.evaluate` — but there's **no command-line tool yet** to run
all of them end-to-end from a labeled data file. Until that exists, training
is driven directly by calling these functions in a script. See their
docstrings in `openjev/train.py` and `openjev/eval.py` for exact usage.

`scripts/train_base.py` is the script that produced the checkpoints in
`checkpoints/` — `answerdotai/ModernBERT-base` on the `typed-decisions`
dataset, three seeds, auto-sized to whatever GPU it finds. Full numbers,
caveats, and what it was and wasn't trained on: `docs/MODEL_CARD.md`.

## Project layout

| Path | What it's for |
|---|---|
| `openjev/encode.py` | Validates questions and lays out the input sequence the model reads |
| `openjev/model.py` | The model itself: encoder + scoring head + per-question-type calibration |
| `openjev/score.py` | The training objective that teaches the model to be calibrated |
| `openjev/data.py` | Loads and formats training data (JSONL, CSV, or Hugging Face datasets) |
| `openjev/train.py` | The training loop, checkpoint saving/loading, calibration fitting |
| `openjev/eval.py` | Measures how calibrated and accurate the model actually is |
| `openjev/__init__.py` | `Agent` — the main `predict`/`load` interface described above |
| `common.py` | The baseline: a prompted chat model scored on its generated text |
| `scripts/train_base.py` | The run that produced `checkpoints/` |
| `checkpoints/` | Trained checkpoints (not in git — see `docs/MODEL_CARD.md`) |
| `docs/MODEL_CARD.md` | What the published checkpoints were trained on, and how accurate they actually are |
| `docs/superpowers/specs/` | The full design document |
| `docs/superpowers/plans/` | The implementation plan this was built from |
| `playground/` | The web app for trying `Agent.predict` interactively |

## Running tests

```bash
pytest -q -W error
```
