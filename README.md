# OpenJEV

A non-autoregressive decision model: typed questions about a state go in,
calibrated probability distributions come out, one batched forward call, no
text generated. Inspired by TypeSafe AI's Jev and built following the
open-source reference implementation [laya](https://github.com/NandhaKishorM/laya)
(sequence format, checkpoint layout, data format — credited in `NOTICE`).

It ships next to a baseline that gets the same question the old way: prompt
a frozen causal LM and score the generated tokens. `common.py` is that
baseline; `openjev/` is the decision model.

## Why

A causal LM asked "is this a scam?" produces a token, not a probability
distribution. Coercing one out of it (normalizing a few candidate tokens'
logits) is usually *not* a calibrated belief — it only looks like one.
`openjev` instead reads a bidirectional encoder's hidden states straight
into a small scorer head, trained against a strictly proper scoring rule and
calibrated with temperature scaling, so the numbers it reports are meant to
be checked, not taken on faith. See
`docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md` for the
full design, including the anti-goals this project is built not to violate.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Needs `torch>=2.2`, `transformers>=5.0`, `datasets`, `safetensors`, `pytest`
(`numpy` is test-only). The default backbone is
`answerdotai/ModernBERT-base`, downloaded from the Hub on first use unless
you point `Agent` at a local checkpoint.

## Try it

```bash
TINY=1 python 2_one_pass.py
```

Runs a fraud-review case through both engines — the prompted baseline and
`openjev.Agent` — with a randomly initialized tiny model so nothing
downloads. The output explains, in place, what an untrained distribution
does and doesn't tell you. Drop `TINY=1` to run the real `ModernBERT-base`
backbone (still untrained — no checkpoint has been published yet), or set
`OPENJEV_CHECKPOINT=/path/to/checkpoint` once one exists to see the real
comparison.

`1_vanilla_decoding.py` shows the problem this project exists to fix: why a
generated token's probability isn't a decision probability.

## Use it

```python
import openjev

agent = openjev.load("/path/to/checkpoint")  # or openjev.Agent() for a fresh, untrained backbone

state = "Customer requests a refund to a different card than the one charged."

questions = {
    "is_fraud": {
        "type": "noul",  # yes/no
        "instructions": "Does this request show signs of card-testing fraud?",
    },
    "priority": {
        "type": "choice",
        "instructions": "How urgently should this be reviewed?",
        "criteria": {"low": "", "medium": "", "high": ""},
    },
}

result = agent.predict(state, questions)
result["answers"]["is_fraud"]["probabilities"]   # {"true": 0.87, "false": 0.13}
result["usage"]["forward_passes"]                # 1, however many questions were asked
```

Question types: `choice` (pick one of N labeled options), `score` (ordered
levels — partial credit for being one level off), `noul` (yes/no). One
state, any number of questions, one forward call — `usage` reports the
actual pass count and latency rather than asserting it.

## Train and evaluate

The pieces exist as library functions — `openjev.train.train`,
`fit_temperatures`, `save_checkpoint`, and `openjev.eval.evaluate` — but
there is **no CLI yet** to drive them end to end from a labelled file in one
command. Until that follow-up task lands, training is scripted directly
against these functions; see their docstrings in `openjev/train.py` and
`openjev/eval.py` and the task plan below for the exact contracts.

## Layout

| Path | What it's for |
|---|---|
| `openjev/encode.py` | Typed-question validation, sequence/marker layout, collation, option-budget enforcement |
| `openjev/model.py` | ModernBERT backbone + head + per-marker scorer + per-type temperature |
| `openjev/score.py` | The strictly proper scoring rule training optimizes |
| `openjev/data.py` | Target construction, JSONL/CSV readers, HF dataset loaders |
| `openjev/train.py` | Train loop, resume, temperature calibration, checkpoint I/O |
| `openjev/eval.py` | ECE, Brier, random baseline, latency |
| `openjev/__init__.py` | `Agent` — the public `predict`/`load` interface |
| `common.py` | The baseline: a prompted causal LM scored by a trie over its generation |
| `docs/superpowers/specs/` | The binding design spec |
| `docs/superpowers/plans/` | The implementation plan this was built from |

## Tests

```bash
pytest -q -W error
```
