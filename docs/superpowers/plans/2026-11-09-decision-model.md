# OpenJEV Decision Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train our own non-autoregressive decision model — typed questions about a state in, calibrated probability distributions out, one forward pass per decision.

**Architecture:** A bidirectional ModernBERT encoder reads `[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`. A 2-layer transformer head plus a scorer MLP turns each `[MASK]` marker's hidden state into one logit; softmax over a question's markers is the answer. Training maximizes a strictly proper scoring rule, so calibration is the objective rather than a post-hoc fix. One temperature per question type is fitted afterwards on a slice training never saw.

**Tech Stack:** Python 3.10+ (dev env 3.14.8), PyTorch 2.15, transformers 5.22, `datasets`, `safetensors`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-openjev-decision-model-design.md`

**Scope:** This plan covers the model, training, calibration, evaluation and the `Agent` API — spec milestones 1 through 6, plus the baseline-parity change to `common.py`. The FastAPI + React playground (spec milestone 7) is a separate plan, written once `Agent.predict` exists.

## Global Constraints

- `max_len` 1024, `head_max_len` 256, per-option truncation 48 tokens. These are the spec's figures, not the library defaults, and no task may quietly change them.
- Reward: `log_score + 0.5 * spherical`, minus `1.0 * RPS` for `score` questions only. Log floor `-9.21`.
- **The scoring rule computes in fp32 regardless of autocast.** It takes `log` of clamped probabilities and divides by `‖q‖`; fp16 overflows it.
- Precision by device: bf16 autocast on compute capability >= 8, fp16 autocast + `GradScaler` below it (a T4 is 7.5 and has no bf16).
- Hard labels enter with label smoothing 0.1. Gold distributions enter as given.
- Three slices: train fits weights, calibration fits temperature, test produces every published number. A type with fewer than 10 calibration items reports its temperature as unfitted (1.0) and warns.
- Question types and their integer ids: `{"choice": 0, "score": 1, "noul": 2}`.
- Checkpoint layout: `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`, `questions.json`, `checkpoint_latest/`. The config filename is laya's so `laya.load` can open our output.
- Dependencies: `torch`, `transformers`, `datasets`, `safetensors`, and `pytest` for development. Nothing else without a stated reason.
- Licence Apache-2.0 with a `NOTICE` crediting laya.
- Every caller-supplied string is stripped of the tokenizer's mask token before encoding.
- Target accuracy band is laya's **0.646** 149M specialist row, not their 0.766 warm-started row.

## Review Focus

Input classes the spec implies but no task's happy path exercises. Each has its test assigned to the task that owns the code.

1. **Caller text containing the literal mask token** forges a marker and silently corrupts the read-out positions. Expect it stripped, in the state and in every option and instruction. → Task 2.
2. **A state longer than the room left after the head** loses evidence. Expect the loss reported to the caller, never inferred or silent. → Task 2 and Task 12.
3. **Two options that truncate to the same token span** make a question unanswerable while looking fine from outside. Expect a loud rejection naming the question. → Task 2.
4. **A non-finite loss under fp16** must abort the run, not train through it and waste five Kaggle hours. → Task 9.
5. **An empty or whitespace-only state** must not produce a confident decision about nothing. Expect a rejection. → Task 1.

---

### Task 1: Question validation and option rendering

**Files:**
- Create: `openjev/encode.py`, `openjev/__init__.py` (empty for now), `tests/test_encode.py`
- Create: `pytest.ini` with `testpaths = tests`
- Modify: `requirements.txt` (add `datasets`, `pytest`)

**Interfaces:**
- Produces: `QTYPES: dict[str, int]`; `validate_question(qid: str, q: dict) -> None` raising `ValueError`; `validate_state(state) -> str` raising `ValueError`; `render_options(q: dict) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from openjev.encode import QTYPES, render_options, validate_question, validate_state

def test_qtypes_ids():
    assert QTYPES == {"choice": 0, "score": 1, "noul": 2}

def test_choice_renders_label_and_description():
    q = {"type": "choice", "instructions": "i", "criteria": {"billing": "invoices, refunds", "tech": ""}}
    assert render_options(q) == ["billing: invoices, refunds", "tech"]

def test_score_renders_numbered_levels():
    q = {"type": "score", "instructions": "i", "criteria": ["low", "high"]}
    assert render_options(q) == ["level 0: low", "level 1: high"]

def test_noul_renders_false_then_true():
    opts = render_options({"type": "noul", "instructions": "i"})
    assert len(opts) == 2 and opts[0].startswith("false") and opts[1].startswith("true")

def test_structured_criterion_renders_as_json():
    q = {"type": "choice", "instructions": "i", "criteria": {"a": {"k": 1}, "b": "x"}}
    assert render_options(q)[0] == 'a: {"k": 1}'

@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "i", "criteria": {"only": "one"}},
    {"type": "choice", "instructions": "i", "criteria": {}},
    {"type": "score", "instructions": "i", "criteria": ["only"]},
    {"type": "nope", "instructions": "i", "criteria": {"a": "", "b": ""}},
    {"type": "choice", "instructions": "", "criteria": {"a": "", "b": ""}},
])
def test_invalid_questions_rejected(q):
    with pytest.raises(ValueError, match="department"):
        validate_question("department", q)

def test_choice_option_cap():
    q = {"type": "choice", "instructions": "i", "criteria": {str(i): "" for i in range(256)}}
    with pytest.raises(ValueError):
        validate_question("big", q)

@pytest.mark.parametrize("state", ["", "   \n\t ", None])
def test_empty_state_rejected(state):
    with pytest.raises(ValueError, match="state"):
        validate_state(state)

def test_state_accepts_str_dict_list():
    assert validate_state("hi") == "hi"
    assert "k" in validate_state({"k": "v"})
    assert "a" in validate_state(["a", "b"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_encode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'openjev'`

- [ ] **Step 3: Implement the four names in `openjev/encode.py`**

`render_options` returns option text in label order: `choice` as `"label: description"` or bare `"label"` when the description is empty or `None` (0 and `False` are legitimate descriptions and must survive); `score` as `"level {i}: {text}"`; `noul` as a false-then-true pair with defaults `"false: no, the statement does not hold"` / `"true: yes, the statement holds"`. A non-string criterion is rendered with `json.dumps(..., ensure_ascii=False, separators=(", ", ": "), default=str)`, not `str()`.

`validate_question` requires a known type, a non-empty `instructions`, and 2–255 options with unique non-empty labels; `score` requires a list of 2+ levels. Every message names `qid`. `validate_state` serializes `str` / `dict` / `list` and rejects anything that serializes to only whitespace.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_encode.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m pip install pytest datasets
git add openjev/ tests/ pytest.ini requirements.txt
git commit -m "feat: validate typed questions and render their options"
```

---

### Task 2: Sequence building, budgets and truncation

**Files:**
- Modify: `openjev/encode.py`
- Modify: `tests/test_encode.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: Task 1's `render_options`, `validate_question`, `validate_state`
- Produces (test support): `tests/conftest.py` holding the `tok` fixture — `AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")`, module scope. Later tasks add their shared fixtures to this one file rather than redefining them.
- Produces: `MAX_LEN = 1024`, `HEAD_MAX_LEN = 256`, `OPTION_MAX_TOKENS = 48`; `strip_mask(tok, text: str) -> str`; `@dataclass Encoded(ids: list[int], markers: list[int], qtype: int, n_options: int, state_tokens: int, state_tokens_used: int)` with a `truncated` property; `build_head(tok, q, head_max_len=HEAD_MAX_LEN) -> tuple[list[int], list[int]]`; `build_sequence(tok, state, q, qid="q", max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, state_ids=None) -> Encoded`; `OptionCollision(ValueError)`

- [ ] **Step 1: Write the failing tests**

The `tok` fixture goes in `tests/conftest.py` so every later task shares it.

```python
def test_one_marker_per_option(tok):
    q = {"type": "choice", "instructions": "which", "criteria": {"a": "x", "b": "y", "c": "z"}}
    enc = build_sequence(tok, "some state", q)
    assert len(enc.markers) == 3 == enc.n_options
    assert all(enc.ids[m] == tok.mask_token_id for m in enc.markers)

def test_markers_are_strictly_increasing_and_in_range(tok):
    enc = build_sequence(tok, "s", {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]})
    assert enc.markers == sorted(enc.markers)
    assert max(enc.markers) < len(enc.ids) <= 1024

def test_mask_token_in_state_is_stripped(tok):
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    enc = build_sequence(tok, f"before {tok.mask_token} after", q)
    assert len(enc.markers) == 2
    assert sum(1 for i in enc.ids if i == tok.mask_token_id) == 2

def test_mask_token_in_instructions_and_options_is_stripped(tok):
    q = {"type": "choice", "instructions": f"pick {tok.mask_token}",
         "criteria": {"a": f"x {tok.mask_token}", "b": "y"}}
    enc = build_sequence(tok, "s", q)
    assert sum(1 for i in enc.ids if i == tok.mask_token_id) == 2

def test_long_state_is_truncated_and_reported(tok):
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    enc = build_sequence(tok, "word " * 4000, q)
    assert len(enc.ids) <= 1024
    assert enc.truncated is True
    assert enc.state_tokens_used < enc.state_tokens

def test_short_state_is_not_reported_as_truncated(tok):
    enc = build_sequence(tok, "short", {"type": "noul", "instructions": "i"})
    assert enc.truncated is False
    assert enc.state_tokens_used == enc.state_tokens

def test_colliding_options_raise_naming_the_question(tok):
    shared = "identical prefix " * 40
    q = {"type": "choice", "instructions": "i", "criteria": {shared + "a": "", shared + "b": ""}}
    with pytest.raises(OptionCollision, match="dept"):
        build_sequence(tok, "s", q, qid="dept")

def test_head_budget_respected_with_many_options(tok):
    q = {"type": "choice", "instructions": "i",
         "criteria": {f"label{i}": "a description that is fairly long" for i in range(20)}}
    enc = build_sequence(tok, "s", q)
    assert len(enc.markers) == 20
    assert enc.markers[-1] < 256

def test_state_ids_reuse_gives_identical_sequence(tok):
    q = {"type": "noul", "instructions": "i"}
    state_ids = tok("shared state", add_special_tokens=False)["input_ids"]
    assert build_sequence(tok, "shared state", q).ids == build_sequence(tok, None, q, state_ids=state_ids).ids
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_encode.py -v -k "marker or mask or truncat or collid or budget or reuse"`
Expected: FAIL — `ImportError: cannot import name 'build_sequence'`

- [ ] **Step 3: Implement `strip_mask`, `Encoded`, `build_head`, `build_sequence` and `OptionCollision` in `openjev/encode.py`**

`build_head` emits `[CLS] <type> question: <ins> [SEP]` then, per option, `[MASK]` followed by that option's tokens, then `[SEP]`. Record each `[MASK]`'s index as its marker. Truncate each option **at the tokenizer** (`truncation=True, max_length=OPTION_MAX_TOKENS`), not by slicing afterwards. If the options overflow `head_max_len`, cut each to an equal share (minimum 4 tokens) and then clamp the instruction, so markers always survive. Detect collisions on the capped option token tuples **before** assembling the sequence — re-checking the finished sequence cannot work, because the last option's span runs into the state.

`build_sequence` strips the mask token from every caller string, validates, builds the head, gives the state whatever room is left of `max_len` after the head and the closing `[SEP]`, and records `state_tokens` and `state_tokens_used` so `truncated` is a fact rather than a guess.

- [ ] **Step 4: Run the full test file**

Run: `.venv/bin/python -m pytest tests/test_encode.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/encode.py tests/test_encode.py
git commit -m "feat: build decision sequences with marker read-out positions"
```

---

### Task 3: Collate a batch

**Files:**
- Modify: `openjev/encode.py`
- Create: `tests/test_collate.py`

**Interfaces:**
- Consumes: Task 2's `Encoded`
- Produces: `collate(encoded: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]` with keys `input_ids` `[B, L]` long, `attention_mask` `[B, L]` long, `marker_pos` `[B, K]` long, `marker_mask` `[B, K]` bool, `qtype` `[B]` long

- [ ] **Step 1: Write the failing tests**

```python
def test_collate_pads_to_longest_and_masks(tok):
    a = build_sequence(tok, "short", {"type": "noul", "instructions": "i"})
    b = build_sequence(tok, "a much longer state " * 20,
                       {"type": "choice", "instructions": "i", "criteria": {"x": "", "y": "", "z": ""}})
    batch = collate([a, b], tok.pad_token_id)
    L = max(len(a.ids), len(b.ids))
    assert batch["input_ids"].shape == (2, L)
    assert batch["attention_mask"][0].sum().item() == len(a.ids)
    assert (batch["input_ids"][0, len(a.ids):] == tok.pad_token_id).all()

def test_collate_marker_mask_matches_option_counts(tok):
    a = build_sequence(tok, "s", {"type": "noul", "instructions": "i"})
    b = build_sequence(tok, "s", {"type": "choice", "instructions": "i",
                                  "criteria": {"x": "", "y": "", "z": ""}})
    batch = collate([a, b], tok.pad_token_id)
    assert batch["marker_mask"].shape == (2, 3)
    assert batch["marker_mask"][0].tolist() == [True, True, False]
    assert batch["qtype"].tolist() == [2, 0]

def test_collate_marker_pos_points_at_mask_tokens(tok):
    encs = [build_sequence(tok, "s", {"type": "noul", "instructions": "i"})]
    batch = collate(encs, tok.pad_token_id)
    pos = batch["marker_pos"][0, :2]
    assert (batch["input_ids"][0, pos] == tok.mask_token_id).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_collate.py -v`
Expected: FAIL — `cannot import name 'collate'`

- [ ] **Step 3: Implement `collate` in `openjev/encode.py`**

Pad `input_ids` with `pad_id` to the longest sequence and `marker_pos` to the largest option count. Padded marker slots must index a valid position (use 0) with `marker_mask` false, so a gather cannot go out of bounds.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/encode.py tests/test_collate.py
git commit -m "feat: collate encoded decisions into a padded batch"
```

---

### Task 4: The decision model

**Files:**
- Create: `openjev/model.py`, `tests/test_model.py`
- Modify: `tests/conftest.py` (add the `tiny_model` fixture and the `_fake_batch` helper, both used by Tasks 9–12)

**Interfaces:**
- Consumes: Task 3's batch dict, Task 1's `QTYPES`
- Produces: `class DecisionModel(nn.Module)` with `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> torch.Tensor` returning fp32 logits `[B, K]`, and a `temperature` buffer of shape `(3,)`; `build_model(backbone: str | None = None, config=None, head_layers: int = 2) -> DecisionModel`; `probabilities(logits, marker_mask, qtype=None, temperature=None) -> torch.Tensor`; `tiny_config(vocab_size: int) -> ModernBertConfig`

- [ ] **Step 1: Write the failing tests**

`tiny_config` exists so tests never download weights. ModernBERT's config defaults its special-token ids to the real tokenizer's (50281/50282/50283), which breaks `nn.Embedding`'s `padding_idx` on a small vocab, so it must pass `pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2` explicitly.

In `tests/conftest.py`: `tiny_model` is a module-scoped `build_model(config=tiny_config(vocab_size=1000)).eval()`, and `_fake_batch(b, length, k, counts=None)` returns the five tensors `collate` produces, with random ids and `marker_pos` pointing at distinct in-range positions.

```python
def test_logits_shape_and_dtype(tiny_model):
    logits = tiny_model(**_fake_batch(b=2, length=32, k=3))
    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32

def test_probabilities_sum_to_one_over_real_options(tiny_model):
    batch = _fake_batch(b=2, length=32, k=3, counts=[2, 3])
    p = probabilities(tiny_model(**batch), batch["marker_mask"])
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)

def test_masked_options_get_zero_probability(tiny_model):
    batch = _fake_batch(b=1, length=32, k=4, counts=[2])
    p = probabilities(tiny_model(**batch), batch["marker_mask"])
    assert p[0, 2:].max().item() < 1e-6

def test_temperature_softens_without_changing_argmax(tiny_model):
    batch = _fake_batch(b=1, length=32, k=3)
    logits = tiny_model(**batch)
    cold = probabilities(logits, batch["marker_mask"])
    warm = probabilities(logits, batch["marker_mask"], qtype=batch["qtype"],
                         temperature=torch.tensor([5.0, 5.0, 5.0]))
    assert cold.argmax(-1).tolist() == warm.argmax(-1).tolist()
    assert warm.max().item() < cold.max().item()

def test_question_type_changes_the_logits(tiny_model):
    batch = _fake_batch(b=1, length=32, k=3)
    a = tiny_model(**batch)
    batch["qtype"] = batch["qtype"] + 1
    assert not torch.allclose(a, tiny_model(**batch))

def test_gradients_reach_encoder_and_scorer(tiny_model):
    batch = _fake_batch(b=2, length=32, k=3)
    tiny_model.train()
    tiny_model(**batch).sum().backward()
    assert tiny_model.scorer[-1].weight.grad is not None
    assert next(tiny_model.encoder.parameters()).grad is not None
    tiny_model.eval()

def test_identical_options_spread_is_measured(tiny_model, tok, capsys):
    # The spec defers the parallel option layout to v2 "with a measurement attached".
    # This is that measurement: five identical options, so any spread is position alone.
    q = {"type": "choice", "instructions": "pick one",
         "criteria": {f"opt{i}": "exactly the same description" for i in range(5)}}
    enc = build_sequence(tok, "a state", q)
    logits = tiny_model(**collate([enc], tok.pad_token_id))
    spread = (logits.max() - logits.min()).item()
    print(f"identical-option logit spread: {spread:.3f}")
    assert spread >= 0.0  # recorded, not gated: laya measured 3.59 on a trained checkpoint
```

The spread test prints rather than thresholds, because an untrained tiny model's number means nothing. Re-run it after the real training run; that figure decides whether the v2 parallel layout is needed.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL — `No module named 'openjev.model'`

- [ ] **Step 3: Implement `openjev/model.py`**

`DecisionModel.__init__(encoder, head_layers=2)` builds: `type_emb = nn.Embedding(3, d)`; `head = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, batch_first=True, norm_first=True), head_layers)` — no positional encoding, deliberately, so the head cannot learn option order; `scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))`; `register_buffer("temperature", torch.ones(3))`.

`forward` runs the encoder, adds `type_emb(qtype)[:, None, :]` to every hidden state, runs the head with `src_key_padding_mask=~attention_mask.bool()`, gathers the hidden states at `marker_pos`, scores each to one logit, and fills masked slots with `-1e4`. Return `.float()` so the loss is fp32 under any autocast. `probabilities` divides by `temperature[qtype]` when both are given, then softmaxes.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_model.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/model.py tests/test_model.py
git commit -m "feat: add the decision model that scores one logit per marker"
```

---

### Task 5: The proper scoring rule

**Files:**
- Create: `openjev/score.py`, `tests/test_score.py`

**Interfaces:**
- Produces: `proper_reward(p, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21) -> torch.Tensor` of shape `[B]`; `decision_loss(logits, target, qtype, mask) -> torch.Tensor` scalar

- [ ] **Step 1: Write the failing tests**

The first test is the property that justifies the whole design and must be tested, not assumed.

```python
def test_strictly_proper_truth_scores_best():
    truth = torch.tensor([[0.6, 0.3, 0.1]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    qtype = torch.zeros(1, dtype=torch.long)
    honest = proper_reward(truth, truth, qtype, mask).item()
    for other in [[0.34, 0.33, 0.33], [0.9, 0.05, 0.05], [0.1, 0.3, 0.6], [0.6, 0.1, 0.3]]:
        assert proper_reward(torch.tensor([other]), truth, qtype, mask).item() < honest

def test_rps_prefers_a_near_miss_for_score_questions():
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    near = torch.tensor([[0.0, 0.1, 0.8, 0.1]])
    far = torch.tensor([[0.1, 0.1, 0.8, 0.0]])
    score_t = torch.tensor([1]); choice_t = torch.tensor([0])
    assert proper_reward(near, target, score_t, mask) > proper_reward(far, target, score_t, mask)
    assert torch.allclose(proper_reward(near, target, choice_t, mask),
                          proper_reward(far, target, choice_t, mask), atol=1e-6)

def test_confident_and_wrong_is_floored_not_infinite():
    p = torch.tensor([[1e-30, 1.0 - 1e-30]])
    target = torch.tensor([[1.0, 0.0]])
    r = proper_reward(p, target, torch.zeros(1, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))
    assert torch.isfinite(r).all() and r.item() >= -9.21 - 1.0

def test_masked_options_are_ignored():
    mask = torch.tensor([[True, True, False]])
    qtype = torch.zeros(1, dtype=torch.long)
    target = torch.tensor([[0.7, 0.3, 0.0]])
    a = proper_reward(torch.tensor([[0.7, 0.3, 0.0]]), target, qtype, mask)
    b = proper_reward(torch.tensor([[0.7, 0.3, 0.9]]), target, qtype, mask)
    assert torch.allclose(a, b, atol=1e-6)

def test_loss_is_fp32_under_autocast_and_decreases_toward_the_target():
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool); qtype = torch.zeros(1, dtype=torch.long)
    first = decision_loss(logits, target, qtype, mask)
    assert first.dtype == torch.float32
    first.backward()
    better = decision_loss((logits - 0.5 * logits.grad).detach(), target, qtype, mask)
    assert better.item() < first.item()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: FAIL — `No module named 'openjev.score'`

- [ ] **Step 3: Implement `openjev/score.py`**

```python
def proper_reward(p, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    p = (p * mask).float()
    log_score = (target * torch.log(p.clamp_min(1e-12)).clamp_min(log_floor)).sum(-1)
    spherical = (target * p).sum(-1) / p.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * spherical
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        rps = (((p.cumsum(-1) - target.cumsum(-1)) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r
```

`decision_loss` turns logits into probabilities with `probabilities`, then returns `-proper_reward(...).mean()`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_score.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/score.py tests/test_score.py
git commit -m "feat: add the strictly proper scoring rule as reward and loss"
```

---

### Task 6: Targets from labels and from gold

**Files:**
- Create: `openjev/data.py`, `tests/test_targets.py`
- Modify: `tests/conftest.py` (add `CHOICE`, `SCORE`, `NOUL`, used by Tasks 9 and 12)

**Interfaces:**
- Produces: `SMOOTHING = 0.1`; `target_vector(q: dict, expected=None, gold: dict | None = None, smoothing: float = SMOOTHING) -> list[float]`

- [ ] **Step 1: Write the failing tests**

These three go in `tests/conftest.py`; Tasks 9 and 12 import them from there.

```python
CHOICE = {"type": "choice", "instructions": "i", "criteria": {"a": "", "b": "", "c": ""}}
SCORE = {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}
NOUL = {"type": "noul", "instructions": "i"}

def test_hard_choice_label_is_smoothed():
    v = target_vector(CHOICE, expected="b")
    assert v == pytest.approx([0.05, 0.9, 0.05])
    assert sum(v) == pytest.approx(1.0)

def test_smoothing_zero_gives_one_hot():
    assert target_vector(CHOICE, expected="a", smoothing=0.0) == pytest.approx([1.0, 0.0, 0.0])

def test_noul_true_is_the_second_slot():
    assert target_vector(NOUL, expected=True)[1] > target_vector(NOUL, expected=True)[0]
    assert target_vector(NOUL, expected=False)[0] > target_vector(NOUL, expected=False)[1]

def test_fractional_level_splits_between_neighbours():
    v = target_vector(SCORE, expected=1.5, smoothing=0.0)
    assert v == pytest.approx([0.0, 0.5, 0.5])

def test_integer_level_is_smoothed_like_a_label():
    assert target_vector(SCORE, expected=2) == pytest.approx([0.05, 0.05, 0.9])

def test_gold_distribution_passes_through_unsmoothed():
    g = {"probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}}
    assert target_vector(CHOICE, gold=g) == pytest.approx([0.7, 0.2, 0.1])

def test_gold_score_keys_are_level_index_strings():
    g = {"probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
    assert target_vector(SCORE, gold=g) == pytest.approx([0.1, 0.6, 0.3])

def test_gold_is_renormalised_and_missing_options_are_zero():
    assert target_vector(CHOICE, gold={"probabilities": {"a": 1.0, "b": 1.0}}) == pytest.approx([0.5, 0.5, 0.0])

@pytest.mark.parametrize("expected", ["nope", 7, -1, None])
def test_unknown_answers_rejected(expected):
    with pytest.raises(ValueError):
        target_vector(CHOICE, expected=expected)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_targets.py -v`
Expected: FAIL — `cannot import name 'target_vector'`

- [ ] **Step 3: Implement `target_vector` in `openjev/data.py`**

`gold` wins when both are given and is renormalised over the question's options. `expected` builds a one-hot (a fractional `score` level splits linearly between its two neighbours and is **not** smoothed on top, since it is already a distribution), then smoothing moves `smoothing` of the mass uniformly across the other options. A label or level outside the question's options raises, naming the question.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_targets.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/data.py tests/test_targets.py
git commit -m "feat: build training targets from hard labels and gold distributions"
```

---

### Task 7: Items from JSONL and CSV, with skip reasons

**Files:**
- Modify: `openjev/data.py`
- Create: `tests/test_data.py`, `tests/fixtures/cases.jsonl`, `tests/fixtures/tickets.csv`

**Interfaces:**
- Consumes: Task 2's `build_sequence`/`OptionCollision`, Task 6's `target_vector`
- Produces: `@dataclass Item(state: str, qid: str, question: dict, target: list[float])`; `items_from_jsonl(path, smoothing=SMOOTHING) -> tuple[list[Item], collections.Counter]`; `items_from_csv(path, text_column, label_column, question_id="label", instructions=..., smoothing=SMOOTHING) -> tuple[list[Item], dict, Counter]` also returning the generated question; `update_budget(n_items: int, micro_batch: int, grad_accum: int, epochs: int) -> int`; `split_items(items, calib_frac=0.1, seed=0) -> tuple[list[Item], list[Item]]`

- [ ] **Step 1: Write the failing tests**

Fixture `cases.jsonl` holds 4 lines: one `expected` case with a choice and a noul, one `gold` case, one with an empty state, one with a question whose type is unknown.

```python
def test_jsonl_reads_both_label_shapes():
    items, skipped = items_from_jsonl("tests/fixtures/cases.jsonl")
    assert len(items) == 3
    assert {i.qid for i in items} == {"department", "urgent", "team"}
    assert all(abs(sum(i.target) - 1.0) < 1e-6 for i in items)

def test_jsonl_counts_skip_reasons():
    _, skipped = items_from_jsonl("tests/fixtures/cases.jsonl")
    assert skipped["empty_text"] == 1
    assert skipped["invalid_question"] == 1

def test_csv_builds_one_choice_question_from_the_label_column():
    items, question, skipped = items_from_csv("tests/fixtures/tickets.csv",
                                              text_column="body", label_column="department")
    assert question["type"] == "choice"
    assert list(question["criteria"]) == ["billing", "technical"]
    assert len(items) == 4 and skipped["empty_label"] == 1

def test_csv_tolerates_a_byte_order_mark():
    items, _, _ = items_from_csv("tests/fixtures/tickets_bom.csv",
                                 text_column="body", label_column="department")
    assert len(items) == 4

def test_update_budget_matches_the_spec_formula():
    assert update_budget(1100, 8, 8, 4) == 72
    assert update_budget(5600, 8, 8, 4) == 352

def test_split_is_deterministic_and_disjoint():
    items, _ = items_from_jsonl("tests/fixtures/cases.jsonl")
    a1, c1 = split_items(items, calib_frac=0.34, seed=0)
    a2, c2 = split_items(items, calib_frac=0.34, seed=0)
    assert [i.qid for i in a1] == [i.qid for i in a2] and [i.qid for i in c1] == [i.qid for i in c2]
    assert not ({id(i) for i in a1} & {id(i) for i in c1})
    assert len(a1) + len(c1) == len(items)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: FAIL — `cannot import name 'items_from_jsonl'`

- [ ] **Step 3: Implement the readers in `openjev/data.py`**

One `Item` per (case, question). Skip reasons counted, never raised past the reader: `empty_text`, `empty_label`, `invalid_question`, `invalid_target`, `options_collapsed` (from `OptionCollision`), `options_beyond_max_len`. Read the file with `encoding="utf-8-sig"` so an Excel byte-order mark is not part of the first column name. `items_from_csv` derives the choice question from the label column's distinct values in first-appearance order. `update_budget` is `epochs * ceil(ceil(n_items / micro_batch) / grad_accum)`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/data.py tests/test_data.py tests/fixtures/
git commit -m "feat: read labelled decisions from JSONL and CSV"
```

---

### Task 8: Public dataset loaders

**Files:**
- Modify: `openjev/data.py`
- Create: `tests/test_datasets.py`

**Interfaces:**
- Produces: `load_typed_decisions(split: str) -> list[dict]`; `load_massive(split: str, n_options: int = 20, seed: int = 0) -> list[dict]`; `load_boolq(split) -> list[dict]`; `load_ag_news(split) -> list[dict]`; `sample_options(labels: list[str], gold: str, n: int, rng) -> list[str]`. All return the JSONL case shape `{"state": ..., "questions": {...}, "expected"|"gold": {...}}`.

- [ ] **Step 1: Write the failing tests**

`sample_options` is the part worth testing without a download; the loaders get one network test marked `@pytest.mark.network`, deselected by default via `addopts = -m "not network"` in `pytest.ini`.

```python
def test_sample_options_always_includes_gold_and_is_sized():
    rng = random.Random(0)
    labels = [f"l{i}" for i in range(60)]
    opts = sample_options(labels, "l42", 20, rng)
    assert "l42" in opts and len(opts) == 20 and len(set(opts)) == 20

def test_sample_options_is_deterministic_for_a_seed():
    labels = [f"l{i}" for i in range(60)]
    assert sample_options(labels, "l0", 20, random.Random(7)) == sample_options(labels, "l0", 20, random.Random(7))

def test_sample_options_handles_fewer_labels_than_requested():
    assert set(sample_options(["a", "b"], "a", 20, random.Random(0))) == {"a", "b"}

@pytest.mark.network
def test_typed_decisions_test_split_shape():
    cases = load_typed_decisions("test")
    assert len(cases) == 400
    assert sum(len(c["questions"]) for c in cases) == 2000
    assert all("gold" in c for c in cases)

@pytest.mark.network
def test_massive_cases_carry_twenty_options():
    cases = load_massive("validation", n_options=20, seed=0)
    q = next(iter(cases[0]["questions"].values()))
    assert len(q["criteria"]) == 20
    assert cases[0]["expected"][next(iter(cases[0]["questions"]))] in q["criteria"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_datasets.py -v`
Expected: FAIL — `cannot import name 'sample_options'`

- [ ] **Step 3: Implement the loaders in `openjev/data.py`**

Dataset ids: `LocalLLaMA/typed-decisions` (config `all`, splits `train`/`test`, already in our case shape with gold distributions), `AmazonScience/massive` (config `en-US`), `google/boolq`, `fancyzhx/ag_news`. Import `datasets` lazily inside each loader so training and tests that do not touch them never pay for it. `sample_options` returns the gold label plus `n - 1` distractors drawn without replacement, shuffled, so position carries no information about correctness. MASSIVE is the spec's 20-option protocol; a `score` question is only produced by typed-decisions.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_datasets.py -v` then `.venv/bin/python -m pytest tests/test_datasets.py -v -m network`
Expected: PASS for both (the second needs network; record in the commit message if it was skipped)

- [ ] **Step 5: Commit**

```bash
git add openjev/data.py tests/test_datasets.py pytest.ini
git commit -m "feat: load typed-decisions and the public benchmark sets"
```

---

### Task 9: The training loop

**Files:**
- Create: `openjev/train.py`, `tests/test_train.py`
- Modify: `tests/conftest.py` (add `_synthetic_items`)

**Interfaces:**
- Consumes: Tasks 3, 4, 5, 7
- Produces: `@dataclass TrainConfig(epochs=4, micro_batch=8, grad_accum=8, lr_encoder=2e-5, lr_head=1e-4, warmup_frac=0.1, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, shuffle_options=True, seed=0, grad_clip=1.0, calib_frac=0.1)`; `autocast_dtype(device: torch.device) -> torch.dtype | None`; `tag_state(item: Item) -> Item`; `shuffle_options(item: Item, rng: random.Random) -> tuple[Item, list[float]]`; `train(items, model, tok, out_dir, cfg) -> dict`; `NonFiniteLoss(RuntimeError)`

- [ ] **Step 1: Write the failing tests**

```python
def test_autocast_dtype_is_fp16_below_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    assert autocast_dtype(torch.device("cuda")) == torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 6))
    assert autocast_dtype(torch.device("cuda")) == torch.bfloat16

def test_cpu_gets_no_autocast():
    assert autocast_dtype(torch.device("cpu")) is None

def test_tag_state_prepends_the_answer():
    item = Item(state="s", qid="q", question=CHOICE, target=[0.0, 1.0, 0.0])
    assert "b" in tag_state(item).state and tag_state(item).target == item.target

def test_positive_control_is_learned(tiny_model, tok):
    items = [tag_state(i) for i in _synthetic_items(n=40)]
    report = train(items, tiny_model, tok, out_dir=tmp, cfg=TrainConfig(epochs=8, micro_batch=4, grad_accum=1))
    assert report["train_accuracy"] > 0.95

def test_overfits_a_small_set(tiny_model, tok):
    report = train(_synthetic_items(n=60), tiny_model, tok, out_dir=tmp,
                   cfg=TrainConfig(epochs=30, micro_batch=4, grad_accum=1))
    assert report["train_accuracy"] > 0.9

def test_report_states_the_optimizer_budget(tiny_model, tok):
    report = train(_synthetic_items(n=40), tiny_model, tok, out_dir=tmp,
                   cfg=TrainConfig(epochs=2, micro_batch=4, grad_accum=2))
    assert report["optimizer_updates"] == update_budget(report["train_items"], 4, 2, 2)

def test_non_finite_loss_aborts(tiny_model, tok, monkeypatch):
    monkeypatch.setattr("openjev.train.decision_loss", lambda *a, **k: torch.tensor(float("nan")))
    with pytest.raises(NonFiniteLoss):
        train(_synthetic_items(n=8), tiny_model, tok, out_dir=tmp, cfg=TrainConfig(epochs=1, micro_batch=4))

def test_rolling_checkpoint_written_each_epoch(tiny_model, tok, tmp_path):
    train(_synthetic_items(n=8), tiny_model, tok, out_dir=tmp_path, cfg=TrainConfig(epochs=2, micro_batch=4))
    assert (tmp_path / "checkpoint_latest" / "model.safetensors").exists()

def test_shuffle_options_keeps_the_answer_aligned(tok):
    item = Item(state="s", qid="d", question=CHOICE, target=[0.0, 1.0, 0.0])
    for seed in range(5):
        shuffled, target = shuffle_options(item, random.Random(seed))
        labels = list(shuffled.question["criteria"])
        assert labels[target.index(max(target))] == "b"
        assert sorted(labels) == ["a", "b", "c"]

def test_shuffle_options_leaves_score_levels_alone(tok):
    item = Item(state="s", qid="u", question=SCORE, target=[0.0, 0.0, 1.0])
    shuffled, target = shuffle_options(item, random.Random(3))
    assert list(shuffled.question["criteria"]) == list(SCORE["criteria"])
    assert target == item.target
```

`shuffle_options(item, rng) -> tuple[Item, list[float]]` is part of this task's surface: it permutes a `choice` question's options and the target together, and returns `score` and `noul` questions untouched, because their option order carries meaning.

`_synthetic_items(n)` goes in `tests/conftest.py`: a separable task the tiny model can actually learn — two states, two labels, the label word present in the state.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`
Expected: FAIL — `No module named 'openjev.train'`

- [ ] **Step 3: Implement `openjev/train.py`**

AdamW with two parameter groups at `lr_encoder` and `lr_head`, cosine schedule with `warmup_frac` warmup. Hold out `calib_frac` with `split_items` **before** training and report both counts. Print the optimizer budget before the first step. Autocast with `autocast_dtype`; wrap in `GradScaler` when the dtype is fp16; clip grad norm at `grad_clip`; raise `NonFiniteLoss` naming the step when the loss is not finite. With `shuffle_options`, re-encode `choice` questions each epoch with a permuted option order and permute the target to match. Write `checkpoint_latest/` after every epoch. Return a report with `train_items`, `calibration_items`, `skipped`, `optimizer_updates`, `epoch_loss`, `train_accuracy`, `seed`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/train.py tests/test_train.py
git commit -m "feat: train the decision head against the scoring rule"
```

---

### Task 10: Calibration and checkpoints

**Files:**
- Modify: `openjev/train.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Produces: `MIN_CALIBRATION_ITEMS = 10`; `fit_temperatures(model, items, tok, cfg) -> tuple[list[float], dict]` returning the three temperatures and a per-type note; `save_checkpoint(model, tok, out_dir, questions=None, config=None) -> None`; `load_checkpoint(path) -> tuple[DecisionModel, AutoTokenizer, dict]`

- [ ] **Step 1: Write the failing tests**

```python
def test_temperature_reduces_nll_of_an_overconfident_model(tiny_model, tok):
    items = _miscalibrated_items(n=200)
    temps, notes = fit_temperatures(tiny_model, items, tok, TrainConfig())
    assert temps[QTYPES["choice"]] > 1.0
    assert notes["choice"]["nll_after"] < notes["choice"]["nll_before"]

def test_temperature_never_changes_accuracy(tiny_model, tok):
    items = _miscalibrated_items(n=200)
    before = _accuracy(tiny_model, items, tok, temperature=None)
    temps, _ = fit_temperatures(tiny_model, items, tok, TrainConfig())
    assert _accuracy(tiny_model, items, tok, temperature=temps) == before

def test_too_few_items_leaves_temperature_unfitted(tiny_model, tok):
    temps, notes = fit_temperatures(tiny_model, _synthetic_items(n=5), tok, TrainConfig())
    assert temps[QTYPES["choice"]] == 1.0
    assert notes["choice"]["fitted"] is False

def test_checkpoint_round_trip_preserves_logits(tiny_model, tok, tmp_path):
    batch = _fake_batch(b=2, length=32, k=3)
    before = tiny_model(**batch)
    save_checkpoint(tiny_model, tok, tmp_path, questions={"q": CHOICE})
    model2, tok2, cfg = load_checkpoint(tmp_path)
    assert torch.allclose(before, model2(**batch), atol=1e-5)

def test_checkpoint_layout_uses_layas_filenames(tiny_model, tok, tmp_path):
    save_checkpoint(tiny_model, tok, tmp_path, questions={"q": CHOICE})
    for name in ["model.safetensors", "rl_agent_config.json", "questions.json"]:
        assert (tmp_path / name).exists()
    assert (tmp_path / "encoder").is_dir() and (tmp_path / "tokenizer").is_dir()

def test_saved_temperature_survives_the_round_trip(tiny_model, tok, tmp_path):
    tiny_model.temperature.copy_(torch.tensor([1.5, 2.0, 0.8]))
    save_checkpoint(tiny_model, tok, tmp_path)
    model2, _, _ = load_checkpoint(tmp_path)
    assert torch.allclose(model2.temperature, torch.tensor([1.5, 2.0, 0.8]))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_calibrate.py -v`
Expected: FAIL — `cannot import name 'fit_temperatures'`

- [ ] **Step 3: Implement calibration and checkpoint IO in `openjev/train.py`**

Fit one temperature per question type by minimising negative log-likelihood on that type's calibration items — a scalar search over a log-spaced grid in `[0.25, 10]` refined by `torch.optim.LBFGS` on `log T`, clamped to that range. A type with fewer than `MIN_CALIBRATION_ITEMS` items keeps 1.0 and records `fitted: False` with a warning. Write the fitted values into the model's `temperature` buffer so they travel with the checkpoint. `save_checkpoint` writes the head and buffers to `model.safetensors`, the encoder with `save_pretrained` to `encoder/`, the tokenizer to `tokenizer/`, and the config (backbone id, `max_len`, `head_max_len`, `head_layers`, temperatures, notes) to `rl_agent_config.json`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/train.py tests/test_calibrate.py
git commit -m "feat: fit per-type temperatures and save loadable checkpoints"
```

---

### Task 11: Evaluation metrics

**Files:**
- Create: `openjev/eval.py`, `tests/test_eval.py`

**Interfaces:**
- Produces: `ece(confidence, correct, bins=15) -> float`; `brier(probs, targets) -> float`; `random_baseline(cases) -> float`; `evaluate(model, tok, cases, batch_size=16) -> dict` with keys `accuracy`, `accuracy_by_type`, `within_one_level`, `ece`, `ece_by_type`, `brier`, `latency_ms_median`, `latency_ms_p95`, `random_baseline`, `n`

- [ ] **Step 1: Write the failing tests**

```python
def test_ece_of_a_perfectly_calibrated_set_is_near_zero():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    correct = rng.uniform(size=20000) < conf
    assert ece(conf, correct, bins=15) < 0.02

def test_ece_of_a_confidently_wrong_set_is_near_one():
    conf = np.full(1000, 0.99)
    assert ece(conf, np.zeros(1000, dtype=bool)) == pytest.approx(0.99, abs=0.01)

def test_ece_is_hand_computable_on_two_bins():
    conf = np.array([0.1, 0.1, 0.9, 0.9])
    correct = np.array([False, False, True, False])
    assert ece(conf, correct, bins=10) == pytest.approx((0.1 + 0.4) / 2, abs=1e-6)

def test_brier_rewards_the_truth():
    t = np.array([[1.0, 0.0, 0.0]])
    assert brier(np.array([[0.8, 0.1, 0.1]]), t) < brier(np.array([[0.4, 0.3, 0.3]]), t)

def test_random_baseline_is_the_mean_reciprocal_option_count():
    cases = [{"questions": {"a": CHOICE, "b": NOUL}, "expected": {"a": "a", "b": True}}]
    assert random_baseline(cases) == pytest.approx((1 / 3 + 1 / 2) / 2)

def test_within_one_level_is_looser_than_exact(tiny_model, tok):
    report = evaluate(tiny_model, tok, _score_cases(n=50))
    assert report["within_one_level"] >= report["accuracy_by_type"]["score"]

def test_report_carries_latency_and_n(tiny_model, tok):
    report = evaluate(tiny_model, tok, _synthetic_cases(n=20))
    assert report["n"] == 20 and report["latency_ms_median"] > 0
    assert report["latency_ms_p95"] >= report["latency_ms_median"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: FAIL — `No module named 'openjev.eval'`

- [ ] **Step 3: Implement `openjev/eval.py`**

`ece` bins by confidence and returns the count-weighted mean of `|accuracy - mean confidence|` per non-empty bin. `brier` is the mean squared difference between the reported distribution and the target over options. `random_baseline` is the mean of `1 / n_options` across all questions. `evaluate` runs the model over the cases in batches, timing each call, and reports exact accuracy per type plus `within_one_level` for `score` only.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/eval.py tests/test_eval.py
git commit -m "feat: report accuracy, calibration error and latency"
```

---

### Task 12: The Agent API

**Files:**
- Modify: `openjev/__init__.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 4, 10
- Produces: `class Agent` with `__init__(self, path: str | None = None, backbone: str | None = None, device=None)`, `predict(self, state, questions: dict) -> dict`; `load(path) -> Agent`

The answer shape, which Task 13's demo and the playground plan both depend on:

```python
{"answers": {"department": {"choice": "billing",
                            "probabilities": {"billing": 0.91, "technical": 0.09},
                            "answer_confidence": 0.91},
             "urgency":    {"score": 2.6, "probabilities": {...}, "answer_confidence": 0.7},
             "churn_risk": {"noul": 0.88, "probabilities": {"false": 0.12, "true": 0.88},
                            "answer_confidence": 0.88}},
 "usage": {"forward_passes": 1, "questions": 3, "ms": 34.2,
           "truncated": False, "state_tokens": 21, "state_tokens_used": 21}}
```

- [ ] **Step 1: Write the failing tests**

```python
def test_three_questions_answered_in_one_forward_pass(tiny_agent):
    out = tiny_agent.predict("billed twice", {"department": CHOICE, "urgency": SCORE, "churn_risk": NOUL})
    assert set(out["answers"]) == {"department", "urgency", "churn_risk"}
    assert out["usage"]["forward_passes"] == 1
    assert out["usage"]["questions"] == 3

def test_each_type_reports_its_own_key(tiny_agent):
    out = tiny_agent.predict("s", {"d": CHOICE, "u": SCORE, "c": NOUL})
    assert out["answers"]["d"]["choice"] in CHOICE["criteria"]
    assert 0 <= out["answers"]["u"]["score"] <= 2
    assert 0.0 <= out["answers"]["c"]["noul"] <= 1.0

def test_probabilities_are_keyed_by_label_and_sum_to_one(tiny_agent):
    p = tiny_agent.predict("s", {"d": CHOICE})["answers"]["d"]["probabilities"]
    assert set(p) == set(CHOICE["criteria"])
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-5)

def test_truncation_is_surfaced_in_usage(tiny_agent):
    out = tiny_agent.predict("word " * 4000, {"d": CHOICE})
    assert out["usage"]["truncated"] is True
    assert out["usage"]["state_tokens_used"] < out["usage"]["state_tokens"]

def test_empty_state_is_rejected(tiny_agent):
    with pytest.raises(ValueError, match="state"):
        tiny_agent.predict("   ", {"d": CHOICE})

def test_state_is_tokenized_once_for_all_questions(tiny_agent, monkeypatch):
    seen = []
    real = openjev.encode.build_sequence
    monkeypatch.setattr(openjev.encode, "build_sequence",
                        lambda *a, **kw: (seen.append(kw.get("state_ids")), real(*a, **kw))[1])
    tiny_agent.predict("shared state", {"a": CHOICE, "b": CHOICE, "c": NOUL})
    assert len(seen) == 3
    assert all(s is not None and s is seen[0] for s in seen)

def test_temperature_from_the_checkpoint_is_applied(tmp_path, tiny_model, tok):
    save_checkpoint(tiny_model, tok, tmp_path / "cold")
    tiny_model.temperature.copy_(torch.tensor([6.0, 6.0, 6.0]))
    save_checkpoint(tiny_model, tok, tmp_path / "warm")
    cold = load(tmp_path / "cold").predict("s", {"d": CHOICE})["answers"]["d"]
    warm = load(tmp_path / "warm").predict("s", {"d": CHOICE})["answers"]["d"]
    assert warm["answer_confidence"] < cold["answer_confidence"]
    assert warm["choice"] == cold["choice"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: FAIL — `cannot import name 'Agent'`

- [ ] **Step 3: Implement `Agent` in `openjev/__init__.py`**

`predict` validates the state once, tokenizes it once with `add_special_tokens=False`, builds one `Encoded` per question passing that `state_ids`, collates them into a single batch, runs one forward under `torch.inference_mode()`, applies the checkpoint's temperature, and maps the probability vectors back onto each question's labels. `score` reports the expected level, `noul` reports P(true), `choice` reports the argmax label. `answer_confidence` is the top probability. `usage.truncated` is true when any question's state was truncated, and the token counts come from the shared state so they are the same for every question.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add openjev/__init__.py tests/test_agent.py
git commit -m "feat: answer every question about a state in one call"
```

---

### Task 13: Baseline parity and the demo

**Files:**
- Modify: `common.py` (`SCHEMA`, `compile_field`, `build_trie`, `choices_of`, `decide`)
- Create: `2_one_pass.py`, `tests/test_baseline.py`, `NOTICE`, `LICENSE`

**Interfaces:**
- Consumes: Task 12's `Agent.predict`
- Produces: `common.SCHEMA` fields carrying an optional `descriptions: dict[str, str]`; `common.decide` unchanged in signature

- [ ] **Step 1: Write the failing tests**

The spec's baseline-parity requirement: the trie scorer must see the same option descriptions our model does, or the comparison flatters us.

```python
def test_option_descriptions_reach_the_prompt(tok):
    schema = {"decision": {"type": "enum", "description": "Case decision",
                           "choices": ["FRAUD", "CLEAR"],
                           "descriptions": {"FRAUD": "confirmed fraudulent",
                                            "CLEAR": "legitimate activity"}}}
    prompt = common.parallel_prompt(tok, "a case", schema)
    assert "confirmed fraudulent" in prompt and "legitimate activity" in prompt

def test_score_field_renders_as_ordered_levels(tok):
    schema = {"urgency": {"type": "score", "description": "How urgent",
                          "choices": ["can wait", "today", "now"]}}
    assert common.choices_of(schema["urgency"]) == ["can wait", "today", "now"]
    trie = common.build_trie(tok, schema)
    assert [c for _, c in trie.labels] == ["can wait", "today", "now"]

def test_schema_without_descriptions_still_works(tok):
    trie = common.build_trie(tok, common.SCHEMA)
    assert len(trie.labels) == sum(len(common.choices_of(f)) for f in common.SCHEMA.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_baseline.py -v`
Expected: FAIL — `descriptions` ignored, `score` type unknown to `choices_of`

- [ ] **Step 3: Extend `common.py` and write `2_one_pass.py`**

In `common.py`: `choices_of` accepts `"score"` as a third type whose `choices` are its ordered levels; `parallel_prompt` renders each option's description when one is given. Nothing else changes, and `1_vanilla_decoding.py` is not touched — it is the illustration, not the baseline.

`2_one_pass.py` runs the spec's `CONTEXT` through both engines and prints them side by side: the trie scorer's per-field probabilities with its forward-pass count, then `Agent.predict`'s answers with `usage`. End with the two latencies and the two forward-pass counts, so the claim in the spec's success #2 is something the reader watches happen.

Add `LICENSE` (Apache-2.0) and `NOTICE` crediting laya for the sequence format, checkpoint layout and data format.

- [ ] **Step 4: Run the tests and the demo**

Run: `.venv/bin/python -m pytest tests/ -v` then `TINY=1 .venv/bin/python 2_one_pass.py`
Expected: tests PASS; the demo prints both engines' answers and both forward-pass counts

- [ ] **Step 5: Commit**

```bash
git add common.py 2_one_pass.py tests/test_baseline.py LICENSE NOTICE
git commit -m "feat: compare the decision model against the trie-scored baseline"
```

---

## After the plan: the runs

These are not implementation tasks; they are the spec's milestone 5, and they consume the code above.

- [ ] Local base-size run: `answerdotai/ModernBERT-base`, typed-decisions train split, 3 seeds, positive control first. Record how many seeds escaped `ln K`.
- [ ] Kaggle 2xT4 run: `answerdotai/ModernBERT-large`, fp16, DDP via `torchrun --nproc_per_node=2`, typed-decisions + MASSIVE (20-option) + BoolQ + AG News.
- [ ] Report on the typed-decisions test split against the 0.318 random baseline and the 0.646 target band, with ECE per type and the trie-scorer baseline alongside.
