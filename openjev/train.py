"""Fine-tune the decision model against the proper scoring rule.

Training is the product here: the backbone starts near chance on this
benchmark and only fine-tuning moves it. A fresh head on a raw backbone can
also park at ``ln K`` — uniform over K options — and never escape, so this
module ships the two instruments that tell a dead run from a bug:

- ``tag_state`` builds a *positive control*: prepend the answer to the state
  and the model must nail it. If it cannot learn a signal that is trivially
  present, the plumbing or the token budget is broken, not the task.
- a non-finite loss raises ``NonFiniteLoss`` naming the step instead of
  training through it, because the loss is log- and norm-heavy and the
  published numbers come off a T4's fp16 path.

A single dead seed is not evidence of a bug; multi-seed runs are how that call
gets made, which is why ``TrainConfig.seed`` is a knob and the report states it.
"""

import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file, load_model, save_file, save_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from openjev.data import Item, _gold_keys, split_items, update_budget
from openjev.encode import HEAD_MAX_LEN, MAX_LEN, QTYPES, build_sequence, collate, strip_mask
from openjev.model import DecisionModel
from openjev.score import decision_loss

# A type with fewer calibration items than this keeps temperature 1.0 rather
# than fit one to noise — honesty about a fit nobody should trust.
MIN_CALIBRATION_ITEMS = 10

# The range a fitted temperature is searched and clamped to.
_TEMP_LO, _TEMP_HI = 0.25, 10.0


class NonFiniteLoss(RuntimeError):
    """The loss went NaN or inf; training stops rather than corrupting the weights."""


@dataclass
class TrainConfig:
    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 8
    lr_encoder: float = 2e-5
    lr_head: float = 1e-4
    warmup_frac: float = 0.1
    max_len: int = MAX_LEN
    head_max_len: int = HEAD_MAX_LEN
    shuffle_options: bool = True
    seed: int = 0
    grad_clip: float = 1.0
    calib_frac: float = 0.1
    # Off by default: it buys memory with ~30% more compute, which a CPU
    # development run and this project's tests have no use for. The spec's
    # Kaggle recipe turns it on.
    gradient_checkpointing: bool = False


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """The autocast dtype for a device: bf16 from Ampere up, fp16 below, none on CPU.

    A T4 is compute capability 7.5 and has no bf16, and the real training run
    happens on Kaggle's 2xT4 — so the fp16 path (which also needs a
    ``GradScaler``) is the one the published numbers come off.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def enable_gradient_checkpointing(model) -> None:
    """Trade compute for memory on the encoder and the head, as the spec's recipe asks.

    Its ~12 GB/rank estimate depends on it: the specced ``micro_batch=8`` at
    ``max_len=1024`` on ModernBERT-large does not otherwise fit a 16 GB T4.

    ``nn.TransformerEncoder`` has no checkpointing of its own, so each head
    layer's ``forward`` is wrapped instead — ``Module.__call__`` dispatches
    through the instance attribute, so the wrap holds however the layer is
    called. ``use_reentrant=False`` on both halves: the reentrant
    implementation needs its inputs to require grad and silently yields no
    gradient when they do not, and transformers warns when the choice is left
    implicit. Idempotent, so a resumed run cannot wrap a layer twice.
    """
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for layer in model.head.layers:
        if getattr(layer, "_openjev_checkpointed", False):
            continue
        layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        layer._openjev_checkpointed = True


def tag_state(item: Item) -> Item:
    """Prepend the item's own answer to its state, as a positive control.

    The signal is then trivially present, so a model that still cannot learn
    the task has a broken loop or too small a token budget, not a hard problem.
    """
    best = max(range(len(item.target)), key=item.target.__getitem__)
    # ``_gold_keys`` is the one place that knows each type's option keys in render
    # order, which is exactly what a tag has to spell; a second copy would drift.
    return replace(item, state=f"answer: {_gold_keys(item.question)[best]}\n{item.state}")


def shuffle_options(item: Item, rng: random.Random) -> tuple[Item, list[float]]:
    """Permute a ``choice`` question's options and its target together.

    ``score`` and ``noul`` questions come back untouched: their option order
    is their meaning (ordered levels, false-then-true). The project ships the
    sequential option layout and trains the position effect down rather than
    engineering around it, so this permutation is load-bearing.
    """
    if item.question["type"] != "choice":
        return item, item.target

    criteria = item.question["criteria"]
    order = list(range(len(criteria)))
    rng.shuffle(order)

    labels = list(criteria)
    permuted = {labels[i]: criteria[labels[i]] for i in order}
    target = [item.target[i] for i in order]
    return replace(item, question={**item.question, "criteria": permuted}, target=target), target


def _batch(tok, items: list[Item], cfg: TrainConfig, rng: random.Random | None, state_ids: dict):
    """Encode and collate one micro-batch, with its padded target matrix.

    The state tokenization is cached across epochs (``build_sequence`` takes
    pre-tokenized state ids for exactly this reason) because option shuffling
    re-encodes every item every epoch, while its state never changes.
    """
    encoded, targets = [], []
    for item in items:
        if rng is not None:
            item, target = shuffle_options(item, rng)
        else:
            target = item.target
        ids = state_ids.get(item.state)
        if ids is None:
            ids = state_ids[item.state] = tok(strip_mask(tok, item.state), add_special_tokens=False)["input_ids"]
        encoded.append(
            build_sequence(
                tok,
                item.state,
                item.question,
                qid=item.qid,
                max_len=cfg.max_len,
                head_max_len=cfg.head_max_len,
                state_ids=ids,
            )
        )
        targets.append(target)

    batch = collate(encoded, tok.pad_token_id)
    target = torch.zeros(len(targets), batch["marker_mask"].size(1))
    for i, row in enumerate(targets):
        target[i, : len(row)] = torch.tensor(row)
    return batch, target


def _pad_to(t: torch.Tensor, k: int, fill) -> torch.Tensor:
    """Right-pad a ``(B, K_chunk)`` tensor out to ``K`` columns with ``fill``."""
    if t.size(1) == k:
        return t
    out = t.new_full((t.size(0), k), fill)
    out[:, : t.size(1)] = t
    return out


@torch.no_grad()
def _forward_items(model, tok, items: list[Item], cfg: TrainConfig, device):
    """Forward ``items`` in ``cfg.micro_batch`` chunks; return CPU logits, targets and mask.

    Chunked, and the results kept on the CPU, because the alternative is one
    forward over every item at once: at the spec's data size that is thousands
    of sequences padded to ``max_len`` through the encoder in a single call,
    which on a real GPU is an out-of-memory crash. Both callers below want the
    same pass, so they share this one instead of each having a batching policy.

    Chunks disagree about ``K`` (the batch's largest option count), so every
    chunk is padded out to the widest one, logits with the same ``-1e4`` the
    model already fills masked slots with and targets with 0.
    """
    was_training = model.training
    model.eval()
    state_ids: dict[str, list[int]] = {}
    chunks = []
    try:
        for i in range(0, len(items), cfg.micro_batch):
            batch, target = _batch(tok, items[i : i + cfg.micro_batch], cfg, None, state_ids)
            batch = {k: v.to(device) for k, v in batch.items()}
            chunks.append((model(**batch).cpu(), target.cpu(), batch["marker_mask"].cpu()))
    finally:
        model.train(was_training)

    if not chunks:
        return torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0, dtype=torch.bool)

    k = max(logits.size(1) for logits, _, _ in chunks)
    return (
        torch.cat([_pad_to(logits, k, -1e4) for logits, _, _ in chunks]),
        torch.cat([_pad_to(target, k, 0.0) for _, target, _ in chunks]),
        torch.cat([_pad_to(mask, k, False) for _, _, mask in chunks]),
    )


def accuracy(model, tok, items: list[Item], cfg: TrainConfig, device) -> float:
    """The share of ``items`` whose most likely option is the target's, in eval mode.

    Deliberately not the running tally the epochs print: that one is measured
    with dropout live and the weights still moving, so it reads a point or two
    low and jitters. The number the report states — the one that answers "did
    this run learn anything" — is worth one clean pass.
    """
    logits, target, _ = _forward_items(model, tok, items, cfg, device)
    return int((logits.argmax(-1) == target.argmax(-1)).sum()) / len(items)


def _type_nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, t) -> torch.Tensor:
    """Mean negative log-likelihood of ``target`` under ``logits`` scaled by ``t``."""
    scaled = (logits / t).masked_fill(~mask, float("-inf"))
    return -(target * torch.log_softmax(scaled, dim=-1)).sum(-1).mean()


def _fit_one_temperature(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Minimise NLL over ``T`` with a log-spaced grid, refined by LBFGS on ``log T``."""
    grid = torch.logspace(math.log10(_TEMP_LO), math.log10(_TEMP_HI), 25)
    grid_nll = torch.tensor([_type_nll(logits, target, mask, float(t)) for t in grid])
    best_t = float(grid[grid_nll.argmin()])

    log_t = torch.tensor(math.log(best_t), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t = log_t.exp().clamp(_TEMP_LO, _TEMP_HI)
        loss = _type_nll(logits, target, mask, t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except IndexError:
        # torch's strong_wolfe line search can raise on a flat enough loss landscape
        # (an empty bracket list) -- the grid search already found a reasonable
        # minimum, so keep it rather than losing calibration for that type.
        return best_t
    return float(log_t.detach().exp().clamp(_TEMP_LO, _TEMP_HI))


def fit_temperatures(model, items: list[Item], tok, cfg: TrainConfig) -> tuple[list[float], dict]:
    """Fit one softmax temperature per question type, on that type's own items only.

    Dividing every logit by the same positive scalar can never change which
    one is largest, so this can only reshape a question's probabilities —
    never its accuracy. Each type is minimised separately (grid search over
    ``[0.25, 10]``, refined by ``LBFGS`` on ``log T``) because ``choice``,
    ``score`` and ``noul`` are miscalibrated differently. A type with fewer
    than ``MIN_CALIBRATION_ITEMS`` items keeps temperature 1.0 and is marked
    ``fitted: False`` rather than fit to noise.

    Writes the fitted values into ``model.temperature`` so they travel with
    whatever checkpoint gets saved next, and returns them again as a plain
    list (in ``QTYPES`` order) alongside a per-type note.
    """
    device = next(model.parameters()).device

    by_type: dict[str, list[Item]] = {name: [] for name in QTYPES}
    for item in items:
        by_type[item.question["type"]].append(item)

    temps = model.temperature.detach().clone().cpu()
    notes: dict[str, dict] = {}

    for name, type_items in by_type.items():
        idx = QTYPES[name]
        if len(type_items) < MIN_CALIBRATION_ITEMS:
            temps[idx] = 1.0
            notes[name] = {
                "fitted": False,
                "n_items": len(type_items),
                "warning": f"only {len(type_items)} {name!r} items, need at least {MIN_CALIBRATION_ITEMS}",
            }
            continue

        logits, target, mask = _forward_items(model, tok, type_items, cfg, device)

        nll_before = float(_type_nll(logits, target, mask, 1.0))
        fitted_t = _fit_one_temperature(logits, target, mask)
        nll_after = float(_type_nll(logits, target, mask, fitted_t))

        temps[idx] = fitted_t
        notes[name] = {
            "fitted": True,
            "n_items": len(type_items),
            "temperature": fitted_t,
            "nll_before": nll_before,
            "nll_after": nll_after,
        }

    model.temperature.copy_(temps.to(model.temperature.device))
    return temps.tolist(), notes


def train(
    items: list[Item], model, tok, out_dir, cfg: TrainConfig | None = None, skipped=None, resume: bool = False
) -> dict:
    """Fine-tune ``model`` on ``items`` and return a report on the run.

    The calibration set is held out first and comes back in the report as
    ``calibration_set``, ready to hand straight to ``fit_temperatures``. A run
    with nothing left to train on is refused: a loader whose rows were all
    rejected must not report success with a large skip count and a training
    set of zero.

    ``resume=True`` picks a crashed run back up from ``out_dir/checkpoint_latest``
    at the epoch after the last one it finished, with the schedule, the Adam
    moments and the loss history it left off with. It is explicit rather than
    automatic, so a stale checkpoint left in ``out_dir`` can never silently
    hijack a fresh run.
    """
    cfg = cfg or TrainConfig()
    train_items, calib_items = split_items(items, cfg.calib_frac, cfg.seed)
    if not train_items:
        raise ValueError(
            f"no training items after the calibration split: {len(items)} items in, "
            f"{len(calib_items)} held out for calibration, {sum((skipped or {}).values())} skipped"
        )

    torch.manual_seed(cfg.seed)
    if cfg.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    device = next(model.parameters()).device
    dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)

    backbone = list(model.encoder.parameters())
    backbone_ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": cfg.lr_encoder},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": cfg.lr_head},
        ]
    )
    budget = update_budget(len(train_items), cfg.micro_batch, cfg.grad_accum, cfg.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(budget * cfg.warmup_frac), budget)

    print(
        f"[train] {len(train_items)} train / {len(calib_items)} calibration items, seed {cfg.seed}: "
        f"{budget} optimizer updates over {cfg.epochs} epochs "
        f"(micro_batch={cfg.micro_batch}, grad_accum={cfg.grad_accum}, "
        f"effective batch {cfg.micro_batch * cfg.grad_accum})"
    )

    updates, epoch_loss, start_epoch = 0, [], 0
    if resume:
        state = load_checkpoint(model, out_dir, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = state.get("epoch", 0)
        updates = state.get("updates", 0)
        epoch_loss = list(state.get("epoch_loss", []))
        print(f"[train] resumed after epoch {start_epoch} at update {updates} of {budget}")

    state_ids: dict[str, list[int]] = {}
    was_training = model.training
    model.train()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            # Seeded per epoch rather than from one rolling generator, so a resumed epoch
            # sees the item order and option permutations it would have seen anyway.
            rng = random.Random(f"{cfg.seed}:{epoch}")
            order = list(train_items)
            rng.shuffle(order)
            micro = [order[i : i + cfg.micro_batch] for i in range(0, len(order), cfg.micro_batch)]

            total_loss, correct, seen = 0.0, 0, 0
            group_start = 0
            optimizer.zero_grad(set_to_none=True)

            for step, chunk in enumerate(micro, start=1):
                batch, target = _batch(tok, chunk, cfg, rng if cfg.shuffle_options else None, state_ids)
                batch = {k: v.to(device) for k, v in batch.items()}
                target = target.to(device)

                with torch.autocast(device.type, dtype=dtype) if dtype else nullcontext():
                    logits = model(**batch)
                loss = decision_loss(logits, target, batch["qtype"], batch["marker_mask"])
                if not torch.isfinite(loss):
                    raise NonFiniteLoss(
                        f"non-finite loss ({loss.item()}) at epoch {epoch + 1}/{cfg.epochs}, "
                        f"micro-batch {step}/{len(micro)}, after {updates} optimizer updates"
                    )

                # Divided by the group's own size so the accumulated gradient is the mean
                # over the effective batch — its magnitude must not scale with grad_accum.
                # The trailing group of an epoch is usually short, and dividing it by
                # grad_accum anyway would quietly shrink that update by up to grad_accum
                # times (and shrink *every* update when an epoch has fewer micro-batches
                # than grad_accum), which moves the effective learning rate with a setting
                # that is supposed to only change how the batch is assembled.
                group_size = min(cfg.grad_accum, len(micro) - group_start)
                scaler.scale(loss / group_size).backward()

                total_loss += loss.item() * len(chunk)
                correct += int((logits.argmax(-1) == target.argmax(-1)).sum())
                seen += len(chunk)

                if step % cfg.grad_accum == 0 or step == len(micro):
                    scaler.unscale_(optimizer)  # a no-op unless the scaler is enabled
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    group_start = step

            epoch_loss.append(total_loss / seen)
            print(f"[train] epoch {epoch + 1}/{cfg.epochs}: loss {epoch_loss[-1]:.4f}, accuracy {correct / seen:.3f}")
            save_checkpoint(
                model,
                out_dir,
                state={
                    "epoch": epoch + 1,
                    "updates": updates,
                    "epoch_loss": epoch_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
            )
    finally:
        model.train(was_training)

    train_accuracy = accuracy(model, tok, train_items, cfg, device)
    print(f"[train] done: train accuracy {train_accuracy:.3f} over {updates} optimizer updates")

    return {
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        # The held-out items themselves, not just their count: the temperature
        # has to be fitted on this exact slice, and a caller that re-derived it
        # by calling ``split_items`` again with the same seed would be trusting
        # that nothing about the split drifted in between.
        "calibration_set": calib_items,
        "skipped": dict(skipped or {}),
        "optimizer_updates": updates,
        "epoch_loss": epoch_loss,
        "train_accuracy": train_accuracy,
        "seed": cfg.seed,
    }


def save_checkpoint(model, *args, state: dict | None = None, name: str = "checkpoint_latest",
                     questions: dict | None = None, config: dict | None = None,
                     cfg: TrainConfig | None = None):
    """Write a checkpoint. Two call shapes share this one pair of functions.

    ``save_checkpoint(model, out_dir, state=...)`` is the *resume* shape this
    loop calls after every epoch: the full weights go to
    ``out_dir/name/model.safetensors``, and ``state`` (optimizer, scheduler,
    scaler, epoch index) goes beside them in ``trainer_state.pt``. A crash or
    a Kaggle timeout then costs one epoch rather than the run — weights alone
    would restart the cosine schedule at warmup with cold Adam moments, which
    is a new run from a warm start, not a resume.

    ``save_checkpoint(model, tok, out_dir, questions=..., config=...)`` is the
    *publishable* shape: the encoder goes to ``out_dir/encoder`` via
    ``save_pretrained``, the tokenizer to ``out_dir/tokenizer``, everything
    else (head, scorer, type embedding, the ``temperature`` buffer) to
    ``out_dir/model.safetensors``, and a reference-implementation-compatible
    ``rl_agent_config.json`` (backbone id, the ``max_len``/``head_max_len``
    the run used — pass the run's ``cfg`` — ``head_layers``, temperatures, and
    whatever notes the ``config`` dict adds) beside it. ``questions``, if
    given, is written to ``questions.json``. This shape also accepts ``state``, so the same out_dir can carry both a rolling
    ``checkpoint_latest`` child (resume shape) and this publishable parent.

    Returns ``None`` for the publishable shape, the checkpoint directory
    (``out_dir/name``) for the resume shape.
    """
    if len(args) == 1:
        out_dir = args[0]
        path = Path(out_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        save_model(model, str(path / "model.safetensors"))
        if state is not None:
            torch.save(state, path / "trainer_state.pt")
        return path

    tok, out_dir = args
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    model.encoder.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")

    head_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
                  if not k.startswith("encoder.")}
    save_file(head_state, str(path / "model.safetensors"))

    # The budget the weights were actually trained at, not the module defaults:
    # ``TrainConfig.max_len``/``head_max_len`` are real knobs (the spec reaches for
    # head_max_len=512 for MASSIVE's 60 labels), and whatever lands here is what
    # the checkpoint is later *served* at.
    budget = cfg or TrainConfig()
    rl_agent_config = {
        "backbone": getattr(model.encoder.config, "name_or_path", "") or None,
        "max_len": budget.max_len,
        "head_max_len": budget.head_max_len,
        "head_layers": len(model.head.layers),
        "temperature": model.temperature.tolist(),
        **(config or {}),
    }
    (path / "rl_agent_config.json").write_text(json.dumps(rl_agent_config, indent=2))

    if questions is not None:
        (path / "questions.json").write_text(json.dumps(questions, indent=2))

    if state is not None:
        torch.save(state, path / "trainer_state.pt")
    return None


def load_checkpoint(*args, optimizer=None, scheduler=None, scaler=None, name="checkpoint_latest"):
    """Load a checkpoint written by ``save_checkpoint``. Two call shapes share this pair.

    ``load_checkpoint(model, out_dir, optimizer=..., scheduler=..., scaler=...)``
    is the *resume* shape: it loads weights into the given ``model`` in place
    from ``out_dir/name/model.safetensors``, restores the optimizer/scheduler
    /scaler when they are passed, and returns the saved state dict (``{}``
    for a weights-only checkpoint).

    ``load_checkpoint(path)`` is the *publishable* shape: it rebuilds a fresh
    ``DecisionModel`` from ``path/encoder`` and ``path/model.safetensors``,
    the tokenizer from ``path/tokenizer``, and returns
    ``(model, tokenizer, rl_agent_config)``.
    """
    if isinstance(args[0], torch.nn.Module):
        model, out_dir = args
        path = Path(out_dir) / name
        load_model(model, str(path / "model.safetensors"))

        state_path = path / "trainer_state.pt"
        if not state_path.exists():
            return {}
        state = torch.load(state_path, weights_only=True)
        for key, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            if obj is not None:
                obj.load_state_dict(state[key])
        return state

    path = Path(args[0])
    rl_agent_config = json.loads((path / "rl_agent_config.json").read_text())

    # local_files_only: a caller-supplied path that doesn't resolve to a real checkpoint directory
    # must fail loudly, not be reinterpreted as a Hub repo id and trigger a network download.
    encoder = AutoModel.from_pretrained(str(path / "encoder"), local_files_only=True)
    model = DecisionModel(encoder, head_layers=rl_agent_config.get("head_layers", 2))

    head_state = load_file(str(path / "model.safetensors"))
    # ``strict=False`` because every ``encoder.*`` key legitimately lives in the
    # encoder directory instead — which is exactly why ``missing`` cannot just be
    # ignored: any *other* absent key is a head, scorer, type embedding or
    # temperature left randomly initialised, and the model would then serve
    # confident garbage while calling itself a trained checkpoint. Raised rather
    # than asserted, because ``python -O`` strips an assert and an integrity check
    # that protects a published number must not vanish under an optimisation flag.
    missing, unexpected = model.load_state_dict(head_state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys in model.safetensors: {unexpected}")
    missing_head = [key for key in missing if not key.startswith("encoder.")]
    if missing_head:
        raise ValueError(
            f"{path / 'model.safetensors'} is missing {len(missing_head)} non-encoder "
            f"key(s), which would be served randomly initialised: {missing_head}"
        )
    model.eval()  # a published checkpoint is for inference; dropout stays off until asked otherwise

    tok = AutoTokenizer.from_pretrained(str(path / "tokenizer"), local_files_only=True)
    return model, tok, rl_agent_config
