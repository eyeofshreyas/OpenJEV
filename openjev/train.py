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

import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors.torch import save_model
from transformers import get_cosine_schedule_with_warmup

from openjev.data import Item, _gold_keys, split_items, update_budget
from openjev.encode import HEAD_MAX_LEN, MAX_LEN, build_sequence, collate, strip_mask
from openjev.score import decision_loss


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


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """The autocast dtype for a device: bf16 from Ampere up, fp16 below, none on CPU.

    A T4 is compute capability 7.5 and has no bf16, and the real training run
    happens on Kaggle's 2xT4 — so the fp16 path (which also needs a
    ``GradScaler``) is the one the published numbers come off.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


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


@torch.no_grad()
def accuracy(model, tok, items: list[Item], cfg: TrainConfig, device) -> float:
    """The share of ``items`` whose most likely option is the target's, in eval mode.

    Deliberately not the running tally the epochs print: that one is measured
    with dropout live and the weights still moving, so it reads a point or two
    low and jitters. The number the report states — the one that answers "did
    this run learn anything" — is worth one clean pass.
    """
    was_training = model.training
    model.eval()
    state_ids: dict[str, list[int]] = {}
    correct = 0
    try:
        for i in range(0, len(items), cfg.micro_batch):
            batch, target = _batch(tok, items[i : i + cfg.micro_batch], cfg, None, state_ids)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            correct += int((logits.argmax(-1).cpu() == target.argmax(-1)).sum())
    finally:
        model.train(was_training)
    return correct / len(items)


def train(items: list[Item], model, tok, out_dir, cfg: TrainConfig | None = None, skipped=None) -> dict:
    """Fine-tune ``model`` on ``items`` and return a report on the run.

    The calibration set is held out first, and a run with nothing left to
    train on is refused: a loader whose rows were all rejected must not report
    success with a large skip count and a training set of zero.
    """
    cfg = cfg or TrainConfig()
    train_items, calib_items = split_items(items, cfg.calib_frac, cfg.seed)
    if not train_items:
        raise ValueError(
            f"no training items after the calibration split: {len(items)} items in, "
            f"{len(calib_items)} held out for calibration, {sum((skipped or {}).values())} skipped"
        )

    torch.manual_seed(cfg.seed)
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

    rng = random.Random(cfg.seed)
    state_ids: dict[str, list[int]] = {}
    was_training = model.training
    model.train()

    updates, epoch_loss = 0, []
    try:
        for epoch in range(cfg.epochs):
            order = list(train_items)
            rng.shuffle(order)
            micro = [order[i : i + cfg.micro_batch] for i in range(0, len(order), cfg.micro_batch)]

            total_loss, correct, seen = 0.0, 0, 0
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

                # Divided by grad_accum so the accumulated gradient is the mean over the
                # effective batch — its magnitude must not scale with grad_accum.
                scaler.scale(loss / cfg.grad_accum).backward()

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

            epoch_loss.append(total_loss / seen)
            print(f"[train] epoch {epoch + 1}/{cfg.epochs}: loss {epoch_loss[-1]:.4f}, accuracy {correct / seen:.3f}")
            save_checkpoint(model, out_dir)
    finally:
        model.train(was_training)

    train_accuracy = accuracy(model, tok, train_items, cfg, device)
    print(f"[train] done: train accuracy {train_accuracy:.3f} over {updates} optimizer updates")

    return {
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "skipped": dict(skipped or {}),
        "optimizer_updates": updates,
        "epoch_loss": epoch_loss,
        "train_accuracy": train_accuracy,
        "seed": cfg.seed,
    }


def save_checkpoint(model, out_dir, name: str = "checkpoint_latest") -> Path:
    """Write the model's weights to ``out_dir/name/model.safetensors``.

    Called after every epoch, so a crash costs one epoch rather than the run.
    """
    path = Path(out_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path / "model.safetensors"))
    return path
