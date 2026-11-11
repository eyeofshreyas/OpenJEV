"""Milestone 5's base-size run: ModernBERT-base on typed-decisions.

Positive control first (spec's plateau guard), then 3 seeds of the real
run, each calibrated and evaluated on the held-out test split. Batch size
and gradient checkpointing auto-size to the GPU's total memory, so the same
script runs on a cramped local card or a Kaggle T4/2xT4 unchanged.

Local:

    .venv/bin/python scripts/train_base.py

Kaggle (new notebook, GPU accelerator on, internet on):

    !git clone https://github.com/eyeofshreyas/OpenJEV.git
    %cd OpenJEV
    !pip install -q -r requirements.txt
    !python scripts/train_base.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from openjev.data import items_from_cases, load_typed_decisions
from openjev.eval import evaluate, random_baseline
from openjev.model import build_model
from openjev.train import TrainConfig, fit_temperatures, save_checkpoint, tag_state, train

BACKBONE = "answerdotai/ModernBERT-base"
OUT = Path("runs/base")
SEEDS = [0, 1, 2]


def pick_config(device, **overrides) -> TrainConfig:
    """Size micro_batch/grad_accum/checkpointing off the GPU's total memory.

    Below 8 GB (this project's local card is 4 GB), the 150M full fine-tune's
    activations at this spec's max_len/head_max_len don't fit a batch worth
    running at, so gradient checkpointing trades compute for memory and
    micro_batch drops to 4; grad_accum rises to keep the same effective batch
    (64) as the spec's default, which a Kaggle T4 (16 GB) runs directly.
    """
    if device.type != "cuda" or torch.cuda.get_device_properties(device).total_memory / 1e9 < 8:
        return TrainConfig(epochs=4, micro_batch=4, grad_accum=16, gradient_checkpointing=True, **overrides)
    return TrainConfig(epochs=4, micro_batch=8, grad_accum=8, **overrides)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BACKBONE)

    train_cases = load_typed_decisions("train")
    test_cases = load_typed_decisions("test")
    items, skipped = items_from_cases(train_cases)
    baseline = random_baseline(train_cases)
    print(f"[run] {len(items)} train items, skipped {dict(skipped)}, random baseline {baseline:.3f}")

    print("\n[run] positive control (tagged state must be learned near-perfectly)")
    control = build_model(BACKBONE).to(device)
    control_cfg = pick_config(device, epochs=5, grad_accum=4, seed=0)
    control_report = train([tag_state(i) for i in items[:200]], control, tok, OUT / "control", control_cfg)
    print(f"[run] positive control train accuracy: {control_report['train_accuracy']:.3f}")
    del control
    torch.cuda.empty_cache()
    if control_report["train_accuracy"] < 0.9:
        print("[run] positive control failed -- plumbing or token budget is broken, aborting")
        sys.exit(1)

    results = []
    for seed in SEEDS:
        print(f"\n[run] seed {seed}")
        model = build_model(BACKBONE).to(device)
        cfg = pick_config(device, seed=seed)
        report = train(items, model, tok, OUT / f"seed{seed}", cfg, skipped=skipped)

        escaped = report["train_accuracy"] > 2 * baseline
        temps, notes = fit_temperatures(model, report["calibration_set"], tok, cfg)
        save_checkpoint(model, tok, OUT / f"seed{seed}" / "published", config={"notes": notes}, cfg=cfg)

        metrics = evaluate(model, tok, test_cases)
        result = {
            "seed": seed,
            "train_accuracy": report["train_accuracy"],
            "escaped_ln_k": escaped,
            "test_accuracy": metrics["accuracy"],
            "test_accuracy_by_type": metrics["accuracy_by_type"],
            "ece": metrics["ece"],
            "brier": metrics["brier"],
            "random_baseline": metrics["random_baseline"],
            "temperatures": temps,
        }
        results.append(result)
        print(f"[run] seed {seed}: {json.dumps(result, indent=2)}")

        del model
        torch.cuda.empty_cache()

    n_escaped = sum(r["escaped_ln_k"] for r in results)
    print(f"\n[run] {n_escaped}/{len(results)} seeds escaped the ln K plateau")
    (OUT / "report.json").write_text(json.dumps({"results": results, "random_baseline": baseline}, indent=2))
    print(f"[run] wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
