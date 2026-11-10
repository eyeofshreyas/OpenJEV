"""The strictly proper scoring rule for calibrated probability estimates.

A strictly proper scoring rule has the property that reporting your true belief
maximizes its expected value. This makes calibration the training objective
rather than a post-hoc patch.
"""

import torch

from openjev.encode import QTYPES
from openjev.model import probabilities


def proper_reward(p, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    """Score predictions against targets using a strictly proper scoring rule.

    Args:
        p: predicted probabilities, shape [B, K]
        target: target distribution, shape [B, K]
        qtype: question type indices, shape [B]
        mask: boolean mask indicating valid options, shape [B, K]
        w_sph: weight for spherical component (default 0.5)
        w_rps: weight for ranked probability score component (default 1.0)
        log_floor: minimum value for log component (default -9.21)

    Returns:
        Reward tensor of shape [B], computed as:
        - log_score + w_sph * spherical
        - w_rps * rps (only for score-type questions)
    """
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


def decision_loss(logits, target, qtype, mask):
    """Compute training loss from logits as negative expected proper reward.

    Args:
        logits: raw model outputs, shape [B, K]
        target: target distribution, shape [B, K]
        qtype: question type indices, shape [B]
        mask: boolean mask indicating valid options, shape [B, K]

    Returns:
        Scalar loss (negative mean proper reward).
    """
    p = probabilities(logits, mask)
    return -proper_reward(p, target, qtype, mask).mean()
