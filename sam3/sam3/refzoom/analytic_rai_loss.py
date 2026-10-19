"""Minimal supervision for Analytic-Risk RAI.

Only the shared evidence scorer is learned.
LOOK and UPDATE structure are fixed by the method.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    w = mask.to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(1.0)


class AnalyticRAILoss(nn.Module):
    def __init__(
        self,
        belief_temperature: float = 0.10,
        prior_weight: float = 1.0,
        posterior_weight: float = 1.0,
    ):
        super().__init__()
        self.belief_temperature = belief_temperature
        self.prior_weight = prior_weight
        self.posterior_weight = posterior_weight

    def belief_target(
        self,
        candidate_iou: Tensor,
        candidate_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        quality = candidate_iou.clamp(0.0, 1.0)
        best = quality.masked_fill(
            ~candidate_valid, -1.0
        ).max(dim=-1).values.clamp(0.0, 1.0)

        # Open-set quality is residual segmentation risk.
        unknown_quality = 1.0 - best
        logits = quality / self.belief_temperature
        logits = logits.masked_fill(~candidate_valid, -1e4)
        logits = torch.cat(
            [
                logits,
                (unknown_quality / self.belief_temperature).unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.softmax(logits, dim=-1), unknown_quality

    def forward(
        self,
        prior_probs: Tensor,
        posterior_probs: Tensor,
        initial_iou: Tensor,
        initial_valid: Tensor,
        local_iou: Tensor,
        local_valid: Tensor,
        action_valid: Tensor,
    ) -> Dict[str, Tensor]:
        b, a, k = local_iou.shape
        m = initial_iou.shape[1]

        prior_target, _ = self.belief_target(
            initial_iou, initial_valid
        )
        prior_ce = -(
            prior_target
            * prior_probs.clamp_min(1e-8).log()
        ).sum(dim=-1)
        loss_prior = prior_ce.mean()

        init_iou_a = initial_iou[:, None, :].expand(-1, a, -1)
        init_valid_a = initial_valid[:, None, :].expand(-1, a, -1)
        post_iou = torch.cat([init_iou_a, local_iou], dim=-1)
        post_valid = torch.cat([init_valid_a, local_valid], dim=-1)

        post_target, _ = self.belief_target(
            post_iou.reshape(b * a, m + k),
            post_valid.reshape(b * a, m + k),
        )
        post_target = post_target.view(b, a, m + k + 1)
        post_ce = -(
            post_target
            * posterior_probs.clamp_min(1e-8).log()
        ).sum(dim=-1)
        loss_post = masked_mean(post_ce, action_valid)

        total = (
            self.prior_weight * loss_prior
            + self.posterior_weight * loss_post
        )
        return {
            "loss_prior_belief": loss_prior,
            "loss_posterior_belief": loss_post,
            "loss_total": total,
        }
