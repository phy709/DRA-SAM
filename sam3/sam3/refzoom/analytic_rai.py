"""Analytic-Risk Referential Active Inference.

Final simplified semantics:
    one referent posterior
    + one learned evidence scorer
    + analytic LOOK
    + additive evidence UPDATE
    + MAP.

The EvidenceScorer supplies candidate evidence; the analytic utility selects
the observation, and the updated posterior determines the output mask.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EPS = 1e-6


def masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    return torch.softmax(logits.masked_fill(~valid, -1e4), dim=-1)


def box_iou_from_geometry(geometry: Tensor, valid: Tensor) -> Tensor:
    """Pairwise box IoU from normalized cx,cy,w,h. Training proxy only.

    End-to-end inference should pass exact mask disagreement when masks exist.
    """
    cx, cy, w, h = [geometry[..., i] for i in range(4)]
    x0 = cx - w / 2
    y0 = cy - h / 2
    x1 = cx + w / 2
    y1 = cy + h / 2

    ix0 = torch.maximum(x0[:, :, None], x0[:, None, :])
    iy0 = torch.maximum(y0[:, :, None], y0[:, None, :])
    ix1 = torch.minimum(x1[:, :, None], x1[:, None, :])
    iy1 = torch.minimum(y1[:, :, None], y1[:, None, :])
    inter = (ix1 - ix0).clamp_min(0) * (iy1 - iy0).clamp_min(0)
    area = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    union = area[:, :, None] + area[:, None, :] - inter
    iou = inter / union.clamp_min(EPS)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    return iou * pair_valid.to(iou.dtype)


class EvidenceScorer(nn.Module):
    """The only learned semantic component.

    It embeds an expression, concrete mask hypotheses, and local observations
    into one shared evidence space. All posterior updates use these same
    representations.
    """

    def __init__(self, dim: int = 256, geometry_dim: int = 7, hidden: int = 192):
        super().__init__()
        self.dim = dim
        self.geometry_dim = geometry_dim

        self.text_norm = nn.LayerNorm(dim)
        self.text_proj = nn.Linear(dim, dim)
        self.visual_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.candidate_feature = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.candidate_context = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.candidate_geometry = nn.Sequential(
            nn.Linear(geometry_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.candidate_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self.evidence_scale = nn.Parameter(torch.tensor(4.0))
        self.raw_score_scale = nn.Parameter(torch.tensor(0.5))
        self.unknown_bias = nn.Parameter(torch.tensor(0.0))
        self.unknown_context = nn.Linear(dim * 2, 1)

        # Pair evidence remains a single calibrated equation, not another head.
        self.pair_cos_scale = nn.Parameter(torch.tensor(4.0))
        self.pair_distance_scale = nn.Parameter(torch.tensor(2.0))
        self.pair_area_scale = nn.Parameter(torch.tensor(0.5))
        self.miss_penalty = nn.Parameter(torch.tensor(2.0))

    @staticmethod
    def text_mean(text_tokens: Tensor, text_padding_mask: Tensor) -> Tensor:
        valid = (~text_padding_mask).to(text_tokens.dtype).unsqueeze(-1)
        return (text_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def sample_context(global_visual: Tensor, geometry: Tensor) -> Tensor:
        b, m, _ = geometry.shape
        grid = geometry[..., :2].clamp(0, 1).mul(2).sub(1).view(b, m, 1, 2)
        sampled = F.grid_sample(
            global_visual, grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def encode_text(self, text_tokens: Tensor, text_padding_mask: Tensor) -> Tensor:
        x = self.text_mean(text_tokens, text_padding_mask)
        return F.normalize(self.text_proj(self.text_norm(x)), dim=-1)

    def encode_candidates(
        self,
        candidate_features: Tensor,
        candidate_geometry: Tensor,
        raw_scores: Tensor,
        global_visual: Tensor,
    ) -> Tensor:
        if raw_scores.ndim == 2:
            raw_scores = raw_scores.unsqueeze(-1)
        context = self.sample_context(global_visual, candidate_geometry)
        x = (
            self.candidate_feature(candidate_features)
            + self.candidate_context(context)
            + self.candidate_geometry(
                torch.cat([candidate_geometry, raw_scores], dim=-1)
            )
        )
        return F.normalize(self.candidate_out(x), dim=-1)

    def candidate_evidence(
        self,
        candidate_tokens: Tensor,
        text_state: Tensor,
        raw_scores: Tensor,
    ) -> Tensor:
        if raw_scores.ndim == 3:
            raw_scores = raw_scores.squeeze(-1)
        cosine = (candidate_tokens * text_state[:, None, :]).sum(dim=-1)
        return (
            F.softplus(self.evidence_scale) * cosine
            + self.raw_score_scale * raw_scores
        )

    def unknown_logit(
        self, global_visual: Tensor, text_state: Tensor
    ) -> Tensor:
        gv = F.normalize(
            self.visual_proj(global_visual).mean(dim=(-2, -1)), dim=-1
        )
        return self.unknown_bias + self.unknown_context(
            torch.cat([gv, text_state], dim=-1)
        ).squeeze(-1)

    def unknown_spatial(
        self, global_visual: Tensor, text_state: Tensor
    ) -> Tensor:
        fmap = F.normalize(self.visual_proj(global_visual), dim=1)
        logits = (fmap * text_state[:, :, None, None]).sum(dim=1)
        return torch.softmax(logits.flatten(1), dim=-1).view_as(logits)

    def pair_evidence(
        self,
        prior_tokens: Tensor,
        prior_geometry: Tensor,
        local_tokens: Tensor,
        local_geometry: Tensor,
    ) -> Tensor:
        cosine = torch.einsum("bmc,bkc->bmk", prior_tokens, local_tokens)
        pc = prior_geometry[..., :2][:, :, None, :]
        lc = local_geometry[..., :2][:, None, :, :]
        dist = (pc - lc).square().sum(dim=-1).sqrt()

        pa = (
            prior_geometry[..., 2] * prior_geometry[..., 3]
        ).clamp_min(EPS)[:, :, None]
        la = (
            local_geometry[..., 2] * local_geometry[..., 3]
        ).clamp_min(EPS)[:, None, :]
        area_delta = (la / pa).log().abs().clamp_max(6.0)

        return (
            F.softplus(self.pair_cos_scale) * cosine
            - F.softplus(self.pair_distance_scale) * dist
            - F.softplus(self.pair_area_scale) * area_delta
        )


class AnalyticReferentialRisk:
    """Parameter-free LOOK operator."""

    @staticmethod
    def visibility(
        geometry: Tensor,
        actions: Tensor,
        hypothesis_valid: Tensor,
        temperature: float = 28.0,
    ) -> Tensor:
        # geometry [B,M,7], actions [B,A,3] = cx,cy,scale
        hc = geometry[..., :2][:, None, :, :]
        hw = geometry[..., 2:4][:, None, :, :].clamp_min(0.005)
        ac = actions[..., :2][:, :, None, :]
        half = actions[..., 2][:, :, None, None] / 2.0

        # Mildly account for hypothesis extent when deciding whether it can be
        # inspected by the crop.
        effective_half = (half + 0.15 * hw).clamp_max(1.0)
        delta = (hc - ac).abs()
        inside = torch.sigmoid(
            temperature * (effective_half - delta)
        ).prod(dim=-1)

        # Smaller crops provide higher effective resolution.
        magnification = (
            0.30 / actions[..., 2].clamp_min(0.30)
        ).sqrt()[:, :, None]
        vis = inside * magnification
        return vis * hypothesis_valid[:, None, :].to(vis.dtype)

    @staticmethod
    def unknown_crop_mass(
        unknown_spatial: Tensor,
        actions: Tensor,
        temperature: float = 28.0,
    ) -> Tensor:
        b, h, w = unknown_spatial.shape
        ys = (torch.arange(h, device=actions.device, dtype=actions.dtype) + 0.5) / h
        xs = (torch.arange(w, device=actions.device, dtype=actions.dtype) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        cx = actions[..., 0, None, None]
        cy = actions[..., 1, None, None]
        half = actions[..., 2, None, None] / 2.0
        wx = torch.sigmoid(temperature * (half - (xx - cx).abs()))
        wy = torch.sigmoid(temperature * (half - (yy - cy).abs()))
        cover = wx * wy
        return (
            cover * unknown_spatial[:, None, :, :]
        ).sum(dim=(-2, -1)).clamp(0.0, 1.0)

    @staticmethod
    def score(
        belief_probs: Tensor,
        hypothesis_geometry: Tensor,
        hypothesis_valid: Tensor,
        actions: Tensor,
        unknown_spatial: Tensor,
        action_valid: Optional[Tensor] = None,
        mask_disagreement: Optional[Tensor] = None,
        unknown_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        concrete = belief_probs[:, :-1]
        unknown = belief_probs[:, -1]
        vis = AnalyticReferentialRisk.visibility(
            hypothesis_geometry, actions, hypothesis_valid
        )

        if mask_disagreement is None:
            overlap = box_iou_from_geometry(
                hypothesis_geometry, hypothesis_valid
            )
            disagreement = 1.0 - overlap
        else:
            disagreement = mask_disagreement

        pair_valid = (
            hypothesis_valid[:, :, None]
            & hypothesis_valid[:, None, :]
        )
        eye = torch.eye(
            hypothesis_valid.shape[1],
            dtype=torch.bool,
            device=hypothesis_valid.device,
        )[None]
        pair_valid = pair_valid & ~eye

        pair_risk = (
            concrete[:, :, None]
            * concrete[:, None, :]
            * disagreement
            * pair_valid.to(concrete.dtype)
        )

        # An observation can resolve a pair by inspecting either hypothesis.
        vi = vis[:, :, :, None]
        vj = vis[:, :, None, :]
        testability = 1.0 - (1.0 - vi) * (1.0 - vj)
        competition = 0.5 * (
            pair_risk[:, None, :, :] * testability
        ).sum(dim=(-2, -1))

        unknown_mass = AnalyticReferentialRisk.unknown_crop_mass(
            unknown_spatial, actions
        )
        discovery = unknown_weight * unknown[:, None] * unknown_mass
        utility = competition + discovery

        if action_valid is not None:
            utility = utility.masked_fill(~action_valid, -1e4)

        total_risk = 0.5 * pair_risk.sum(dim=(-2, -1))
        return {
            "utility": utility,
            "competition_utility": competition,
            "discovery_utility": discovery,
            "visibility": vis,
            "unknown_crop_mass": unknown_mass,
            "referential_risk": total_risk,
            "disagreement": disagreement,
        }


class AnalyticRAI(nn.Module):
    """One scorer + analytic LOOK + additive UPDATE."""

    update_rule_version = "candidate_belief_v1"

    def __init__(
        self,
        dim: int = 256,
        geometry_dim: int = 7,
        hidden: int = 192,
        unknown_weight: float = 1.0,
        posterior_temperature: float = 1.0,
    ):
        super().__init__()
        self.scorer = EvidenceScorer(
            dim=dim, geometry_dim=geometry_dim, hidden=hidden
        )
        self.unknown_weight = unknown_weight
        self.posterior_temperature = float(posterior_temperature)
        if self.posterior_temperature <= 0:
            raise ValueError("posterior_temperature must be positive")

    def prior(
        self,
        text_tokens: Tensor,
        text_padding_mask: Tensor,
        global_visual: Tensor,
        candidate_features: Tensor,
        candidate_geometry: Tensor,
        raw_scores: Tensor,
        candidate_valid: Tensor,
    ) -> Dict[str, Tensor]:
        text_state = self.scorer.encode_text(
            text_tokens, text_padding_mask
        )
        tokens = self.scorer.encode_candidates(
            candidate_features,
            candidate_geometry,
            raw_scores,
            global_visual,
        )
        concrete_logits = self.scorer.candidate_evidence(
            tokens, text_state, raw_scores
        ).masked_fill(~candidate_valid, -1e4)
        unknown_logit = self.scorer.unknown_logit(
            global_visual, text_state
        )
        logits = torch.cat(
            [concrete_logits, unknown_logit[:, None]], dim=-1
        )
        belief_valid = torch.cat(
            [
                candidate_valid,
                torch.ones(
                    candidate_valid.shape[0], 1,
                    dtype=torch.bool,
                    device=candidate_valid.device,
                ),
            ],
            dim=-1,
        )
        probs = masked_softmax(logits / self.posterior_temperature, belief_valid)
        return {
            "text_state": text_state,
            "hypothesis_tokens": tokens,
            "logits": logits,
            "probs": probs,
            "unknown_spatial": self.scorer.unknown_spatial(
                global_visual, text_state
            ),
        }

    def look(
        self,
        prior: Dict[str, Tensor],
        hypothesis_geometry: Tensor,
        hypothesis_valid: Tensor,
        actions: Tensor,
        action_valid: Optional[Tensor] = None,
        mask_disagreement: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return AnalyticReferentialRisk.score(
            belief_probs=prior["probs"],
            hypothesis_geometry=hypothesis_geometry,
            hypothesis_valid=hypothesis_valid,
            actions=actions,
            unknown_spatial=prior["unknown_spatial"],
            action_valid=action_valid,
            mask_disagreement=mask_disagreement,
            unknown_weight=self.unknown_weight,
        )

    def update(
        self,
        prior_logits: Tensor,
        text_state: Tensor,
        global_visual: Tensor,
        prior_tokens: Tensor,
        prior_geometry: Tensor,
        prior_valid: Tensor,
        local_features: Tensor,
        local_geometry: Tensor,
        local_raw_scores: Tensor,
        local_valid: Tensor,
        action: Tensor,
        unknown_crop_mass: Tensor,
    ) -> Dict[str, Tensor]:
        """Add observation evidence to the same posterior state."""
        local_tokens = self.scorer.encode_candidates(
            local_features,
            local_geometry,
            local_raw_scores,
            global_visual,
        )
        local_evidence = self.scorer.candidate_evidence(
            local_tokens, text_state, local_raw_scores
        )
        pair = self.scorer.pair_evidence(
            prior_tokens, prior_geometry, local_tokens, local_geometry
        )
        pair_valid = prior_valid[:, :, None] & local_valid[:, None, :]

        visibility = AnalyticReferentialRisk.visibility(
            prior_geometry,
            action[:, None, :],
            prior_valid,
        )[:, 0]

        pair_masked = pair.masked_fill(~pair_valid, -1e4)
        has_local = local_valid.any(dim=-1)
        count = local_valid.sum(dim=-1).clamp_min(1).to(pair.dtype)
        matched = torch.logsumexp(pair_masked, dim=-1) - count.log()[:, None]
        matched = torch.where(
            has_local[:, None],
            matched,
            torch.zeros_like(matched),
        )

        # Negative evidence is automatic when an inspected hypothesis is not
        # supported by the local observation.
        evidence_existing = visibility * matched
        no_match = (~pair_valid).all(dim=-1) | (matched < 0)
        evidence_existing = evidence_existing - (
            visibility
            * no_match.to(visibility.dtype)
            * F.softplus(self.scorer.miss_penalty)
        )

        concrete_prior = prior_logits[:, :-1]
        unknown_prior = prior_logits[:, -1]
        existing_logits = (
            concrete_prior + evidence_existing
        ).masked_fill(~prior_valid, -1e4)

        # A new local mask inherits evidence from either a matching existing
        # hypothesis or unresolved open-set mass.
        log_belief = torch.log_softmax(prior_logits / self.posterior_temperature, dim=-1)
        source_existing = (
            log_belief[:, :-1, None] + pair_masked
        )
        source_existing = torch.logsumexp(source_existing, dim=1)

        spawn_mass = unknown_crop_mass.clamp_min(EPS)
        source_unknown = (
            log_belief[:, -1, None]
            + spawn_mass.log()[:, None]
        )
        source = torch.logaddexp(source_existing, source_unknown)
        new_logits = source + local_evidence
        new_logits = new_logits.masked_fill(~local_valid, -1e4)

        # Explored unresolved mass is consumed analytically.
        residual = (1.0 - unknown_crop_mass).clamp_min(EPS)
        unknown_logit = unknown_prior + residual.log()

        logits = torch.cat(
            [existing_logits, new_logits, unknown_logit[:, None]], dim=-1
        )
        valid = torch.cat(
            [
                prior_valid,
                local_valid,
                torch.ones(
                    prior_valid.shape[0], 1,
                    dtype=torch.bool,
                    device=prior_valid.device,
                ),
            ],
            dim=-1,
        )
        probs = masked_softmax(logits / self.posterior_temperature, valid)
        return {
            "logits": logits,
            "probs": probs,
            "valid": valid,
            "local_tokens": local_tokens,
            "pair_evidence": pair,
            "expanded_tokens": torch.cat(
                [prior_tokens, local_tokens], dim=1
            ),
            "expanded_geometry": torch.cat(
                [prior_geometry, local_geometry], dim=1
            ),
            "expanded_valid": torch.cat(
                [prior_valid, local_valid], dim=1
            ),
        }

    @staticmethod
    def map_decision(
        belief_probs: Tensor,
        concrete_valid: Tensor,
        fallback_index: int = 0,
    ) -> Tensor:
        concrete = belief_probs[:, :-1].masked_fill(
            ~concrete_valid, -1.0
        )
        best_prob, best_idx = concrete.max(dim=-1)
        unknown = belief_probs[:, -1]
        fallback = torch.full_like(best_idx, fallback_index)
        return torch.where(unknown > best_prob, fallback, best_idx)
