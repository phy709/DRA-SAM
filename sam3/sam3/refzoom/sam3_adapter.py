"""Bridges frozen SAM3-I backbone outputs into ReferZoom feature layouts."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F


def language_to_batch_first(language_features: Tensor, batch_size: int) -> Tensor:
    """SAM3 text memory is normally [L,B,C]; ReferZoom uses [B,L,C]."""
    if language_features.ndim != 3:
        raise ValueError(f"expected 3D language features, got {tuple(language_features.shape)}")
    if language_features.shape[1] == batch_size:
        return language_features.transpose(0, 1).contiguous()
    if language_features.shape[0] == batch_size:
        return language_features
    raise ValueError(
        f"cannot infer text layout for shape={tuple(language_features.shape)}, batch={batch_size}"
    )


def select_visual_map(image_output: Dict[str, object]) -> Tensor:
    """Select the SAM3 feature map used by ReferZoom."""
    x = image_output.get("vision_features")
    if not torch.is_tensor(x) or x.ndim != 4:
        raise ValueError("SAM3 image output does not contain a 4D vision_features tensor")
    return x


def masked_pool_candidates(feature_map: Tensor, masks: Tensor, eps: float = 1e-6) -> Tensor:
    """Pool one feature vector per candidate mask."""
    if feature_map.ndim != 4 or masks.ndim != 4:
        raise ValueError("feature_map must be [B,C,H,W] and masks [B,M,H0,W0]")
    b, c, h, w = feature_map.shape
    if masks.shape[0] != b:
        raise ValueError("feature-map and mask batch sizes differ")
    m = masks.shape[1]
    resized = F.interpolate(
        masks.float().reshape(b * m, 1, masks.shape[-2], masks.shape[-1]),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).reshape(b, m, h, w)
    weights = resized.clamp_min(0.0)
    denom = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    weights = weights / denom
    return torch.einsum("bmhw,bchw->bmc", weights, feature_map)


class FrozenSAM3FeatureAdapter:
    """Thin feature extractor around a frozen SAM3-I model."""

    def __init__(self, sam3i_model, text_type: int = 4):
        self.model = sam3i_model
        self.text_type = text_type

    def encode_global(self, image_batch: Tensor, captions) -> Dict[str, Tensor]:
        image_out = self.model.backbone.forward_image(image_batch)
        text_out = self.model.backbone.forward_text(
            captions=captions,
            text_type=self.text_type,
            device=image_batch.device,
        )
        visual_map = select_visual_map(image_out)
        text_tokens = language_to_batch_first(
            text_out["language_features"], batch_size=image_batch.shape[0]
        )
        return {
            "visual_map": visual_map,
            "text_tokens": text_tokens,
            "text_padding_mask": text_out["language_mask"],
        }

    def encode_candidate_masks(
        self,
        image_batch: Tensor,
        candidate_masks: Tensor,
    ) -> Tensor:
        image_out = self.model.backbone.forward_image(image_batch)
        visual_map = select_visual_map(image_out)
        return masked_pool_candidates(visual_map, candidate_masks)