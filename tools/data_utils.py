#!/usr/bin/env python3
"""Shared DRA-SAM data, mask and crop utilities.

Training labels use GT masks. Inference uses image/text features, candidate
geometry, SAM scores and the current candidate posterior.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sam3"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from inference import DEFAULT_ADAPTER, DEFAULT_BPE, build_transform, process_batch


DEFAULT_ANN = None
DEFAULT_IMG = None
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "stage3.pt")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "cache")


def dec_rle(rle):
    r = dict(rle)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode()
    return mask_utils.decode(r).astype(bool)


def enc_rle(mask):
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(r["counts"], bytes):
        r["counts"] = r["counts"].decode()
    return {"size": list(r["size"]), "counts": r["counts"]}


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 1, 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def crop_from_center(cx, cy, scale, width, height):
    cw = max(2, int(round(width * scale)))
    ch = max(2, int(round(height * scale)))
    x0 = int(round(cx - cw / 2))
    y0 = int(round(cy - ch / 2))
    x0 = max(0, min(width - cw, x0))
    y0 = max(0, min(height - ch, y0))
    return (x0, y0, x0 + cw, y0 + ch)


def expanded_box(box, factor, width, height):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    bw = max(1.0, x1 - x0) * factor
    bh = max(1.0, y1 - y0) * factor
    return (
        max(0.0, cx - bw / 2),
        max(0.0, cy - bh / 2),
        min(float(width), cx + bw / 2),
        min(float(height), cy + bh / 2),
    )


def box_intersection_fraction(inner, outer):
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(inter / area)


def sample_actions(gt_mask, scales, random_count, rng):
    h, w = gt_mask.shape
    x0, y0, x1, y1 = mask_bbox(gt_mask)
    gcx, gcy = (x0 + x1) / 2, (y0 + y1) / 2
    items = []

    # GT-centered actions provide offline training supervision.
    for scale in scales:
        crop_size = min(w, h) * scale
        for jitter_frac in (0.0, 0.08):
            jx = rng.uniform(-jitter_frac, jitter_frac) * crop_size
            jy = rng.uniform(-jitter_frac, jitter_frac) * crop_size
            items.append(("gt_jitter", crop_from_center(gcx + jx, gcy + jy, scale, w, h)))

    # Nearby actions provide contrasting observations for scorer supervision.
    for scale in scales[: min(2, len(scales))]:
        step = min(w, h) * scale * 0.55
        for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
            items.append(("shifted", crop_from_center(gcx + dx, gcy + dy, scale, w, h)))

    # Uniform random actions broaden the training observation distribution.
    for _ in range(random_count):
        scale = rng.choice(scales)
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        items.append(("random", crop_from_center(cx, cy, scale, w, h)))

    dedup = []
    seen = set()
    for kind, box in items:
        if box not in seen:
            seen.add(box)
            dedup.append((kind, box))
    return dedup


@dataclass
class QueryInfo:
    annotation_id: int
    image_id: int
    file_name: str
    prompt: object
    prompt_type: str
    source: str
    action_id: int
    kind: str
    x0: int
    y0: int
    width: int
    height: int
    full_width: int
    full_height: int


class CropDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r = self.items[idx]
        image = Image.open(r["image_path"]).convert("RGB")
        if r["source"] == "zoom":
            image = image.crop(r["xyxy"])
        width, height = image.size
        q = FindQueryLoaded(
            query_text=r["prompt"],
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx,
                original_image_id=r["image_id"],
                original_category_id=1,
                original_size=[height, width],
                object_id=0,
                frame_index=0,
            ),
        )
        dp = Datapoint(
            find_queries=[q],
            images=[SAMImage(data=image, objects=[], size=[height, width])],
        )
        qi = QueryInfo(
            annotation_id=r["annotation_id"],
            image_id=r["image_id"],
            file_name=os.path.basename(r["image_path"]),
            prompt=r["prompt"],
            prompt_type="complex_query_0",
            source=r["source"],
            action_id=r["action_id"],
            kind=r["kind"],
            x0=r["xyxy"][0],
            y0=r["xyxy"][1],
            width=width,
            height=height,
            full_width=r["full_width"],
            full_height=r["full_height"],
        )
        return self.transform(dp), qi


def collate_batch(batch):
    return collate([x[0] for x in batch], dict_key="dummy")["dummy"], [x[1] for x in batch]
