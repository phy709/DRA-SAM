"""Cached-data adapter for Referential Active Inference."""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class RAICacheDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no cache records under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        x = torch.load(self.files[idx], map_location="cpu")
        meta = x["candidate_meta"]
        initial_idx = [
            i for i, m in enumerate(meta)
            if m.get("source") in {"full_union", "full"}
        ]
        if not initial_idx or initial_idx[0] != 0:
            raise RuntimeError(
                f"{self.files[idx]} does not start with full_union candidate"
            )

        by_action = defaultdict(list)
        for i, m in enumerate(meta):
            if m.get("source") == "zoom":
                by_action[int(m["action_id"])].append(i)

        actions = x["action_meta"]
        action_vectors = []
        action_groups = []
        for a in actions:
            aid = int(a["action_id"])
            action_vectors.append(torch.tensor(a["action"], dtype=torch.float32))
            action_groups.append(by_action.get(aid, []))

        return {
            "annotation_id": int(x["annotation_id"]),
            "global_visual": x["global_visual"].float(),
            "text_tokens": x["text_tokens"].float(),
            "text_padding_mask": x["text_padding_mask"].bool(),
            "initial_features": x["candidate_features"][initial_idx].float(),
            "initial_geometry": x["candidate_geometry"][initial_idx].float(),
            "initial_raw": x["raw_scores"][initial_idx].float(),
            "initial_iou": x["candidate_iou"][initial_idx].float(),
            "actions": torch.stack(action_vectors)
                if action_vectors else torch.empty(0, 3),
            "action_groups": action_groups,
            "all_candidate_features": x["candidate_features"].float(),
            "all_candidate_geometry": x["candidate_geometry"].float(),
            "all_raw_scores": x["raw_scores"].float(),
            "all_candidate_iou": x["candidate_iou"].float(),
            "base_iou": float(x["base_iou"]),
            "_cache_file": self.files[idx],
        }


def collate_rai(records: List[Dict]) -> Dict[str, torch.Tensor]:
    b = len(records)
    c = records[0]["global_visual"].shape[0]
    h, w = records[0]["global_visual"].shape[-2:]
    max_l = max(r["text_tokens"].shape[0] for r in records)
    max_m = max(r["initial_features"].shape[0] for r in records)
    max_a = max(r["actions"].shape[0] for r in records)
    max_k = 1
    for r in records:
        for inds in r["action_groups"]:
            max_k = max(max_k, len(inds))

    text = torch.zeros(b, max_l, c)
    text_pad = torch.ones(b, max_l, dtype=torch.bool)
    global_visual = torch.zeros(b, c, h, w)

    init_feat = torch.zeros(b, max_m, c)
    init_geom = torch.zeros(b, max_m, 7)
    init_raw = torch.zeros(b, max_m)
    init_iou = torch.zeros(b, max_m)
    init_valid = torch.zeros(b, max_m, dtype=torch.bool)

    actions = torch.zeros(b, max_a, 3)
    action_valid = torch.zeros(b, max_a, dtype=torch.bool)
    local_feat = torch.zeros(b, max_a, max_k, c)
    local_geom = torch.zeros(b, max_a, max_k, 7)
    local_raw = torch.zeros(b, max_a, max_k)
    local_iou = torch.zeros(b, max_a, max_k)
    local_valid = torch.zeros(b, max_a, max_k, dtype=torch.bool)

    annotation_id = torch.zeros(b, dtype=torch.long)
    base_iou = torch.zeros(b)

    for bi, r in enumerate(records):
        l = r["text_tokens"].shape[0]
        m = r["initial_features"].shape[0]
        a = r["actions"].shape[0]
        annotation_id[bi] = r["annotation_id"]
        base_iou[bi] = r["base_iou"]
        global_visual[bi] = r["global_visual"]
        text[bi, :l] = r["text_tokens"]
        text_pad[bi, :l] = r["text_padding_mask"]

        init_feat[bi, :m] = r["initial_features"]
        init_geom[bi, :m] = r["initial_geometry"]
        init_raw[bi, :m] = r["initial_raw"]
        init_iou[bi, :m] = r["initial_iou"]
        init_valid[bi, :m] = True

        if a:
            actions[bi, :a] = r["actions"]
            action_valid[bi, :a] = True
        for ai, inds in enumerate(r["action_groups"]):
            if not inds:
                continue
            k = len(inds)
            local_feat[bi, ai, :k] = r["all_candidate_features"][inds]
            local_geom[bi, ai, :k] = r["all_candidate_geometry"][inds]
            local_raw[bi, ai, :k] = r["all_raw_scores"][inds]
            local_iou[bi, ai, :k] = r["all_candidate_iou"][inds]
            local_valid[bi, ai, :k] = True

    return {
        "annotation_id": annotation_id,
        "global_visual": global_visual,
        "text_tokens": text,
        "text_padding_mask": text_pad,
        "initial_features": init_feat,
        "initial_geometry": init_geom,
        "initial_raw": init_raw,
        "initial_iou": init_iou,
        "initial_valid": init_valid,
        "actions": actions,
        "action_valid": action_valid,
        "local_features": local_feat,
        "local_geometry": local_geom,
        "local_raw": local_raw,
        "local_iou": local_iou,
        "local_valid": local_valid,
        "base_iou": base_iou,
    }
