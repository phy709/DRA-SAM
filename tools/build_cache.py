#!/usr/bin/env python3
"""Build the frozen SAM3-I observation cache for EvidenceScorer training.

Preprocessing runs the full image plus sampled
multi-scale zoom actions, captures SAM3-I visual/text features with hooks, pools
candidate-specific mask features, and writes one .pt record per annotation.

GT-dependent fields supply the offline training labels.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sam3"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import data_utils as sup
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.refzoom import masked_pool_candidates
from inference import DEFAULT_ADAPTER, DEFAULT_BPE, build_transform, process_batch


def make_candidate_geometry(mask, source, crop_xyxy, full_w, full_h, is_union=False):
    x0, y0, x1, y1 = sup.mask_bbox(mask)
    cx = ((x0 + x1) / 2.0) / full_w
    cy = ((y0 + y1) / 2.0) / full_h
    bw = (x1 - x0) / full_w
    bh = (y1 - y0) / full_h
    crop_scale = max(
        (crop_xyxy[2] - crop_xyxy[0]) / full_w,
        (crop_xyxy[3] - crop_xyxy[1]) / full_h,
    )
    return [
        float(cx),
        float(cy),
        float(bw),
        float(bh),
        1.0 if source == "zoom" else 0.0,
        float(crop_scale),
        1.0 if is_union else 0.0,
    ]


def pool_one_batch(feature_map, prediction_rles):
    if not prediction_rles:
        return torch.empty(0, feature_map.shape[1], dtype=torch.float16)
    masks = np.stack([sup.dec_rle(x) for x in prediction_rles], axis=0)
    masks_t = torch.from_numpy(masks).to(feature_map.device)
    pooled = masked_pool_candidates(feature_map, masks_t.unsqueeze(0))[0]
    return pooled.detach().float().cpu().half()


def add_sparse_target(target, valid, context, action, gain, context_value, scales):
    k, h, w = target.shape
    cx, cy, scale = action
    ix = max(0, min(w - 1, int(cx * w)))
    iy = max(0, min(h - 1, int(cy * h)))
    si = int(np.argmin(np.abs(np.asarray(scales) - scale)))
    if not valid[si, iy, ix] or gain > float(target[si, iy, ix]):
        target[si, iy, ix] = float(gain)
        context[si, iy, ix] = float(context_value)
        valid[si, iy, ix] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--checkpoint", default=sup.DEFAULT_CKPT)
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "outputs/refzoom/cache_train"),
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="validate and reuse completed per-annotation records")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--scales", default="0.30,0.45,0.60,0.75")
    ap.add_argument("--random-crops", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--feature-grid", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records_dir = os.path.join(args.out, "records")
    os.makedirs(records_dir, exist_ok=True)

    scales = tuple(float(x) for x in args.scales.split(","))
    data = json.load(open(args.ann))
    images = {x["id"]: x for x in data["images"]}
    anns = data["annotations"]
    if args.max_samples is not None:
        anns = anns[: args.max_samples]
    anns_by_id = {a["id"]: a for a in anns}

    manifest = []
    completed_ids = set()
    if args.resume:
        for ann in anns:
            path = os.path.join(records_dir, f"{int(ann['id']):08d}.pt")
            if not os.path.isfile(path):
                continue
            rec = torch.load(path, map_location="cpu", weights_only=False)
            if int(rec["annotation_id"]) != int(ann["id"]):
                raise RuntimeError(f"cache annotation mismatch: {path}")
            required = ("global_visual", "candidate_features", "candidate_geometry", "candidate_iou", "action_meta")
            if any(k not in rec for k in required):
                raise RuntimeError(f"incomplete cache record: {path}")
            vals = rec["candidate_iou"].float()
            manifest.append({"annotation_id":ann["id"], "record":os.path.basename(path), "candidate_count":len(vals), "valid_zoom_cells":int(rec["gain_valid"].sum()), "base_iou":float(rec["base_iou"]), "oracle_candidate_iou":float(vals.max())})
            completed_ids.add(ann["id"])
        print("resumed_records", len(completed_ids), flush=True)

    def aligned_gt(ann):
        from PIL import Image
        imeta = images[ann["image_id"]]
        gt = sup.dec_rle(ann["segmentation"])
        shape = (int(imeta["height"]), int(imeta["width"]))
        if gt.shape != shape:
            gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize((shape[1], shape[0]), Image.Resampling.NEAREST)).astype(bool)
        return gt

    items = []
    for ann in anns:
        if ann["id"] in completed_ids:
            continue
        aid = ann["id"]
        imeta = images[ann["image_id"]]
        w, h = imeta["width"], imeta["height"]
        image_path = os.path.join(args.image_root, imeta["file_name"])
        gt = aligned_gt(ann)
        rng = random.Random(args.seed + int(aid))
        actions = sup.sample_actions(gt, scales, args.random_crops, rng)
        prompt = imeta["text_inst_input"]

        items.append(
            {
                "annotation_id": aid,
                "image_id": ann["image_id"],
                "image_path": image_path,
                "prompt": prompt,
                "source": "full",
                "action_id": -1,
                "kind": "full",
                "xyxy": (0, 0, w, h),
                "full_width": w,
                "full_height": h,
            }
        )
        for action_id, (kind, xyxy) in enumerate(actions):
            items.append(
                {
                    "annotation_id": aid,
                    "image_id": ann["image_id"],
                    "image_path": image_path,
                    "prompt": prompt,
                    "source": "zoom",
                    "action_id": action_id,
                    "kind": kind,
                    "xyxy": xyxy,
                    "full_width": w,
                    "full_height": h,
                }
            )

    print(
        "annotations", len(anns),
        "inputs", len(items),
        "avg_inputs", round(len(items) / max(1, len(anns)), 2),
        flush=True,
    )

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    model = build_sam3_image_model(
        bpe_path=DEFAULT_BPE,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        enable_segmentation=True,
        device="cuda",
        load_from_HF=False,
        inst_stage="1_2",
        adapter_config=DEFAULT_ADAPTER,
    ).to(device).eval()

    captured = {}

    # Capture exactly the frozen backbone outputs consumed by SAM3-I without
    # running the backbone twice. Module hooks on the checkpointed vision neck
    # are not reliable here, so wrap the public forward_image/forward_text calls.
    original_forward_image = model.backbone.forward_image
    original_forward_text = model.backbone.forward_text

    def captured_forward_image(*a, **kw):
        out = original_forward_image(*a, **kw)
        captured["visual"] = out["vision_features"]
        return out

    def captured_forward_text(*a, **kw):
        out = original_forward_text(*a, **kw)
        captured["text_memory"] = out["language_features"]
        captured["text_mask"] = out["language_mask"]
        return out

    model.backbone.forward_image = captured_forward_image
    model.backbone.forward_text = captured_forward_text

    ds = sup.CropDataset(items, build_transform())
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=sup.collate_batch,
        pin_memory=True,
    )
    post = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.threshold,
        to_cpu=False,
    )

    def finalize_record(aid, group):
        ann = anns_by_id[aid]
        gt = aligned_gt(ann)
        h, w = gt.shape
        full = [e for e in group if e["source"] == "full"]
        if len(full) != 1:
            raise RuntimeError(f"annotation {aid} has {len(full)} full entries")
        full = full[0]

        base_union = np.zeros_like(gt)
        for p in full["predictions"]:
            base_union |= sup.dec_rle(p["rle"])
        base_iou = sup.mask_iou(gt, base_union)

        candidate_features = []
        candidate_geometry_rows = []
        raw_scores = []
        candidate_iou = []
        candidate_meta = []

        # Explicit full-image union fallback.
        full_feats = full["candidate_features"].float()
        if len(full_feats):
            union_feature = full_feats.mean(dim=0)
            union_score = max(p["score"] for p in full["predictions"])
        else:
            union_feature = full["global_visual"].float().mean(dim=(-2, -1))
            union_score = 0.0
        candidate_features.append(union_feature)
        candidate_geometry_rows.append(
            make_candidate_geometry(
                base_union, "full", (0, 0, w, h), w, h, is_union=True
            )
        )
        raw_scores.append(union_score)
        candidate_iou.append(base_iou)
        candidate_meta.append({"source": "full_union", "action_id": -1, "index": -1})

        # Full and zoom individual candidates.
        for e in group:
            for j, p in enumerate(e["predictions"]):
                m = sup.dec_rle(p["rle"])
                candidate_features.append(e["candidate_features"][j].float())
                candidate_geometry_rows.append(
                    make_candidate_geometry(m, e["source"], e["xyxy"], w, h, is_union=False)
                )
                raw_scores.append(float(p["score"]))
                candidate_iou.append(sup.mask_iou(gt, m))
                candidate_meta.append(
                    {
                        "source": e["source"],
                        "action_id": e["action_id"],
                        "index": j,
                        "kind": e["kind"],
                    }
                )

        gain_target = torch.zeros(len(scales), args.feature_grid, args.feature_grid)
        gain_valid = torch.zeros_like(gain_target, dtype=torch.bool)
        context_target = torch.zeros_like(gain_target)
        gt_box = sup.mask_bbox(gt)
        context_box = sup.expanded_box(gt_box, 2.0, w, h)

        action_meta = []
        for e in [x for x in group if x["source"] == "zoom"]:
            vals = [sup.mask_iou(gt, sup.dec_rle(p["rle"])) for p in e["predictions"]]
            best_iou = max(vals, default=0.0)
            x0, y0, x1, y1 = e["xyxy"]
            scale = max((x1 - x0) / w, (y1 - y0) / h)
            action = [((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h, scale]
            context_value = sup.box_intersection_fraction(context_box, e["xyxy"])
            gain = best_iou - base_iou
            add_sparse_target(
                gain_target,
                gain_valid,
                context_target,
                action,
                gain,
                context_value,
                scales,
            )
            action_meta.append(
                {
                    "action_id": e["action_id"],
                    "kind": e["kind"],
                    "xyxy": e["xyxy"],
                    "action": action,
                    "candidate_iou": best_iou,
                    "oracle_gain": gain,
                    "context_target": context_value,
                }
            )

        rec = {
            "annotation_id": aid,
            "global_visual": full["global_visual"],
            "text_tokens": full["text_tokens"],
            "text_padding_mask": full["text_padding_mask"],
            "gain_target": gain_target,
            "gain_valid": gain_valid,
            "context_target": context_target,
            "candidate_features": torch.stack(candidate_features).half(),
            "candidate_geometry": torch.tensor(candidate_geometry_rows, dtype=torch.float32),
            "raw_scores": torch.tensor(raw_scores, dtype=torch.float32),
            "candidate_iou": torch.tensor(candidate_iou, dtype=torch.float32),
            "base_iou": float(base_iou),
            "candidate_meta": candidate_meta,
            "action_meta": action_meta,
        }
        path = os.path.join(records_dir, f"{int(aid):08d}.pt")
        temporary_path = path + ".tmp"
        torch.save(rec, temporary_path)
        os.replace(temporary_path, path)
        return (
            {
                "annotation_id": aid,
                "record": os.path.basename(path),
                "candidate_count": len(candidate_iou),
                "valid_zoom_cells": int(gain_valid.sum()),
                "base_iou": base_iou,
                "oracle_candidate_iou": max(candidate_iou, default=base_iou),
            }
        )

    pending = {}
    expected_counts = {}
    for item in items:
        aid = item["annotation_id"]
        expected_counts[aid] = expected_counts.get(aid, 0) + 1
    start = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step, (batch, infos) in enumerate(dl):
            batch = copy_data_to_device(batch, device, non_blocking=True)
            captured.clear()
            output = model(batch, "1_2")
            results = process_batch(output, batch, infos, post)

            visual = captured["visual"]
            text_memory = captured["text_memory"]
            text_mask = captured["text_mask"]
            # Collator deduplicates identical prompts in find_text_batch. Several
            # crops from one annotation therefore share one encoded text entry.
            # Map each query back to the corresponding unique text via text_ids.
            text_ids = batch.find_inputs[0].text_ids
            if text_memory.ndim != 3:
                raise RuntimeError(f"unexpected text memory shape {tuple(text_memory.shape)}")
            if text_memory.shape[1] != len(batch.find_text_batch):
                raise RuntimeError(
                    f"text memory/text batch mismatch: {tuple(text_memory.shape)} vs {len(batch.find_text_batch)}"
                )

            for i, (result, qi) in enumerate(zip(results, infos)):
                crop_preds = result.get("predictions") or []
                crop_rles = [p["rle"] for p in crop_preds]
                cand_features = pool_one_batch(visual[i : i + 1], crop_rles)

                mapped = []
                for pred in crop_preds:
                    mask = sup.dec_rle(pred["rle"])
                    full = np.zeros((qi.full_height, qi.full_width), dtype=bool)
                    # Map the crop's valid spatial extent onto the image canvas.
                    hh = min(mask.shape[0], qi.height, max(0, qi.full_height - qi.y0))
                    ww = min(mask.shape[1], qi.width, max(0, qi.full_width - qi.x0))
                    if hh > 0 and ww > 0:
                        full[qi.y0 : qi.y0 + hh, qi.x0 : qi.x0 + ww] = mask[:hh, :ww]
                    mapped.append(
                        {
                            "rle": sup.enc_rle(full),
                            "score": float(pred.get("score", 0.0)),
                        }
                    )

                entry = {
                    "annotation_id": qi.annotation_id,
                    "source": qi.source,
                    "action_id": qi.action_id,
                    "kind": qi.kind,
                    "xyxy": [qi.x0, qi.y0, qi.x0 + qi.width, qi.y0 + qi.height],
                    "predictions": mapped,
                    "candidate_features": cand_features,
                }
                if qi.source == "full":
                    entry["global_visual"] = (
                        F.adaptive_avg_pool2d(
                            visual[i : i + 1].float(),
                            (args.feature_grid, args.feature_grid),
                        )[0]
                        .cpu()
                        .half()
                    )
                    tid = int(text_ids[i].item()) if torch.is_tensor(text_ids) else int(text_ids[i])
                    entry["text_tokens"] = text_memory[:, tid, :].float().cpu().half()
                    entry["text_padding_mask"] = text_mask[tid].bool().cpu()
                aid = qi.annotation_id
                group = pending.setdefault(aid, [])
                group.append(entry)
                if len(group) == expected_counts[aid]:
                    manifest.append(finalize_record(aid, group))
                    del pending[aid]
                    if len(manifest) % 100 == 0:
                        print("saved_records", len(manifest), flush=True)

            if step % 100 == 0:
                print("step", step, "elapsed_sec", round(time.time() - start, 1), flush=True)

    # Restore methods before serializing/finishing.
    model.backbone.forward_image = original_forward_image
    model.backbone.forward_text = original_forward_text

    if pending:
        raise RuntimeError(f"incomplete annotation groups: {list(pending)}")
    manifest.sort(key=lambda x: int(x["annotation_id"]))

    meta = {
        "checkpoint": args.checkpoint,
        "annotation_count": len(manifest),
        "input_count": len(items),
        "resumed_records": len(completed_ids),
        "gt_alignment": "nearest to RGB canvas for cached training labels only; original dataset GT unchanged",
        "feature_grid": args.feature_grid,
        "scales": scales,
        "random_crops": args.random_crops,
        "seed": args.seed,
        "elapsed_sec": time.time() - start,
        "geometry_layout": [
            "mask_cx",
            "mask_cy",
            "mask_w",
            "mask_h",
            "source_is_zoom",
            "crop_scale",
            "is_full_union",
        ],
        "warning": "candidate_iou, base_iou, oracle_gain and GT-derived action metadata are training labels only",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE records", len(manifest), "elapsed_sec", round(meta["elapsed_sec"], 1), flush=True)


if __name__ == "__main__":
    main()
