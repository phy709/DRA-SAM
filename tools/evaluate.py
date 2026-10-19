#!/usr/bin/env python3
"""DRA-SAM candidate construction, active observation and segmentation evaluation."""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/sam3")
sys.path.insert(0, ROOT + "/scripts")
sys.path.insert(0, ROOT + "/tools")

import data_utils as sup
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.refzoom import masked_pool_candidates
from sam3.refzoom.analytic_rai import AnalyticRAI, masked_softmax
from inference import DEFAULT_ADAPTER, DEFAULT_BPE, build_transform, process_batch

IMG_ROOT = os.path.join(ROOT, "data", "images")
SAM_CKPT = os.path.join(ROOT, "checkpoints", "stage3.pt")
ANALYTIC_CKPT = os.path.join(ROOT, "checkpoints", "evidence.pt")


def align_mask_nearest(mask, target_shape):
    """Align a decoded binary mask to target H,W using nearest-neighbor indexing."""
    if tuple(mask.shape) == tuple(target_shape):
        return mask
    h, w = mask.shape[:2]
    th, tw = target_shape[:2]
    yi = np.minimum((np.arange(th) * h // th), h - 1)
    xi = np.minimum((np.arange(tw) * w // tw), w - 1)
    return mask[np.ix_(yi, xi)]


def merge_preds(preds, shape):
    out = np.zeros(shape, dtype=bool)
    for p in preds or []:
        out |= align_mask_nearest(sup.dec_rle(p["rle"]), shape)
    return out


def make_geom(mask, source, crop_xyxy, W, H, is_union=False):
    x0, y0, x1, y1 = sup.mask_bbox(mask)
    cx = ((x0 + x1) / 2) / W
    cy = ((y0 + y1) / 2) / H
    bw = (x1 - x0) / W
    bh = (y1 - y0) / H
    crop_scale = max(
        (crop_xyxy[2] - crop_xyxy[0]) / W,
        (crop_xyxy[3] - crop_xyxy[1]) / H,
    )
    return [
        cx, cy, bw, bh,
        1.0 if source == "zoom" else 0.0,
        crop_scale,
        1.0 if is_union else 0.0,
    ]


def pool_features(feature_map, pred_rles):
    if not pred_rles:
        return torch.empty(0, feature_map.shape[1])
    masks = np.stack([sup.dec_rle(x) for x in pred_rles], axis=0)
    mt = torch.from_numpy(masks).to(feature_map.device)
    return masked_pool_candidates(feature_map, mt.unsqueeze(0))[0].float().cpu()


def exact_disagreement(masks):
    n = len(masks)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a = masks[i]
            b = masks[j]
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            iou = float(inter / union) if union else 1.0
            d[i, j] = d[j, i] = 1.0 - iou
    return d


def fixed_action_grid(scales=(0.30, 0.45, 0.60, 0.75), cells=24):
    actions = []
    for scale in scales:
        xs = np.linspace(scale / 2, 1.0 - scale / 2, cells)
        ys = np.linspace(scale / 2, 1.0 - scale / 2, cells)
        for y in ys:
            for x in xs:
                actions.append([float(x), float(y), float(scale)])
    return torch.tensor(actions, dtype=torch.float32)


def crop_cover_map(action, H=24, W=24, temperature=28.0, device=None):
    dtype = action.dtype
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy, scale = action
    half = scale / 2.0
    wx = torch.sigmoid(temperature * (half - (xx - cx).abs()))
    wy = torch.sigmoid(temperature * (half - (yy - cy).abs()))
    return wx * wy


def update_unknown_spatial(q, action):
    cover = crop_cover_map(
        action, H=q.shape[-2], W=q.shape[-1],
        device=q.device,
    )
    nxt = q * (1.0 - cover).clamp_min(0.0)
    z = nxt.sum()
    if float(z) <= 1e-8:
        return torch.full_like(q, 1.0 / q.numel())
    return nxt / z


def compress_update(update, prior_count, local_count):
    # Evaluation is one sample at a time. The updater may receive one invalid
    # dummy local hypothesis when SAM3-I returns no local mask.
    if local_count > 0:
        return {
            "logits": update["logits"],
            "probs": update["probs"],
            "tokens": update["expanded_tokens"],
            "geometry": update["expanded_geometry"],
            "valid": update["expanded_valid"],
        }

    logits = torch.cat(
        [update["logits"][:, :prior_count], update["logits"][:, -1:]],
        dim=-1,
    )
    valid = torch.cat(
        [
            update["valid"][:, :prior_count],
            update["valid"][:, -1:],
        ],
        dim=-1,
    )
    probs = masked_softmax(logits, valid)
    return {
        "logits": logits,
        "probs": probs,
        "tokens": update["expanded_tokens"][:, :prior_count],
        "geometry": update["expanded_geometry"][:, :prior_count],
        "valid": update["expanded_valid"][:, :prior_count],
    }


def compute_metrics(rows, key):
    vals = []
    inter = 0
    union = 0
    for r in rows:
        gt = r["_gt"]
        mask = r[key]
        iou = sup.mask_iou(gt, mask)
        vals.append(iou)
        inter += np.logical_and(gt, mask).sum()
        union += np.logical_or(gt, mask).sum()
    vals = np.asarray(vals, dtype=float)
    return {
        "n": len(vals),
        "mIoU": float(vals.mean() * 100),
        "oIoU": float(inter / union * 100) if union else 0.0,
        "P@0.5": float((vals >= 0.5).mean() * 100),
        "P@0.6": float((vals >= 0.6).mean() * 100),
        "P@0.7": float((vals >= 0.7).mean() * 100),
        "P@0.8": float((vals >= 0.8).mean() * 100),
        "P@0.9": float((vals >= 0.9).mean() * 100),
        "IoU=0": float((vals == 0).mean() * 100),
    }


def build_initial_state(
    model, gvis, text_tokens, text_mask, base_preds, H, W, device
):
    base_mask = merge_preds(base_preds, (H, W))
    indiv_feats = pool_features(
        gvis, [p["rle"] for p in base_preds]
    )
    indiv_masks = [sup.dec_rle(p["rle"]) for p in base_preds]
    indiv_scores = [float(p.get("score", 0.0)) for p in base_preds]

    if len(indiv_feats):
        union_feat = indiv_feats.mean(0)
        union_score = max(indiv_scores)
    else:
        union_feat = gvis[0].mean(dim=(-2, -1)).detach().float().cpu()
        union_score = 0.0

    masks = [base_mask] + indiv_masks
    features = [union_feat] + [x for x in indiv_feats]
    scores = [union_score] + indiv_scores
    geometry = [
        make_geom(base_mask, "full", (0, 0, W, H), W, H, True)
    ]
    geometry.extend(
        make_geom(m, "full", (0, 0, W, H), W, H, False)
        for m in indiv_masks
    )

    cf = torch.stack([x.float() for x in features]).unsqueeze(0).to(device)
    cg = torch.tensor(
        geometry, dtype=torch.float32, device=device
    ).unsqueeze(0)
    rs = torch.tensor(
        scores, dtype=torch.float32, device=device
    ).unsqueeze(0)
    valid = torch.ones(
        1, len(masks), dtype=torch.bool, device=device
    )

    prior = model.prior(
        text_tokens=text_tokens.float(),
        text_padding_mask=text_mask,
        global_visual=gvis.float(),
        candidate_features=cf,
        candidate_geometry=cg,
        raw_scores=rs,
        candidate_valid=valid,
    )
    return {
        "base_mask": base_mask,
        "masks": masks,
        "raw_scores": scores,
        "text_state": prior["text_state"],
        "hypothesis_tokens": prior["hypothesis_tokens"],
        "geometry": cg,
        "valid": valid,
        "logits": prior["logits"],
        "probs": prior["probs"],
        "unknown_spatial": prior["unknown_spatial"],
        "text_tokens": text_tokens.detach().cpu(),
        "gvis": gvis.detach().cpu(),
        "text_mask": text_mask.detach().cpu(),
        "candidate_features": cf.detach().cpu(),
        "candidate_raw": rs.detach().cpu(),
    }


def select_action(model, state, actions, device):
    """Select an observation region using referential utility."""
    a = actions.to(device).unsqueeze(0)
    action_valid = torch.ones(1, a.shape[1], dtype=torch.bool, device=device)
    look = model.look(
        prior={"probs": state["probs"], "unknown_spatial": state["unknown_spatial"]},
        hypothesis_geometry=state["geometry"],
        hypothesis_valid=state["valid"],
        actions=a,
        action_valid=action_valid,
        mask_disagreement=torch.from_numpy(exact_disagreement(state["masks"])).to(device).unsqueeze(0),
    )
    idx = int(look["utility"][0].argmax().item())
    diag = {
        "utility": float(look["utility"][0, idx]),
        "unknown_mass": float(look["unknown_crop_mass"][0, idx]),
        "competition_utility": float(look["competition_utility"][0, idx]),
        "discovery_utility": float(look["discovery_utility"][0, idx]),
        "referential_risk": float(look["referential_risk"][0]),
    }
    return idx, actions[idx].clone(), diag


def make_crop_item(aid, ann, imeta, action, step):
    W, H = imeta["width"], imeta["height"]
    cx, cy, scale = [float(x) for x in action]
    box = sup.crop_from_center(cx * W, cy * H, scale, W, H)
    return {
        "annotation_id": aid,
        "image_id": ann["image_id"],
        "image_path": os.path.join(IMG_ROOT, imeta["file_name"]),
        "prompt": imeta["text_inst_input"],
        "source": "zoom",
        "action_id": step,
        "kind": "analytic_rai",
        "xyxy": box,
        "full_width": W,
        "full_height": H,
    }


def local_inference(sam, items, batch_size, device, post, t0):
    if not items:
        return {}
    loader = DataLoader(
        sup.CropDataset(items, build_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=sup.collate_batch,
        pin_memory=True,
    )
    local_by = {}
    captured = {}
    orig = sam.backbone.forward_image

    def cap(*a, **kw):
        out = orig(*a, **kw)
        captured["visual"] = out["vision_features"]
        return out

    sam.backbone.forward_image = cap
    try:
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            for bi, (batch, infos) in enumerate(loader):
                batch = copy_data_to_device(
                    batch, device, non_blocking=True
                )
                captured.clear()
                out = sam(batch, "1_2")
                results = process_batch(out, batch, infos, post)
                visual = captured["visual"]
                for i, (res, qi) in enumerate(zip(results, infos)):
                    preds = res.get("predictions") or []
                    feats = pool_features(
                        visual[i:i+1], [p["rle"] for p in preds]
                    )
                    entries = []
                    for j, p in enumerate(preds):
                        lm = sup.dec_rle(p["rle"])
                        full = np.zeros(
                            (qi.full_height, qi.full_width), dtype=bool
                        )
                        full[
                            qi.y0:qi.y0 + qi.height,
                            qi.x0:qi.x0 + qi.width,
                        ] = lm[:qi.height, :qi.width]
                        entries.append({
                            "mask": full,
                            "feature": feats[j],
                            "score": float(p.get("score", 0.0)),
                            "crop_xyxy": [
                                qi.x0, qi.y0,
                                qi.x0 + qi.width,
                                qi.y0 + qi.height,
                            ],
                        })
                    local_by[qi.annotation_id] = entries
                if bi % 25 == 0:
                    print(
                        "local_batch", bi,
                        "elapsed", round(time.time() - t0, 1),
                        flush=True,
                    )
    finally:
        sam.backbone.forward_image = orig
    return local_by


def apply_observation(model, state, entries, action, W, H, gvis, device):
    n = len(entries)
    k = max(1, n)
    lf = torch.zeros(1, k, 256, device=device)
    lg = torch.zeros(1, k, 7, device=device)
    lr = torch.zeros(1, k, device=device)
    lv = torch.zeros(1, k, dtype=torch.bool, device=device)

    if n:
        for j, e in enumerate(entries):
            lf[0, j] = e["feature"].float().to(device)
            lg[0, j] = torch.tensor(
                make_geom(
                    e["mask"], "zoom", e["crop_xyxy"],
                    W, H, False,
                ),
                dtype=torch.float32,
                device=device,
            )
            lr[0, j] = e["score"]
            lv[0, j] = True

    action_t = action.to(device).unsqueeze(0)
    qmass_t = model.look(
        prior={
            "probs": state["probs"],
            "unknown_spatial": state["unknown_spatial"],
        },
        hypothesis_geometry=state["geometry"],
        hypothesis_valid=state["valid"],
        actions=action_t[:, None, :],
        action_valid=torch.ones(1, 1, dtype=torch.bool, device=device),
        mask_disagreement=torch.from_numpy(
            exact_disagreement(state["masks"])
        ).to(device).unsqueeze(0),
    )["unknown_crop_mass"][:, 0]

    with torch.inference_mode():
        upd = model.update(
            prior_logits=state["logits"],
            text_state=state["text_state"],
            global_visual=gvis.float(),
            prior_tokens=state["hypothesis_tokens"],
            prior_geometry=state["geometry"],
            prior_valid=state["valid"],
            local_features=lf,
            local_geometry=lg,
            local_raw_scores=lr,
            local_valid=lv,
            action=action_t,
            unknown_crop_mass=qmass_t,
        )

    compact = compress_update(
        upd,
        prior_count=state["geometry"].shape[1],
        local_count=n,
    )
    state["logits"] = compact["logits"]
    state["probs"] = compact["probs"]
    state["hypothesis_tokens"] = compact["tokens"]
    state["geometry"] = compact["geometry"]
    state["valid"] = compact["valid"]
    state["unknown_spatial"] = update_unknown_spatial(
        state["unknown_spatial"][0], action.to(device)
    ).unsqueeze(0)
    state["masks"].extend(e["mask"] for e in entries)
    state["raw_scores"].extend(e["score"] for e in entries)
    return state


def main():
    ap = argparse.ArgumentParser()
    global IMG_ROOT
    ap.add_argument("--image-root", default=IMG_ROOT, help="RGB image directory")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--subset-name", required=True)
    ap.add_argument("--selected-manifest", default=None)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sam-checkpoint", default=SAM_CKPT, help="frozen SAM3-I checkpoint; defaults to RIS-LAD Stage3")
    ap.add_argument("--checkpoint", default=ANALYTIC_CKPT)
    ap.add_argument("--unknown-weight", type=float, default=None, help="override checkpoint unknown discovery weight at inference")
    ap.add_argument("--posterior-temperature", type=float, default=1.2, help="inference-time posterior temperature; >1 softens belief")
    ap.add_argument("--save-visual-masks", action="store_true", help="save GT/base/final/oracle masks as PNGs for selected-sample inspection")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    IMG_ROOT = args.image_root

    os.makedirs(args.out, exist_ok=True)
    data = json.load(open(args.ann))
    imgs = {x["id"]: x for x in data["images"]}
    anns = {a["id"]: a for a in data["annotations"]}
    baseline = {
        x["annotation_id"]: x for x in json.load(open(args.baseline))
    }

    if args.selected_manifest:
        selected = json.load(open(args.selected_manifest))
        aids = [x["annotation_id"] for x in selected]
        selected_meta = {
            x["annotation_id"]: x for x in selected
        }
    else:
        aids = list(anns.keys())
        selected_meta = {}
    if args.max_samples > 0:
        aids = aids[:args.max_samples]

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    sam = build_sam3_image_model(
        bpe_path=DEFAULT_BPE,
        checkpoint_path=args.sam_checkpoint,
        eval_mode=True,
        enable_segmentation=True,
        device="cuda",
        load_from_HF=False,
        inst_stage="1_2",
        adapter_config=DEFAULT_ADAPTER,
    ).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    unknown_weight = float(cfg["model"]["unknown_weight"]) if args.unknown_weight is None else float(args.unknown_weight)
    rai = AnalyticRAI(
        dim=int(cfg["model"]["dim"]),
        geometry_dim=int(cfg["model"]["geometry_dim"]),
        hidden=int(cfg["model"]["hidden"]),
        unknown_weight=unknown_weight,
        posterior_temperature=float(args.posterior_temperature),
    ).to(device).eval()
    rai.load_state_dict(ckpt["model"], strict=True)
    actions = fixed_action_grid(cells=24)
    post = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    full_items = []
    for aid in aids:
        a = anns[aid]
        imeta = imgs[a["image_id"]]
        full_items.append({
            "annotation_id": aid,
            "image_id": a["image_id"],
            "image_path": os.path.join(
                IMG_ROOT, imeta["file_name"]
            ),
            "prompt": imeta["text_inst_input"],
            "source": "full",
            "action_id": -1,
            "kind": "full",
            "xyxy": (0, 0, imeta["width"], imeta["height"]),
            "full_width": imeta["width"],
            "full_height": imeta["height"],
        })

    loader = DataLoader(
        sup.CropDataset(full_items, build_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=sup.collate_batch,
        pin_memory=True,
    )
    states = {}
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        for bi, (batch, infos) in enumerate(loader):
            batch = copy_data_to_device(
                batch, device, non_blocking=True
            )
            vis = sam.backbone.forward_image(
                batch.img_batch
            )["vision_features"]
            captions = [
                x["complex_query"][0] for x in batch.find_text_batch
            ]
            tout = sam.backbone.forward_text(
                captions=captions,
                text_type=4,
                device=device,
            )
            tids = batch.find_inputs[0].text_ids
            text_tokens = tout["language_features"][
                :, tids, :
            ].permute(1, 0, 2).contiguous()
            text_mask = tout["language_mask"][tids]
            gvis = F.adaptive_avg_pool2d(
                vis.float(), (24, 24)
            )

            for i, qi in enumerate(infos):
                aid = qi.annotation_id
                a = anns[aid]
                imeta = imgs[a["image_id"]]
                H, W = imeta["height"], imeta["width"]
                base_preds = baseline[aid].get(
                    "predictions"
                ) or []
                st = build_initial_state(
                    rai,
                    gvis[i:i+1],
                    text_tokens[i:i+1],
                    text_mask[i:i+1],
                    base_preds,
                    H, W,
                    device,
                )
                st["gvis"] = gvis[i:i+1].detach().cpu().half()
                st["trace"] = []
                states[aid] = st

            if bi % 20 == 0:
                print(
                    "global_batch", bi,
                    "elapsed", round(time.time() - t0, 1),
                    flush=True,
                )

    # Acquire the selected local observation for each referring query.
    crop_items = []
    chosen = {}
    with torch.inference_mode():
        for aid in aids:
            st = states[aid]
            idx, action, diag = select_action(rai, st, actions, device)
            st["trace"].append({
                "step": 0,
                "skipped": False,
                "action_index": idx,
                "action": [float(x) for x in action],
                "pre_update_top_prob": float(st["probs"][0, :-1].max().item()),
                **diag,
            })
            chosen[aid] = action
            a = anns[aid]
            crop_items.append(make_crop_item(aid, a, imgs[a["image_id"]], action, 0))

    local_by = local_inference(sam, crop_items, args.batch_size, device, post, t0)
    with torch.inference_mode():
        for aid in aids:
            st = states[aid]
            a = anns[aid]
            imeta = imgs[a["image_id"]]
            H, W = imeta["height"], imeta["width"]
            entries = local_by.get(aid, [])
            st["trace"][-1]["local_candidate_count"] = len(entries)
            st["gvis"] = st["gvis"].to(device)
            apply_observation(rai, st, entries, chosen[aid], W, H, st["gvis"].float(), device)
            st["gvis"] = st["gvis"].cpu().half()
            post_top = float(st["probs"][0, :-1].max().item())
            st["trace"][-1]["post_update_top_prob"] = post_top
            st["trace"][-1]["confidence_gain"] = post_top - st["trace"][-1]["pre_update_top_prob"]
    print("finished_observation", "elapsed", round(time.time() - t0, 1), flush=True)

    rows = []
    for aid in aids:
        a = anns[aid]
        gt = sup.dec_rle(a["segmentation"])
        st = states[aid]
        concrete_valid = st["valid"]
        idx = int(rai.map_decision(st["probs"], concrete_valid, fallback_index=0)[0].item())
        unknown = float(st["probs"][0, -1])
        best_concrete = float(
            st["probs"][0, :-1].max().item()
        )
        final = st["masks"][idx]
        base = st["base_mask"]
        cand_ious = [sup.mask_iou(gt, m) for m in st["masks"]]
        oracle_idx = int(np.argmax(cand_ious))
        local_idx = int(rai.map_decision(
            st["probs"], concrete_valid, fallback_index=0
        )[0].item())
        local_mask = st["masks"][local_idx]
        base_inter = int(np.logical_and(gt, base).sum())
        base_union = int(np.logical_or(gt, base).sum())
        local_inter = int(np.logical_and(gt, local_mask).sum())
        local_union = int(np.logical_or(gt, local_mask).sum())
        rows.append({
            "annotation_id": aid,
            "base_iou": sup.mask_iou(gt, base),
            "base_inter": base_inter,
            "base_union": base_union,
            "local_inter": local_inter,
            "local_union": local_union,
            "iou": sup.mask_iou(gt, final),
            "oracle_iou": float(cand_ious[oracle_idx]),
            "selected_index": idx,
            "candidate_count": len(st["masks"]),
            "unknown_prob": unknown,
            "best_concrete_prob": best_concrete,
            "unknown_fallback": bool(unknown > best_concrete),
            "trace": st["trace"],
            "stratum": selected_meta.get(aid, {}).get("stratum"),
            "_gt": gt,
            "_base": base,
            "_final": final,
            "_oracle": st["masks"][oracle_idx],
        })

    base_m = compute_metrics(rows, "_base")
    final_m = compute_metrics(rows, "_final")
    oracle_m = compute_metrics(rows, "_oracle")
    zeros = [r for r in rows if r["base_iou"] == 0]
    highs = [r for r in rows if r["base_iou"] >= 0.5]

    scale_counter = Counter()
    for r in rows:
        for tr in r["trace"]:
            if not tr.get("skipped", False):
                scale_counter[str(round(tr["action"][2], 2))] += 1

    summary = {
        "image_root": IMG_ROOT,
        "annotation_file": args.ann,
        "baseline_file": args.baseline,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt["epoch"]) + 1,
        "subset": args.subset_name,
        "grid_cells": 24,
        "action_count": int(actions.shape[0]),
        "action_scales": [float(x) for x in sorted(set(actions[:, 2].tolist()))],
        "budget": 1,
        "look_strategy": "analytic",
        "selector_version": "referential_utility",
        "unknown_weight": rai.unknown_weight,
        "posterior_temperature": rai.posterior_temperature,
        "posterior_update_version": rai.update_rule_version,
        "empty_local_probability_rule": "masked_softmax",
        "final_keep_global_rate_pct": float(100.0 * sum(r["selected_index"] == 0 for r in rows) / len(rows)),
        "mean_observations": float(sum(sum(not t.get("skipped", False) for t in r["trace"]) for r in rows) / len(rows)),
        "base": base_m,
        "analytic_rai": final_m,
        "observed_candidate_oracle": oracle_m,
        "unknown_fallback_rate_pct": float(
            np.mean([r["unknown_fallback"] for r in rows]) * 100
        ),
        "switch_rate_pct": float(
            np.mean([r["selected_index"] != 0 for r in rows]) * 100
        ),
        "zero_count": len(zeros),
        "zero_recovery_nonzero": sum(r["iou"] > 0 for r in zeros),
        "zero_recovery_iou50": sum(r["iou"] >= 0.5 for r in zeros),
        "high_iou_count": len(highs),
        "high_iou_decreased": sum(
            r["iou"] < r["base_iou"] for r in highs
        ),
        "high_iou_fell_below_50": sum(
            r["iou"] < 0.5 for r in highs
        ),
        "action_scale_counts": dict(scale_counter),
        "elapsed_sec": time.time() - t0,
    }

    if args.save_visual_masks:
        from PIL import Image
        mask_dir = os.path.join(args.out, "visual_masks")
        os.makedirs(mask_dir, exist_ok=True)
        for r in rows:
            for label, key in (("gt", "_gt"), ("base", "_base"), ("final", "_final"), ("oracle", "_oracle")):
                Image.fromarray(r[key].astype(np.uint8) * 255).save(
                    os.path.join(mask_dir, f"{r['annotation_id']}_{label}.png")
                )

    serial = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in rows
    ]
    json.dump(
        serial,
        open(os.path.join(args.out, "rows.json"), "w"),
        indent=2,
    )
    json.dump(
        summary,
        open(os.path.join(args.out, "summary.json"), "w"),
        indent=2,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
