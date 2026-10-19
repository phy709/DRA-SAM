#!/usr/bin/env python3
"""Train the simplified Analytic-Risk RAI evidence scorer."""

import argparse
import math
import os
import random
import sys
from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sam3"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from sam3.refzoom.analytic_rai import AnalyticRAI
from sam3.refzoom.analytic_rai_loss import AnalyticRAILoss
from rai_data import RAICacheDataset, collate_rai


def move(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def repeat_actions(x, a):
    return x[:, None].expand(
        x.shape[0], a, *x.shape[1:]
    ).reshape(x.shape[0] * a, *x.shape[1:])


def forward_all_actions(model, batch):
    prior = model.prior(
        text_tokens=batch["text_tokens"],
        text_padding_mask=batch["text_padding_mask"],
        global_visual=batch["global_visual"],
        candidate_features=batch["initial_features"],
        candidate_geometry=batch["initial_geometry"],
        raw_scores=batch["initial_raw"],
        candidate_valid=batch["initial_valid"],
    )
    look = model.look(
        prior=prior,
        hypothesis_geometry=batch["initial_geometry"],
        hypothesis_valid=batch["initial_valid"],
        actions=batch["actions"],
        action_valid=batch["action_valid"],
        mask_disagreement=None,  # Training diagnostics use geometric box disagreement.
    )

    b, a, k, c = batch["local_features"].shape
    m = batch["initial_features"].shape[1]

    update = model.update(
        prior_logits=repeat_actions(prior["logits"], a),
        text_state=repeat_actions(prior["text_state"], a),
        global_visual=repeat_actions(batch["global_visual"], a),
        prior_tokens=repeat_actions(prior["hypothesis_tokens"], a),
        prior_geometry=repeat_actions(batch["initial_geometry"], a),
        prior_valid=repeat_actions(batch["initial_valid"], a),
        local_features=batch["local_features"].reshape(b * a, k, c),
        local_geometry=batch["local_geometry"].reshape(b * a, k, -1),
        local_raw_scores=batch["local_raw"].reshape(b * a, k),
        local_valid=batch["local_valid"].reshape(b * a, k),
        action=batch["actions"].reshape(b * a, 3),
        unknown_crop_mass=look["unknown_crop_mass"].reshape(b * a),
    )
    posterior = update["probs"].reshape(b, a, m + k + 1)
    return prior, look, update, posterior


@torch.no_grad()
def proxy_metrics(model, prior, look, posterior, batch):
    b, a, k = batch["local_iou"].shape
    m = batch["initial_iou"].shape[1]

    utility = look["utility"].masked_fill(
        ~batch["action_valid"], -1e4
    )
    action_idx = utility.argmax(dim=-1)

    gather_post = action_idx[:, None, None].expand(
        -1, 1, m + k + 1
    )
    chosen_post = posterior.gather(1, gather_post).squeeze(1)

    local_iou = batch["local_iou"].gather(
        1, action_idx[:, None, None].expand(-1, 1, k)
    ).squeeze(1)
    local_valid = batch["local_valid"].gather(
        1, action_idx[:, None, None].expand(-1, 1, k)
    ).squeeze(1)

    concrete_iou = torch.cat(
        [batch["initial_iou"], local_iou], dim=-1
    )
    concrete_valid = torch.cat(
        [batch["initial_valid"], local_valid], dim=-1
    )
    decision = model.map_decision(
        chosen_post, concrete_valid, fallback_index=0
    )
    selected_iou = concrete_iou.gather(
        1, decision[:, None]
    ).squeeze(1)

    prior_decision = model.map_decision(
        prior["probs"], batch["initial_valid"], fallback_index=0
    )
    prior_iou = batch["initial_iou"].gather(
        1, prior_decision[:, None]
    ).squeeze(1)

    oracle = concrete_iou.masked_fill(
        ~concrete_valid, -1.0
    ).max(dim=-1).values

    return {
        "n": int(b),
        "prior_sum": float(prior_iou.sum()),
        "selected_sum": float(selected_iou.sum()),
        "oracle_sum": float(oracle.sum()),
        "p50": int((selected_iou >= 0.5).sum()),
        "zero": int((selected_iou == 0).sum()),
        "mean_risk_sum": float(look["referential_risk"].sum()),
        "mean_comp_sum": float(
            look["competition_utility"]
            .masked_fill(~batch["action_valid"], 0)
            .max(dim=-1).values.sum()
        ),
        "mean_disc_sum": float(
            look["discovery_utility"]
            .masked_fill(~batch["action_valid"], 0)
            .max(dim=-1).values.sum()
        ),
    }


def merge(dst, src):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def finalize(s):
    n = max(1, s.get("n", 0))
    return {
        "prior_mIoU": 100.0 * s.get("prior_sum", 0.0) / n,
        "one_step_mIoU": 100.0 * s.get("selected_sum", 0.0) / n,
        "one_step_oracle_mIoU": 100.0 * s.get("oracle_sum", 0.0) / n,
        "P@0.5": 100.0 * s.get("p50", 0) / n,
        "IoU=0": 100.0 * s.get("zero", 0) / n,
        "mean_referential_risk": s.get("mean_risk_sum", 0.0) / n,
        "max_competition_utility": s.get("mean_comp_sum", 0.0) / n,
        "max_discovery_utility": s.get("mean_disc_sum", 0.0) / n,
    }


@torch.no_grad()
def evaluate(model, loss_fn, loader, device, amp, amp_dtype):
    model.eval()
    loss_sums = {}
    stats = {}
    batches = 0
    for batch in loader:
        batch = move(batch, device)
        ctx = (
            torch.autocast("cuda", dtype=amp_dtype)
            if amp else nullcontext()
        )
        with ctx:
            prior, look, update, posterior = forward_all_actions(
                model, batch
            )
            losses = loss_fn(
                prior["probs"],
                posterior,
                batch["initial_iou"],
                batch["initial_valid"],
                batch["local_iou"],
                batch["local_valid"],
                batch["action_valid"],
            )
        for k, v in losses.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach())
        merge(
            stats,
            proxy_metrics(model, prior, look, posterior, batch),
        )
        batches += 1
    return (
        {k: v / max(1, batches) for k, v in loss_sums.items()},
        finalize(stats),
    )


def make_scheduler(optimizer, total_steps, warmup_steps):
    def scale(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )
    return LambdaLR(optimizer, scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = int(cfg["runtime"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
    amp = bool(cfg["runtime"].get("amp", True)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if cfg["runtime"].get("amp_dtype") == "bfloat16"
        else torch.float16
    )

    model = AnalyticRAI(
        dim=int(cfg["model"]["dim"]),
        geometry_dim=int(cfg["model"]["geometry_dim"]),
        hidden=int(cfg["model"]["hidden"]),
        unknown_weight=float(cfg["model"]["unknown_weight"]),
    ).to(device)
    loss_fn = AnalyticRAILoss(**cfg["loss"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    train_ds = RAICacheDataset(args.train_cache)
    val_ds = RAICacheDataset(args.val_cache)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_rai,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_rai,
        pin_memory=True,
        drop_last=False,
    )

    epochs = int(cfg["train"]["epochs"])
    total_steps = epochs * len(train_loader)
    warmup_steps = int(cfg["train"]["warmup_epochs"]) * len(train_loader)
    scheduler = make_scheduler(
        optimizer, total_steps, warmup_steps
    )

    start_epoch = 0
    global_step = 0
    best_proxy = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_proxy = float(ckpt.get("best_proxy", float("inf")))

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(
        "train_records", len(train_ds),
        "val_records", len(val_ds),
        "trainable_params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        flush=True,
    )
    print(
        "method STRICT_ANALYTIC_RAI "
        "one_scorer_plus_analytic_look_plus_additive_update",
        flush=True,
    )

    grad_clip = float(cfg["train"]["grad_clip"])
    for epoch in range(start_epoch, epochs):
        model.train()
        running = {}
        for step, batch in enumerate(train_loader):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            ctx = (
                torch.autocast("cuda", dtype=amp_dtype)
                if amp else nullcontext()
            )
            with ctx:
                prior, look, update, posterior = forward_all_actions(
                    model, batch
                )
                losses = loss_fn(
                    prior["probs"],
                    posterior,
                    batch["initial_iou"],
                    batch["initial_valid"],
                    batch["local_iou"],
                    batch["local_valid"],
                    batch["action_valid"],
                )
                loss = losses["loss_total"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            if step % 100 == 0:
                print(
                    "epoch", epoch,
                    "step", step,
                    "loss", round(float(loss.detach()), 5),
                    "lr", round(float(optimizer.param_groups[0]["lr"]), 8),
                    flush=True,
                )

        train_summary = {
            k: v / max(1, len(train_loader))
            for k, v in running.items()
        }
        val_loss, val_proxy = evaluate(
            model, loss_fn, val_loader, device, amp, amp_dtype
        )
        print("epoch", epoch, "train", train_summary, flush=True)
        print("epoch", epoch, "val_loss", val_loss, flush=True)
        print("epoch", epoch, "val_proxy", val_proxy, flush=True)

        score = float(val_loss["loss_total"])
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "train_loss": train_summary,
            "val_loss": val_loss,
            "val_proxy": val_proxy,
            "best_proxy": min(best_proxy, score),
        }
        if (epoch + 1) % int(cfg["train"]["save_every"]) == 0:
            path = os.path.join(
                out_dir, f"checkpoint_{epoch + 1}.pt"
            )
            torch.save(payload, path)
            print("saved", path, flush=True)

        if score < best_proxy:
            best_proxy = score
            payload["best_proxy"] = best_proxy
            path = os.path.join(out_dir, "best.pt")
            torch.save(payload, path)
            print(
                "saved_best", path,
                "val_loss_total", round(best_proxy, 6),
                flush=True,
            )


if __name__ == "__main__":
    main()
