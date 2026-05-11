"""Causal-LM training loop for Stage 2 SASRec.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §5.
"""

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage_2.data.dataset import SASRecDataset
from stage_2.models.sasrec import SASRec
from stage_2.training.evaluate import evaluate_pipeline_mode
from stage_2.training.loss import (
    build_logits,
    masked_ce_loss,
    sample_hard_negatives,
    sample_random_negatives,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms — slows things slightly, removes nondeterminism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)


def save_checkpoint(run_dir: Path, tag: str, model: SASRec):
    """Save a checkpoint as a directory with model.pt + item_embeddings.npy."""
    ckpt_dir = run_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Item embedding table as .npy (large, separate)
    embed = model.item_emb.weight.detach().cpu().float().numpy()
    np.save(ckpt_dir / "item_embeddings.npy", embed)

    # Everything else as state_dict
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k != "item_emb.weight"}
    torch.save(state, ckpt_dir / "model.pt")


def load_checkpoint(run_dir: Path, tag: str, model: SASRec):
    """Load a checkpoint produced by save_checkpoint."""
    ckpt_dir = run_dir / tag
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    embed = np.load(ckpt_dir / "item_embeddings.npy")
    with torch.no_grad():
        model.item_emb.weight.copy_(torch.from_numpy(embed))
    model.load_state_dict(state, strict=False)


def train_main(cfg: Dict, run_id: str, smoke: bool = False) -> Dict:
    """Full training loop. Returns a dict of summary metrics.

    If smoke=True: 1 epoch, no early-stop, exits after final val eval.
    """
    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_dir = Path(cfg["paths"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading derived data...")
    padded = np.load(cfg["paths"]["playlists_padded"])     # (N, L) int32
    lengths = np.load(cfg["paths"]["playlists_lengths"])   # (N,)   int16
    candidates = np.load(cfg["paths"]["candidates"])       # (N, 1000) int32

    n_train = cfg["data"]["train_size"]
    n_val = cfg["data"]["val_size"]
    train_idx = np.arange(n_train)
    # Filter train only: length >= min_train_length
    min_train = cfg["data"]["min_train_length"]
    train_idx = train_idx[lengths[train_idx] >= min_train]
    print(f"Train (filtered len>={min_train}): {len(train_idx):,}  Val: {n_val:,}")

    train_ds = SASRecDataset(
        playlists_padded=padded[train_idx],
        playlists_lengths=lengths[train_idx],
        candidates=candidates[train_idx],
        max_seq=cfg["model"]["max_seq_len"],
    )

    # Val subset
    val_start = n_train
    val_subset_size = cfg["data"]["val_eval_subset"]
    val_idx = np.arange(val_start, val_start + val_subset_size)
    val_padded = padded[val_idx]
    val_lengths = lengths[val_idx]
    val_candidates = candidates[val_idx]

    # ── Load ALS factors into model ──────────────────────────────────────────
    print("Loading ALS factors into model...")
    als_raw = np.load(cfg["paths"]["als_factors"]).astype(np.float32)
    model = SASRec(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_als(torch.from_numpy(als_raw).to(device))

    # ── Optimizer with two parameter groups ──────────────────────────────────
    embed_params, other_params = model.param_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": cfg["training"]["peak_lr_embedding"],
             "weight_decay": 0.0},
            {"params": other_params, "lr": cfg["training"]["peak_lr_transformer"],
             "weight_decay": cfg["training"]["weight_decay"]},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    epochs = 1 if smoke else cfg["training"]["epochs"]
    steps_per_epoch = math.ceil(len(train_ds) / cfg["training"]["batch_size"])
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(cfg["training"]["warmup_steps"], total_steps),
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    use_bf16 = (cfg["training"]["precision"] == "bf16") and (device == "cuda")
    history = []
    best_ndcg = -1.0
    patience_left = cfg["training"]["early_stopping_patience_evals"]
    last_ckpt_tag = None
    best_ckpt_tag = None
    t_start = time.time()

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum, epoch_loss_n = 0.0, 0

        epoch_bar = pbar(loader, desc=f"epoch {epoch}/{epochs}")
        for seq, positives, pos_mask, cand in epoch_bar:
            seq = seq.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            pos_mask = pos_mask.to(device, non_blocking=True)
            cand = cand.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                queries = model(seq)
                hn = sample_hard_negatives(cand, k=cfg["training"]["num_hard_negs"])
                rn = sample_random_negatives(
                    k=cfg["training"]["num_random_negs"],
                    vocab_size=cfg["model"]["vocab_size"],
                    device=seq.device,
                )
                logits = build_logits(queries, positives, hn, rn, model.item_emb)
                loss = masked_ce_loss(logits, pos_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += loss.item()
            epoch_loss_n += 1
            epoch_bar.set_postfix({
                "loss": f"{epoch_loss_sum / epoch_loss_n:.3f}",
                "lr": f"{scheduler.get_last_lr()[1]:.2e}",
            })

        avg_train_loss = epoch_loss_sum / max(1, epoch_loss_n)

        # ── Validation (every val_every_n_epochs OR always for smoke) ────────
        should_eval = smoke or (epoch % cfg["training"]["val_every_n_epochs"] == 0)
        if should_eval:
            val_metrics = evaluate_pipeline_mode(
                model=model,
                padded=val_padded,
                lengths=val_lengths,
                candidates=val_candidates,
                batch_size=cfg["training"]["batch_size"],
                device=device,
                use_bf16=use_bf16,
            )
            wall_min = (time.time() - t_start) / 60
            print(
                f"[epoch {epoch}/{epochs}] "
                f"train_loss={avg_train_loss:.3f}  "
                f"val_R@10={val_metrics['R@10']:.3f}  "
                f"val_R@100={val_metrics['R@100']:.3f}  "
                f"val_NDCG@10={val_metrics['NDCG@10']:.3f}  "
                f"wall={wall_min:.1f}m",
                flush=True,
            )
            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val": val_metrics,
                "wall_min": wall_min,
            })

            # Checkpoint management
            new_tag = f"ckpt_epoch_{epoch}"
            save_checkpoint(run_dir, new_tag, model)

            if val_metrics["NDCG@10"] > best_ndcg:
                best_ndcg = val_metrics["NDCG@10"]
                # Promote new ckpt to "best"
                if best_ckpt_tag is not None and best_ckpt_tag != last_ckpt_tag:
                    shutil.rmtree(run_dir / best_ckpt_tag, ignore_errors=True)
                best_ckpt_tag = new_tag
                patience_left = cfg["training"]["early_stopping_patience_evals"]
            else:
                # Delete the non-best, non-last older ckpt
                if last_ckpt_tag is not None and last_ckpt_tag != best_ckpt_tag:
                    shutil.rmtree(run_dir / last_ckpt_tag, ignore_errors=True)
                patience_left -= 1

            last_ckpt_tag = new_tag

            if (not smoke) and patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val improvement).")
                break

    # ── Final: save best + last as their final names ─────────────────────────
    if best_ckpt_tag is not None:
        shutil.copytree(run_dir / best_ckpt_tag, run_dir / "best", dirs_exist_ok=True)
    if last_ckpt_tag is not None:
        shutil.copytree(run_dir / last_ckpt_tag, run_dir / "final", dirs_exist_ok=True)

    # Save history
    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"best_ndcg": best_ndcg, "history": history, "run_dir": str(run_dir)}
