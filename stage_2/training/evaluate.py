"""Pipeline-mode and unconstrained evaluation for Stage 2.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §6.
"""

import math
import sys
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from stage_2.models.sasrec import SASRec


def _pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)


def _recall_at_k(top_k_ids: np.ndarray, targets: np.ndarray, k: int) -> float:
    """top_k_ids: (B, k), targets: (B,) → fraction where target in top_k."""
    return float((top_k_ids[:, :k] == targets[:, None]).any(axis=1).mean())


def _ndcg_at_k(top_k_ids: np.ndarray, targets: np.ndarray, k: int) -> float:
    """NDCG@K with binary relevance (single target). top_k_ids must be sorted best-first."""
    match = (top_k_ids[:, :k] == targets[:, None])             # (B, k) bool
    found = match.any(axis=1)
    ranks = match.argmax(axis=1) + 1                            # 1-indexed
    gain = np.where(found, 1.0 / np.log2(ranks + 1), 0.0)
    return float(gain.mean())


def evaluate_pipeline_mode(
    model: SASRec,
    padded: np.ndarray,
    lengths: np.ndarray,
    candidates: np.ndarray,
    batch_size: int = 256,
    device: str = "cuda",
    use_bf16: bool = False,
) -> Dict[str, float]:
    """Pipeline-mode eval: score this playlist's 1000 Stage 1 candidates.

    Returns dict with R@10, R@50, R@100, NDCG@10. Skips playlists with length < 2.
    """
    model.eval()
    N, L = padded.shape

    r_at = {10: 0, 50: 0, 100: 0}
    ndcg_total = 0.0
    counted = 0

    with torch.no_grad():
        for start in _pbar(range(0, N, batch_size), desc="val pipeline"):
            end = min(start + batch_size, N)
            B = end - start

            seqs_np = padded[start:end].copy()
            lens = lengths[start:end]
            cands_np = candidates[start:end]

            # Build seq = playlist[:-1] padded (zero out the last real token)
            seqs = seqs_np.copy()
            valid = lens >= 2
            for j in range(B):
                if not valid[j]:
                    continue
                # Position of the last real token = L - 1 (left-padded), so zero it out
                seqs[j, L - 1] = 0

            # Targets = last real token of original seq
            targets = np.zeros(B, dtype=np.int64)
            targets[valid] = seqs_np[valid, L - 1]

            seq_t = torch.from_numpy(seqs.astype(np.int64)).to(device)
            cand_t = torch.from_numpy(cands_np.astype(np.int64)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(seq_t)                              # (B, L, D)
                # Query = output at position L-2 (the new "last" after zeroing L-1)
                # But for left-padded sequences, the query should be at the position
                # right before the held-out token, which after zeroing is L-2.
                # Equivalent: use position L-1 of the zeroed sequence — but that's PAD now,
                # so we must use L-2.
                queries = out[:, L - 2, :]                      # (B, D)
                cand_emb = model.item_emb(cand_t)               # (B, 1000, D)
                scores = (queries.unsqueeze(1) * cand_emb).sum(-1)   # (B, 1000)
                # Mask context tokens (seqs after zeroing): set their scores to -inf
                for j in range(B):
                    if not valid[j]:
                        continue
                    real_seq = seqs_np[j][seqs_np[j] != 0]
                    ctx = real_seq[:-1]
                    if ctx.size == 0:
                        continue
                    ctx_t = torch.from_numpy(ctx.astype(np.int64)).to(device)
                    # Build a mask over candidates (vectorized)
                    isin = (cand_t[j].unsqueeze(0) == ctx_t.unsqueeze(1)).any(0)
                    scores[j, isin] = float("-inf")
                # Mask PAD candidates (id 0) defensively
                scores[cand_t == 0] = float("-inf")

            # Top-100 sorted, then trim for R@10/R@50/R@100/NDCG@10
            top_scores, top_idx = torch.topk(scores, k=100, dim=1)
            top_ids = torch.gather(cand_t, 1, top_idx).cpu().numpy()  # (B, 100)

            v = valid
            if v.sum() == 0:
                continue
            for k in (10, 50, 100):
                r_at[k] += int(((top_ids[v, :k]) == targets[v, None]).any(axis=1).sum())
            ndcg_total += _ndcg_at_k(top_ids[v], targets[v], k=10) * v.sum()
            counted += int(v.sum())

    metrics = {
        "R@10":   r_at[10] / max(1, counted),
        "R@50":   r_at[50] / max(1, counted),
        "R@100":  r_at[100] / max(1, counted),
        "NDCG@10": ndcg_total / max(1, counted),
        "n_evaluated": counted,
    }
    return metrics


def evaluate_unconstrained(
    model: SASRec,
    padded: np.ndarray,
    lengths: np.ndarray,
    vocab_size: int,
    batch_size: int = 256,
    device: str = "cuda",
    use_bf16: bool = False,
) -> Dict[str, float]:
    """Unconstrained eval: score against full vocab. ~10 min on A10G for 150K playlists."""
    model.eval()
    N, L = padded.shape
    r_at = {10: 0, 50: 0, 100: 0, 1000: 0}
    ndcg_total = 0.0
    counted = 0

    with torch.no_grad():
        item_emb_w = model.item_emb.weight                       # (vocab, D)
        for start in _pbar(range(0, N, batch_size), desc="test unconstrained"):
            end = min(start + batch_size, N)
            B = end - start

            seqs_np = padded[start:end].copy()
            lens = lengths[start:end]
            seqs = seqs_np.copy()
            valid = lens >= 2
            for j in range(B):
                if not valid[j]:
                    continue
                seqs[j, L - 1] = 0
            targets = np.zeros(B, dtype=np.int64)
            targets[valid] = seqs_np[valid, L - 1]

            seq_t = torch.from_numpy(seqs.astype(np.int64)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(seq_t)                               # (B, L, D)
                queries = out[:, L - 2, :]                       # (B, D)
                scores = queries @ item_emb_w.T                  # (B, vocab)
                scores[:, 0] = float("-inf")                     # mask PAD
                # Mask context tokens
                for j in range(B):
                    if not valid[j]:
                        continue
                    real_seq = seqs_np[j][seqs_np[j] != 0]
                    ctx = real_seq[:-1]
                    if ctx.size == 0:
                        continue
                    ctx_t = torch.from_numpy(ctx.astype(np.int64)).to(device)
                    scores[j, ctx_t] = float("-inf")

            top1000 = torch.topk(scores, k=1000, dim=1).indices.cpu().numpy()  # (B, 1000)

            v = valid
            if v.sum() == 0:
                continue
            for k in (10, 50, 100, 1000):
                r_at[k] += int(((top1000[v, :k]) == targets[v, None]).any(axis=1).sum())
            ndcg_total += _ndcg_at_k(top1000[v], targets[v], k=10) * v.sum()
            counted += int(v.sum())

    return {
        "R@10":    r_at[10] / max(1, counted),
        "R@50":    r_at[50] / max(1, counted),
        "R@100":   r_at[100] / max(1, counted),
        "R@1000":  r_at[1000] / max(1, counted),
        "NDCG@10": ndcg_total / max(1, counted),
        "n_evaluated": counted,
    }


def derived_metrics(
    pipeline: Dict[str, float],
    unconstrained: Dict[str, float],
    stage1_recall_at_100: float = 0.151,
    stage1_recall_at_1000: float = 0.407,
) -> Dict[str, float]:
    """Pipeline lift, headroom gap, Stage-1-recoverable rate (spec §6.4)."""
    return {
        "pipeline_lift_R@100": pipeline["R@100"] / max(1e-9, stage1_recall_at_100),
        "headroom_gap_R@100": unconstrained["R@100"] - pipeline["R@100"],
        "stage1_recoverable_rate": pipeline["R@10"] / max(1e-9, stage1_recall_at_1000),
    }
