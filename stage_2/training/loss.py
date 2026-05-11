"""Negative-pool construction and softmax CE loss for causal LM training.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §5.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_hard_negatives(candidates: torch.Tensor, k: int) -> torch.Tensor:
    """Sample k items per playlist from the candidate pool.

    candidates: (B, 1000) int64
    returns:    (B, k)    int64
    """
    B, C = candidates.shape
    idx = torch.randint(0, C, (B, k), device=candidates.device)
    return candidates.gather(1, idx)


def sample_random_negatives(k: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Sample k uniform random IDs from [1, vocab_size) (excluding PAD=0).

    Returns shape (k,) int64. Shared across the batch.
    """
    return torch.randint(1, vocab_size, (k,), device=device)


def build_logits(
    queries: torch.Tensor,        # (B, L, D)
    positives: torch.Tensor,      # (B, L)
    hard_negs: torch.Tensor,      # (B, K_hard) — shared across positions in playlist
    random_negs: torch.Tensor,    # (K_rand,)   — shared across the batch
    item_emb: nn.Embedding,
) -> torch.Tensor:
    """Compute scoring logits with positive at index 0.

    Returns: (B, L, 1 + K_hard + K_rand)
    """
    B, L, D = queries.shape

    pos_vec = item_emb(positives)                                  # (B, L, D)
    pos_logits = (queries * pos_vec).sum(-1, keepdim=True)         # (B, L, 1)

    hard_vec = item_emb(hard_negs)                                 # (B, K_hard, D)
    hard_logits = torch.einsum("bld,bkd->blk", queries, hard_vec)  # (B, L, K_hard)

    rand_vec = item_emb(random_negs)                               # (K_rand, D)
    rand_logits = torch.einsum("bld,kd->blk", queries, rand_vec)   # (B, L, K_rand)

    return torch.cat([pos_logits, hard_logits, rand_logits], dim=-1)


def masked_ce_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Softmax CE over the last dim with target=0 (positive at index 0), masked.

    logits:        (B, L, n_total)
    positive_mask: (B, L) bool
    """
    B, L, N = logits.shape
    target = torch.zeros(B, L, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(
        logits.reshape(B * L, N),
        target.reshape(B * L),
        reduction="none",
    ).reshape(B, L)
    denom = positive_mask.sum().clamp(min=1)
    return (loss * positive_mask).sum() / denom
