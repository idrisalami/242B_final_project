"""Vectorized greedy Maximal Marginal Relevance (MMR) re-ranking.

See docs/superpowers/specs/2026-05-11-stage3-mmr-design.md §4-§5.

Inputs to the core function are already-resolved: candidate IDs and their
relevance scores from Stage 2, plus L2-normalized item embeddings for the
diversity term. Operates entirely in NumPy; no PyTorch dependency.
"""

from typing import Tuple

import numpy as np


def minmax_normalize_per_row(x: np.ndarray) -> np.ndarray:
    """Rescale each row of x to [0, 1] independently.

    Rows that are constant (max == min) map to all zeros — those playlists
    have no relevance signal to leverage, MMR falls back to pure diversity
    after the first pick.
    """
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    span = x_max - x_min
    span = np.where(span > 0, span, 1.0)
    return ((x - x_min) / span).astype(np.float32)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. Zero rows map to zero rows."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    return (x / norm).astype(np.float32)


def mmr_batch(
    candidates: np.ndarray,            # (B, C) int32 — candidate IDs
    rel_scores: np.ndarray,            # (B, C) float32 — Stage 2 scores (raw or normalized)
    item_emb_normed: np.ndarray,       # (V, D) float32 — L2-normalized
    K: int = 20,
    lam: float = 0.5,
    normalize_rel: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy MMR over a batch of B playlists, picking K items each.

    Returns:
        out_ids:    (B, K) int32 — selected IDs in selection order
        out_scores: (B, K) float32 — MMR composite score at pick time

    Algorithm: at each step, for each playlist, compute
        score(c) = lam * rel(c) - (1-lam) * max_sim_to_selected(c)
    and pick argmax over unselected candidates. The first pick uses pure
    relevance (no selected items yet).

    O(B * K * C * D) total work, fully vectorized over B and C.
    """
    B, C = candidates.shape
    assert rel_scores.shape == (B, C)
    assert item_emb_normed.ndim == 2

    rel = minmax_normalize_per_row(rel_scores) if normalize_rel else rel_scores.astype(np.float32)

    # Gather embeddings for all candidates: (B, C, D)
    E = item_emb_normed[candidates]                        # (B, C, D)

    selected = np.zeros((B, C), dtype=bool)
    running_max_sim = np.zeros((B, C), dtype=np.float32)
    out_ids = np.zeros((B, K), dtype=np.int32)
    out_scores = np.zeros((B, K), dtype=np.float32)

    b_idx = np.arange(B)

    for step in range(K):
        if step == 0:
            score = rel.copy()
        else:
            score = lam * rel - (1.0 - lam) * running_max_sim
        # Block already-selected candidates from being picked again
        score = np.where(selected, -np.inf, score)

        pick = np.argmax(score, axis=1)                    # (B,)
        out_ids[:, step] = candidates[b_idx, pick]
        out_scores[:, step] = score[b_idx, pick]
        selected[b_idx, pick] = True

        # Update running max similarity: cos(picked, every candidate)
        picked_emb = E[b_idx, pick]                        # (B, D)
        # einsum: per-batch dot product of picked_emb with all C candidate embeddings
        new_sim = np.einsum("bd,bcd->bc", picked_emb, E).astype(np.float32)
        running_max_sim = np.maximum(running_max_sim, new_sim)

    return out_ids, out_scores


def mmr_full(
    test_top100: np.ndarray,           # (N, 100) int32
    test_top100_scores: np.ndarray,    # (N, 100) float32
    item_embeddings: np.ndarray,       # (V, D) float32 — unnormalized OK; normalized internally
    K: int = 20,
    lam: float = 0.5,
    batch_size: int = 1024,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply MMR to the entire test set in batches. Returns (N, K) outputs."""
    N, C = test_top100.shape
    out_ids = np.zeros((N, K), dtype=np.int32)
    out_scores = np.zeros((N, K), dtype=np.float32)

    # L2-normalize once for the full vocab (memory: 2.26M * 128 * 4B = 1.1 GB; same as input)
    item_emb_normed = l2_normalize_rows(item_embeddings)

    if progress:
        from tqdm import tqdm
        it = tqdm(range(0, N, batch_size), desc=f"MMR λ={lam}")
    else:
        it = range(0, N, batch_size)

    for start in it:
        end = min(start + batch_size, N)
        ids, scores = mmr_batch(
            candidates=test_top100[start:end],
            rel_scores=test_top100_scores[start:end],
            item_emb_normed=item_emb_normed,
            K=K,
            lam=lam,
        )
        out_ids[start:end] = ids
        out_scores[start:end] = scores

    return out_ids, out_scores
