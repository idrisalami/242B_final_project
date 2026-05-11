"""Evaluation for Stage 3 — recall@K and intra-list diversity.

See docs/superpowers/specs/2026-05-11-stage3-mmr-design.md §6.
"""

from typing import Dict

import numpy as np


def recall_at_k(final_ids: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of playlists where the target ID appears in final_ids.

    final_ids: (N, K) int32 — final per-playlist IDs (any K)
    targets:   (N,)   int32 — held-out true next song per playlist
    """
    return float((final_ids == targets[:, None]).any(axis=1).mean())


def intra_list_diversity(
    final_ids: np.ndarray,             # (N, K) int32
    item_emb_normed: np.ndarray,       # (V, D) float32 — L2-normalized
    batch_size: int = 4096,
) -> float:
    """Mean over playlists of mean pairwise cosine *distance* among the K items.

    cosine distance = 1 - cosine_similarity.
    Higher = more diverse. Bounded in [0, 2].

    Computed by gathering K embeddings per playlist (shape (B, K, D)),
    forming a K×K cosine-sim matrix per playlist, taking the off-diagonal
    mean, then averaging across all playlists.
    """
    N, K = final_ids.shape
    total = 0.0
    counted = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = final_ids[start:end]                          # (B, K)
        E = item_emb_normed[ids]                            # (B, K, D)
        # Pairwise cos sim: (B, K, K)
        sim = np.einsum("bkd,bjd->bkj", E, E)
        # Off-diagonal mean: subtract diagonal (=1 per item), divide by K*(K-1)
        sim_off = sim.sum(axis=(1, 2)) - sim.diagonal(axis1=1, axis2=2).sum(axis=1)
        denom = K * (K - 1)
        mean_sim = sim_off / denom                          # (B,) — mean pairwise cos sim
        total += float((1.0 - mean_sim).sum())              # cos distance = 1 - cos sim
        counted += end - start
    return total / max(1, counted)


def evaluate_run(
    final_ids: np.ndarray,             # (N, K) int32
    targets: np.ndarray,               # (N,)   int32
    item_emb_normed: np.ndarray,       # (V, D) float32 — L2-normalized
) -> Dict[str, float]:
    """Compute recall@K and ILD for a single MMR configuration."""
    return {
        "recall_at_K": recall_at_k(final_ids, targets),
        "intra_list_diversity": intra_list_diversity(final_ids, item_emb_normed),
        "n_evaluated": int(final_ids.shape[0]),
        "K": int(final_ids.shape[1]),
    }


def baseline_top_k(
    test_top100: np.ndarray,           # (N, 100) int32
    test_scores: np.ndarray,           # (N, 100) float32 — assumed sorted best-first
    K: int = 20,
) -> np.ndarray:
    """Stage 2 raw top-K baseline.

    Stage 2's precompute_test_top100 already writes IDs sorted by score
    (it uses torch.topk which returns sorted), so we can just slice.
    """
    return test_top100[:, :K].astype(np.int32)
