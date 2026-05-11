"""Preprocessing passes for Stage 2.

Pass 1: pad playlists to (N, 50) int32 and record lengths.
Pass 2: precompute Stage 1's top-1000 candidates for every playlist.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §4.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from stage_2.data.dataset import pad_left, shift_ids_plus_one


def pad_playlists(playlists: List[List[int]], L: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Left-pad/truncate playlists to length L. Returns (padded, lengths).

    Inputs may contain IDs from any range; this function does not shift IDs.
    Apply `shift_ids_plus_one` first if working in the +1-shifted space.
    """
    padded = np.zeros((len(playlists), L), dtype=np.int32)
    lengths = np.zeros(len(playlists), dtype=np.int16)
    for i, p in enumerate(playlists):
        padded[i] = pad_left(p, L=L)
        lengths[i] = min(len(p), L)
    return padded, lengths


def build_shifted_als_factors(als_raw: np.ndarray) -> np.ndarray:
    """Prepend a zero row at index 0 (PAD slot) to the ALS factors.

    Input:  (vocab_size - 1, d) float32
    Output: (vocab_size, d)     float32
    """
    assert als_raw.dtype == np.float32
    pad_row = np.zeros((1, als_raw.shape[1]), dtype=np.float32)
    return np.vstack([pad_row, als_raw])


def compute_stage1_candidates(
    padded: np.ndarray,           # (N, L) int32 — shifted IDs
    lengths: np.ndarray,          # (N,) int16
    als_shifted: np.ndarray,      # (vocab, d) float32
    k: int = 1000,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """For each playlist, compute top-k Stage 1 candidates over prefix[:-1].

    Returns (N, k) int32 — IDs in the shifted space, all in [1, vocab).

    GPU-batched matmul. Masking is applied per-row: prefix tokens + PAD=0 get -inf.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N, L = padded.shape
    vocab, d = als_shifted.shape

    als_t = torch.from_numpy(als_shifted).to(device)           # (vocab, d)
    candidates_out = np.zeros((N, k), dtype=np.int32)

    for start in tqdm(range(0, N, batch_size),
                      desc="caching candidates",
                      mininterval=2.0, miniters=10):
        end = min(start + batch_size, N)
        B = end - start

        # Compute mean-pool of prefix[:-1] for each playlist in the batch
        user_embs = torch.zeros(B, d, device=device)
        for j in range(B):
            length = int(lengths[start + j])
            if length < 2:
                continue   # prefix[:-1] empty; leaves zero user_emb (candidates will be ~random)
            seq_j = padded[start + j]
            real_seq = seq_j[seq_j != 0]               # drop padding
            prefix = real_seq[:-1]                      # all but last real
            user_embs[j] = als_t[prefix].mean(dim=0)

        scores = user_embs @ als_t.T                    # (B, vocab)

        # Mask PAD=0 and any token in prefix
        scores[:, 0] = -float("inf")
        for j in range(B):
            length = int(lengths[start + j])
            if length < 2:
                continue
            seq_j = padded[start + j]
            real_seq = seq_j[seq_j != 0]
            prefix = real_seq[:-1]
            scores[j, prefix] = -float("inf")

        top_idx = torch.topk(scores, k=k, dim=1).indices    # (B, k)
        candidates_out[start:end] = top_idx.cpu().numpy().astype(np.int32)

    return candidates_out


def run_preprocessing(
    cfg: dict,
    raw_playlists_path: str,
    als_factors_path: str,
    output_dir: str,
) -> None:
    """Top-level driver: Pass 1 + Pass 2. Writes derived files into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading playlists from {raw_playlists_path}...")
    playlists_raw = np.load(raw_playlists_path, allow_pickle=True).tolist()
    print(f"  loaded {len(playlists_raw):,} playlists")

    print("Shifting IDs +1...")
    playlists = shift_ids_plus_one(playlists_raw)

    print("Pass 1: padding playlists...")
    padded, lengths = pad_playlists(playlists, L=cfg["model"]["max_seq_len"])
    np.save(output_dir / "playlists_padded.npy", padded)
    np.save(output_dir / "playlists_lengths.npy", lengths)
    print(f"  wrote playlists_padded.npy {padded.shape}  playlists_lengths.npy {lengths.shape}")

    print(f"Loading ALS factors from {als_factors_path}...")
    als_raw = np.load(als_factors_path).astype(np.float32)
    als_shifted = build_shifted_als_factors(als_raw)
    print(f"  shifted ALS factors {als_shifted.shape}")

    print("Pass 2: computing Stage 1 candidates...")
    candidates = compute_stage1_candidates(
        padded=padded,
        lengths=lengths,
        als_shifted=als_shifted,
        k=cfg["eval"]["candidates_k"],
        batch_size=512,
    )
    np.save(output_dir / "candidates.npy", candidates)
    print(f"  wrote candidates.npy {candidates.shape}")
