"""Dataset and padding helpers for Stage 2 SASRec training.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §4 (padding)
and §5.1 (training step shapes).
"""

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


def pad_left(p: List[int], L: int = 50) -> List[int]:
    """Truncate to last L (keep recent) and left-pad with PAD=0."""
    p = p[-L:]
    return [0] * (L - len(p)) + p


def shift_ids_plus_one(playlists: List[List[int]]) -> List[List[int]]:
    """Shift every track ID by +1 to reserve PAD=0."""
    return [[t + 1 for t in p] for p in playlists]


class SASRecDataset(Dataset):
    """Yields (seq, positives, positive_mask, candidates) per playlist.

    seq:           (L,) int64 — left-padded playlist (full, not playlist[:-1])
    positives:     (L,) int64 — seq shifted by 1; position i predicts position i+1
    positive_mask: (L,) bool — True where this position has a real next song
    candidates:    (1000,) int64 — this playlist's Stage 1 candidates
    """

    def __init__(
        self,
        playlists_padded: np.ndarray,   # (N, L) int32
        playlists_lengths: np.ndarray,  # (N,) int16
        candidates: np.ndarray,         # (N, 1000) int32
        max_seq: int = 50,
    ):
        assert playlists_padded.shape[1] == max_seq
        assert playlists_padded.shape[0] == playlists_lengths.shape[0]
        assert playlists_padded.shape[0] == candidates.shape[0]
        self.seqs = playlists_padded
        self.lengths = playlists_lengths
        self.candidates = candidates
        self.max_seq = max_seq

    def __len__(self) -> int:
        return self.seqs.shape[0]

    def __getitem__(self, idx: int):
        seq_np = self.seqs[idx]                            # (L,) int32
        length = int(self.lengths[idx])                    # actual non-pad length

        seq = torch.from_numpy(seq_np.astype(np.int64))    # (L,)

        # positives[i] = seq[i+1] only for real (non-PAD) positions that have a next.
        # For left-padded sequences of length=length, real positions are L-length..L-1.
        # Valid prediction positions are L-length..L-2 (last real position has no next).
        L = self.max_seq
        positives = torch.zeros_like(seq)
        if length >= 2:
            positives[L - length : L - 1] = seq[L - length + 1 : L]

        # positive_mask: True where this position has a real next song
        pos_mask = torch.zeros(L, dtype=torch.bool)
        if length >= 2:
            pos_mask[L - length : L - 1] = True

        cand = torch.from_numpy(self.candidates[idx].astype(np.int64))

        return seq, positives, pos_mask, cand
