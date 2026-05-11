"""Tests for preprocess.py — Pass 1 (padded playlists) and Pass 2 (Stage 1 candidates)."""

import numpy as np

from stage_2.training.preprocess import (
    pad_playlists,
    compute_stage1_candidates,
    build_shifted_als_factors,
)


def test_pad_playlists_output_shapes_and_padding():
    playlists = [
        [10, 20, 30],                  # length 3
        list(range(1, 61)),            # length 60 — should keep last 50
        [],                            # empty — edge case
    ]
    padded, lengths = pad_playlists(playlists, L=50)
    assert padded.shape == (3, 50) and padded.dtype == np.int32
    assert lengths.shape == (3,) and lengths.dtype == np.int16

    # row 0: left-pad 47 zeros + [10, 20, 30]
    assert padded[0, :47].tolist() == [0] * 47
    assert padded[0, 47:].tolist() == [10, 20, 30]
    assert lengths[0] == 3

    # row 1: keep last 50 of range(1, 61) = range(11, 61)
    assert padded[1].tolist() == list(range(11, 61))
    assert lengths[1] == 50

    # row 2: all zeros
    assert (padded[2] == 0).all()
    assert lengths[2] == 0


def test_build_shifted_als_factors_zero_pad_row():
    als_raw = np.random.randn(99, 128).astype(np.float32)
    shifted = build_shifted_als_factors(als_raw)
    assert shifted.shape == (100, 128)
    assert shifted.dtype == np.float32
    assert (shifted[0] == 0).all()
    np.testing.assert_array_equal(shifted[1:], als_raw)


def test_compute_stage1_candidates_returns_topk_excluding_seen():
    """For a known small playlist, candidates exclude the prefix's seen IDs."""
    # vocab: 11 (0=PAD, 1..10 real). Use 128-d but trivial setup.
    np.random.seed(0)
    als_raw = np.random.randn(10, 8).astype(np.float32)   # 10 real items
    shifted = build_shifted_als_factors(als_raw)           # (11, 8)

    # One playlist: shifted IDs [1, 2, 3, 4, 5] (prefix is [1,2,3,4])
    padded = np.array([[0] * 45 + [1, 2, 3, 4, 5]], dtype=np.int32)
    lengths = np.array([5], dtype=np.int16)

    candidates = compute_stage1_candidates(
        padded=padded,
        lengths=lengths,
        als_shifted=shifted,
        k=3,
        batch_size=1,
    )
    assert candidates.shape == (1, 3)
    assert candidates.dtype == np.int32
    # IDs in prefix [1,2,3,4] must NOT appear (they were masked to -inf)
    assert 1 not in candidates[0]
    assert 2 not in candidates[0]
    assert 3 not in candidates[0]
    assert 4 not in candidates[0]
    # PAD=0 also masked
    assert 0 not in candidates[0]
    # All returned IDs in [1, 11)
    assert ((candidates[0] >= 1) & (candidates[0] < 11)).all()
