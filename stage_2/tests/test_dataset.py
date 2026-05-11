"""Tests for padding logic and SASRecDataset."""

import numpy as np
import torch

from stage_2.data.dataset import (
    pad_left,
    shift_ids_plus_one,
    SASRecDataset,
)


def test_pad_left_long_playlist_truncates_to_last_L():
    p = list(range(1, 101))   # length 100
    out = pad_left(p, L=50)
    assert len(out) == 50
    assert out == list(range(51, 101))


def test_pad_left_short_playlist_left_pads_with_zero():
    p = [1, 2, 3]
    out = pad_left(p, L=50)
    assert len(out) == 50
    assert out[:47] == [0] * 47
    assert out[47:] == [1, 2, 3]


def test_pad_left_exact_length():
    p = list(range(1, 51))   # length 50
    out = pad_left(p, L=50)
    assert out == p


def test_shift_ids_plus_one():
    playlists = [[0, 1, 2], [10, 20]]
    shifted = shift_ids_plus_one(playlists)
    assert shifted == [[1, 2, 3], [11, 21]]


def test_dataset_returns_correct_shapes():
    """Sequences, positives, masks, candidates all have expected shapes."""
    padded = np.array([[0] * 45 + [1, 2, 3, 4, 5]], dtype=np.int32)   # one playlist, len 5
    lengths = np.array([5], dtype=np.int16)
    candidates = np.zeros((1, 1000), dtype=np.int32)
    ds = SASRecDataset(padded, lengths, candidates, max_seq=50)
    seq, positives, pos_mask, cand = ds[0]
    assert seq.shape == (50,) and seq.dtype == torch.long
    assert positives.shape == (50,) and positives.dtype == torch.long
    assert pos_mask.shape == (50,) and pos_mask.dtype == torch.bool
    assert cand.shape == (1000,) and cand.dtype == torch.long


def test_dataset_positives_are_seq_shifted_by_one():
    padded = np.array([[0] * 45 + [11, 22, 33, 44, 55]], dtype=np.int32)
    lengths = np.array([5], dtype=np.int16)
    candidates = np.zeros((1, 1000), dtype=np.int32)
    ds = SASRecDataset(padded, lengths, candidates, max_seq=50)
    seq, positives, pos_mask, _ = ds[0]
    # positions 45..48 predict 22, 33, 44, 55; position 49 (last real) has no next
    assert positives[45].item() == 22
    assert positives[46].item() == 33
    assert positives[47].item() == 44
    assert positives[48].item() == 55
    # Mask: positions 45..48 are valid prediction positions; 49 (last) and 0..44 (PAD) are not
    expected_mask = torch.zeros(50, dtype=torch.bool)
    expected_mask[45:49] = True
    assert torch.equal(pos_mask, expected_mask)


def test_dataset_pad_positions_have_zero_positives_and_false_mask():
    """Padded positions should have positive=0 and mask=False."""
    padded = np.array([[0] * 47 + [7, 8, 9]], dtype=np.int32)
    lengths = np.array([3], dtype=np.int16)
    candidates = np.zeros((1, 1000), dtype=np.int32)
    ds = SASRecDataset(padded, lengths, candidates, max_seq=50)
    _, positives, pos_mask, _ = ds[0]
    assert (positives[:47] == 0).all()
    assert (pos_mask[:47] == False).all()
