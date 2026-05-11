"""Unit tests for Stage 3 MMR + evaluation."""

import numpy as np
import pytest

from stage_3.evaluate import (
    intra_list_diversity,
    recall_at_k,
    baseline_top_k,
)
from stage_3.mmr import (
    l2_normalize_rows,
    minmax_normalize_per_row,
    mmr_batch,
)


def test_minmax_normalize_per_row_basic():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    out = minmax_normalize_per_row(x)
    expected = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_minmax_normalize_constant_row_maps_to_zeros():
    """Constant rows have no signal — should map to all-zero (not NaN/Inf)."""
    x = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    out = minmax_normalize_per_row(x)
    assert (out == 0.0).all()
    assert np.isfinite(out).all()


def test_l2_normalize_rows_unit_norm():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize_rows(x)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0])


def test_mmr_lambda_1_returns_relevance_order():
    """With λ=1 the diversity term has zero weight; MMR picks by relevance only."""
    candidates = np.array([[5, 6, 7, 8]], dtype=np.int32)
    # Make scores strictly increasing — argmax goes to last candidate first
    rel = np.array([[0.1, 0.4, 0.7, 1.0]], dtype=np.float32)
    # Tiny embeddings — content doesn't matter when λ=1
    emb = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    emb_normed = l2_normalize_rows(emb)
    ids, _ = mmr_batch(
        candidates, rel, emb_normed, K=3, lam=1.0, normalize_rel=False,
    )
    # Expected order: 8 (rel=1.0), 7 (0.7), 6 (0.4)
    np.testing.assert_array_equal(ids[0], [8, 7, 6])


def test_mmr_lambda_0_favors_diversity_after_first_pick():
    """With λ=0 the score is pure -max_sim_to_selected; second pick goes to
    the candidate least similar to the first pick."""
    # First pick is determined by rel alone (the step-0 branch); make it unique
    # so we know which item is the first selection, then the second is purely
    # the most diverse from it.
    candidates = np.array([[100, 200, 300]], dtype=np.int32)
    rel = np.array([[0.9, 0.5, 0.5]], dtype=np.float32)        # 100 is picked first

    # Embedding setup: 100 and 200 are close, 100 and 300 are orthogonal
    emb_table = np.zeros((400, 2), dtype=np.float32)
    emb_table[100] = [1.0, 0.0]
    emb_table[200] = [0.99, 0.14]   # close to 100
    emb_table[300] = [0.0, 1.0]     # orthogonal to 100
    emb_normed = l2_normalize_rows(emb_table)

    ids, _ = mmr_batch(
        candidates, rel, emb_normed, K=2, lam=0.0, normalize_rel=False,
    )
    # First pick: 100 (highest rel). Second pick at λ=0 maximizes -max_sim,
    # i.e. minimizes max_sim_to_selected. 300 is orthogonal to 100, so should win.
    assert ids[0, 0] == 100
    assert ids[0, 1] == 300


def test_mmr_no_duplicates_in_output():
    """MMR must never pick the same candidate twice."""
    np.random.seed(0)
    B, C, V, D = 8, 30, 200, 16
    candidates = np.random.randint(1, V, size=(B, C)).astype(np.int32)
    rel = np.random.rand(B, C).astype(np.float32)
    emb = np.random.randn(V, D).astype(np.float32)
    emb_normed = l2_normalize_rows(emb)
    ids, _ = mmr_batch(candidates, rel, emb_normed, K=10, lam=0.5)
    for b in range(B):
        assert len(set(ids[b])) == 10, f"playlist {b} has duplicates: {ids[b]}"


def test_recall_at_k():
    final_ids = np.array(
        [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [7, 8, 9, 0, 0],
        ],
        dtype=np.int32,
    )
    targets = np.array([3, 100, 8], dtype=np.int32)
    # Hits: row0 (3 in [1..5]) ✓, row1 (100 not present) ✗, row2 (8 in [7..0]) ✓
    assert recall_at_k(final_ids, targets) == pytest.approx(2.0 / 3.0)


def test_intra_list_diversity_orthogonal_set_is_one():
    """For K orthonormal items, mean pairwise cos sim is 0 → ILD = 1.0."""
    V, D = 10, 4
    emb = np.zeros((V, D), dtype=np.float32)
    emb[1] = [1, 0, 0, 0]
    emb[2] = [0, 1, 0, 0]
    emb[3] = [0, 0, 1, 0]
    emb[4] = [0, 0, 0, 1]
    emb_normed = l2_normalize_rows(emb)
    final_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    ild = intra_list_diversity(final_ids, emb_normed)
    assert ild == pytest.approx(1.0, abs=1e-5)


def test_intra_list_diversity_identical_set_is_zero():
    """For K identical items, mean pairwise cos sim is 1 → ILD = 0."""
    emb = np.zeros((5, 4), dtype=np.float32)
    emb[1] = [1, 0, 0, 0]
    emb_normed = l2_normalize_rows(emb)
    final_ids = np.array([[1, 1, 1, 1]], dtype=np.int32)
    ild = intra_list_diversity(final_ids, emb_normed)
    assert ild == pytest.approx(0.0, abs=1e-5)


def test_baseline_top_k_takes_first_k():
    test_top100 = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    test_scores = np.array([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    out = baseline_top_k(test_top100, test_scores, K=3)
    np.testing.assert_array_equal(out, [[1, 2, 3]])
    assert out.dtype == np.int32
