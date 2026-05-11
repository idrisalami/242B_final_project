"""Tests for the SASRec model."""

import numpy as np
import torch

from stage_2.models.sasrec import SASRec


def _make_model(vocab=100, d=128, layers=2, heads=2, max_seq=50, dropout=0.0):
    return SASRec(
        vocab_size=vocab,
        d_model=d,
        n_layers=layers,
        n_heads=heads,
        ffn_dim=d * 2,
        max_seq_len=max_seq,
        dropout=dropout,
    )


def test_forward_shape():
    model = _make_model().eval()
    seq = torch.randint(0, 100, (4, 50))
    out = model(seq)
    assert out.shape == (4, 50, 128), f"got {tuple(out.shape)}"


def test_forward_handles_pad_mask():
    """Padded positions should not crash; output shape is preserved."""
    model = _make_model().eval()
    seq = torch.zeros(2, 50, dtype=torch.long)
    seq[:, -5:] = torch.randint(1, 100, (2, 5))   # only last 5 positions are real
    out = model(seq)
    assert out.shape == (2, 50, 128)
    assert torch.isfinite(out).all()


def test_causal_mask_blocks_future():
    """Position i must not see token at position j > i."""
    torch.manual_seed(0)
    model = _make_model().eval()
    seq1 = torch.randint(1, 100, (1, 50))
    seq2 = seq1.clone()
    seq2[0, 25:] = torch.randint(1, 100, (25,))   # mutate positions 25+
    with torch.no_grad():
        out1 = model(seq1)
        out2 = model(seq2)
    # Positions 0..24 should be unaffected by changes at 25+
    assert torch.allclose(out1[0, :25], out2[0, :25], atol=1e-5), \
        "Causal mask is leaking future positions into earlier ones"


def test_als_init_zero_pad_and_match_rows():
    """After loading ALS factors, row 0 should be zero, rows [1:] should equal ALS rows."""
    torch.manual_seed(0)
    als = np.random.randn(99, 128).astype(np.float32)
    model = _make_model(vocab=100)
    model.load_als(torch.from_numpy(als))
    assert torch.allclose(model.item_emb.weight[0], torch.zeros(128))
    for i in range(99):
        assert torch.allclose(model.item_emb.weight[i + 1], torch.from_numpy(als[i]))


def test_param_groups_separate_embedding_from_transformer():
    """Optimizer param groups should separate the embedding table from everything else."""
    model = _make_model()
    embed_params, other_params = model.param_groups()
    embed_ids = {id(p) for p in embed_params}
    other_ids = {id(p) for p in other_params}
    assert embed_ids.isdisjoint(other_ids), "Param groups overlap"
    # All trainable params accounted for
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert embed_ids | other_ids == all_ids, "Param groups don't cover all params"
    # Embedding group contains exactly the item_emb table
    assert id(model.item_emb.weight) in embed_ids
