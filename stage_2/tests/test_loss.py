"""Tests for negative pool construction and CE loss."""

import torch

from stage_2.training.loss import build_logits, masked_ce_loss


def _make_inputs(B=2, L=50, D=128, vocab=100, K_hard=4, K_rand=4):
    torch.manual_seed(0)
    queries = torch.randn(B, L, D)
    positives = torch.randint(1, vocab, (B, L))
    hard_negs = torch.randint(1, vocab, (B, K_hard))
    random_negs = torch.randint(1, vocab, (K_rand,))
    # Fake embedding table
    item_emb = torch.nn.Embedding(vocab, D)
    return queries, positives, hard_negs, random_negs, item_emb


def test_logits_shape():
    queries, positives, hard_negs, random_negs, item_emb = _make_inputs(
        B=2, L=50, D=128, K_hard=4, K_rand=4
    )
    logits = build_logits(queries, positives, hard_negs, random_negs, item_emb)
    assert logits.shape == (2, 50, 1 + 4 + 4)


def test_positive_is_at_index_0():
    """When positive has a unique embedding pointing exactly along the query, it should score highest."""
    B, L, D, vocab = 1, 1, 4, 10
    item_emb = torch.nn.Embedding(vocab, D)
    item_emb.weight.data.zero_()
    # Make item 5's embedding equal to the query direction
    item_emb.weight.data[5] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    queries = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])    # (1, 1, 4)
    positives = torch.tensor([[5]])                      # (1, 1)
    hard_negs = torch.tensor([[1, 2, 3]])                # (1, 3) — all zero embeddings
    random_negs = torch.tensor([6, 7])                   # (2,) — all zero embeddings
    logits = build_logits(queries, positives, hard_negs, random_negs, item_emb)
    # index 0 = positive, should have logit = 1.0; all others = 0.0
    assert logits[0, 0, 0].item() == 1.0
    assert (logits[0, 0, 1:] == 0.0).all()


def test_masked_ce_zero_when_perfect():
    """If logits put all mass on index 0, masked CE should be near 0 at unmasked positions."""
    B, L = 2, 5
    n_total = 5
    logits = torch.zeros(B, L, n_total)
    logits[..., 0] = 100.0          # huge positive logit at index 0
    mask = torch.ones(B, L, dtype=torch.bool)
    loss = masked_ce_loss(logits, mask)
    assert loss.item() < 1e-3


def test_masked_ce_ignores_masked_positions():
    """Loss should not depend on logits at masked positions."""
    B, L = 1, 3
    n_total = 4
    logits = torch.zeros(B, L, n_total)
    logits[0, 0, 0] = 100.0   # good
    logits[0, 1, 0] = 100.0   # good
    logits[0, 2, 3] = 100.0   # bad — but should be masked out
    mask = torch.tensor([[True, True, False]])
    loss = masked_ce_loss(logits, mask)
    assert loss.item() < 1e-3


def test_one_training_step_decreases_loss():
    """One optimizer step on a synthetic batch should reduce the loss."""
    import torch
    from stage_2.models.sasrec import SASRec
    from stage_2.training.loss import (
        build_logits, masked_ce_loss, sample_hard_negatives, sample_random_negatives,
    )

    torch.manual_seed(0)
    vocab = 50
    model = SASRec(vocab_size=vocab, d_model=32, n_layers=1, n_heads=2,
                   ffn_dim=64, max_seq_len=10, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Synthetic batch: 4 playlists, length 10, IDs in [1, vocab)
    seq = torch.randint(1, vocab, (4, 10))
    positives = torch.roll(seq, shifts=-1, dims=1)
    pos_mask = torch.ones(4, 10, dtype=torch.bool)
    pos_mask[:, -1] = False
    candidates = torch.randint(1, vocab, (4, 32))

    def step():
        queries = model(seq)
        hn = sample_hard_negatives(candidates, k=8)
        rn = sample_random_negatives(k=8, vocab_size=vocab, device=seq.device)
        logits = build_logits(queries, positives, hn, rn, model.item_emb)
        return masked_ce_loss(logits, pos_mask)

    loss0 = step().item()
    opt.zero_grad()
    step().backward()
    opt.step()
    loss1 = step().item()
    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"
