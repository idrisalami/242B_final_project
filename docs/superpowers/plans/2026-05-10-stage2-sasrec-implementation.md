# Stage 2 SASRec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 2 sequential re-ranker (SASRec) that takes Stage 1's top-1000 candidates for a playlist and re-ranks them to a top-100, trained with causal LM on the Spotify MPD and hard-negative mining against Stage 1's candidates, deployed on Modal's free GPU tier.

**Architecture:** Pure-PyTorch SASRec-original (2 layers, 2 heads, d=128, max_seq=50), embedding table initialized from Stage 1's ALS factors and fine-tuned with a 10× lower LR than the transformer. Training is fully causal LM (every position predicts the next song); eval is single-target (held-out last song). Modal orchestrates training and evaluation; data lives on a persistent Modal Volume; code is mounted fresh per call.

**Tech Stack:** Python 3.11, PyTorch 2.4, NumPy, tqdm, PyYAML, Modal, pytest. No external recommender libraries (the `implicit` dep from Stage 1 is not needed here — ALS factors are read as a static `.npy`).

**Spec reference:** `docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md`. Every task below maps to a numbered section in the spec; verify against the spec on any ambiguity.

---

## File Structure

| File | Purpose | Created in |
|---|---|---|
| `stage_2/__init__.py` | Marker | Task 1 |
| `stage_2/config.yaml` | Single source of truth for hyperparameters | Task 2 |
| `stage_2/models/__init__.py`, `stage_2/models/sasrec.py` | SASRec model class + ALS init | Task 3 |
| `stage_2/data/__init__.py`, `stage_2/data/dataset.py` | PyTorch Dataset, padding helpers | Task 4 |
| `stage_2/training/__init__.py`, `stage_2/training/loss.py` | Negative pool sampler + CE loss | Task 5 |
| `stage_2/training/preprocess.py` | Pass 1 (pad playlists) + Pass 2 (Stage 1 candidate cache) | Task 6 |
| `stage_2/training/train.py` | Causal LM training loop with val cadence + early stopping + checkpointing | Task 7 |
| `stage_2/training/evaluate.py` | Pipeline-mode + unconstrained eval | Task 8 |
| `stage_2/inference/__init__.py`, `stage_2/inference/predict.py` | Single-playlist inference + `precompute_test_top100` | Task 9 |
| `stage_2/modal_app.py` | Modal Image + Volume + Functions + local entrypoint | Task 10 |
| `stage_2/tests/__init__.py`, `stage_2/tests/test_*.py` | Unit tests for Tier 1 validation | Tasks 3, 4, 5, 6 (test added with each component) |
| `pyproject.toml` | Add `modal`, `pytest` deps | Task 1 |
| `.gitignore` | Add `stage_2/checkpoints/`, `stage_2/data/`, `.pytest_cache/` | Task 1 |

---

## Task 1: Project skeleton, dependencies, gitignore

**Files:**
- Create: `stage_2/__init__.py`, `stage_2/models/__init__.py`, `stage_2/data/__init__.py`, `stage_2/training/__init__.py`, `stage_2/inference/__init__.py`, `stage_2/tests/__init__.py`
- Modify: `pyproject.toml`, `.gitignore`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p stage_2/{models,data,training,inference,tests,checkpoints}
touch stage_2/__init__.py stage_2/models/__init__.py stage_2/data/__init__.py stage_2/training/__init__.py stage_2/inference/__init__.py stage_2/tests/__init__.py
```

- [ ] **Step 2: Add `modal` and `pytest` to `pyproject.toml`**

Open `pyproject.toml`. Under `[project] dependencies`, add `"modal>=0.64"` and `"pytest>=8.0"`. The final list should be:

```toml
dependencies = [
  "implicit>=0.7.2",
  "modal>=0.64",
  "numpy>=1.24",
  "pandas>=2.0",
  "peft>=0.19.1",
  "pyyaml>=6.0",
  "pytest>=8.0",
  "torch>=2.2",
  "tqdm>=4.67.3",
]
```

- [ ] **Step 3: Install deps**

Run: `uv sync`
Expected: no errors. If `uv` is not available, run `pip install modal pytest`.

- [ ] **Step 4: Update `.gitignore`**

Append to `.gitignore`:

```
# Stage 2
stage_2/checkpoints/
stage_2/data/
.pytest_cache/
```

- [ ] **Step 5: Verify pytest discovery works**

Run: `pytest stage_2/tests/ --collect-only`
Expected: `no tests ran in X.XXs` (no tests yet, but discovery succeeds).

- [ ] **Step 6: Commit**

```bash
git add stage_2/ pyproject.toml .gitignore uv.lock
git commit -m "Add stage_2 skeleton and deps for SASRec implementation"
```

---

## Task 2: Configuration file

**Files:**
- Create: `stage_2/config.yaml`

- [ ] **Step 1: Write `stage_2/config.yaml`**

Create `stage_2/config.yaml` with the exact contents:

```yaml
# Stage 2 SASRec configuration
# Single source of truth — copied verbatim into each run's output dir for reproducibility.

model:
  d_model: 128
  n_layers: 2
  n_heads: 2
  ffn_dim: 256
  max_seq_len: 50
  dropout: 0.2
  vocab_size: 2262293          # 2262292 ALS tracks + 1 PAD slot (ID 0)

data:
  min_train_length: 10         # filter applied to TRAIN ONLY
  train_size: 700000
  val_size: 150000
  test_size: 150000
  val_eval_subset: 10000       # subset of val used for periodic eval during training

training:
  batch_size: 256
  epochs: 15
  warmup_steps: 1000
  peak_lr_transformer: 1.0e-3
  peak_lr_embedding: 1.0e-4
  weight_decay: 0.01
  grad_clip: 1.0
  precision: bf16
  seed: 42
  # Negative pool composition (see spec §5.1, §5.2)
  num_hard_negs: 128
  num_random_negs: 128
  # Early stopping
  val_every_n_epochs: 2
  early_stopping_patience_evals: 3   # = 6 epochs of patience
  # Checkpoint cleanup
  keep_best_and_last_only: true

eval:
  candidates_k: 1000
  ranked_top_k: 100
  metrics_recall_k: [10, 50, 100, 1000]
  ndcg_k: 10

modal:
  gpu: a10g
  volume_name: stage2-data
  app_name: stage2-sasrec
  function_timeout_sec: 14400      # 4 hours

paths:
  vol_root: /vol
  als_factors: /vol/inputs/als_item_factors.npy
  uri_to_id: /vol/inputs/uri_to_id.json
  playlists_raw: /vol/inputs/playlists.npy
  playlists_padded: /vol/derived/playlists_padded.npy
  playlists_lengths: /vol/derived/playlists_lengths.npy
  candidates: /vol/derived/candidates.npy
  runs_dir: /vol/runs
```

- [ ] **Step 2: Commit**

```bash
git add stage_2/config.yaml
git commit -m "Add stage_2 config.yaml with all hyperparameters"
```

---

## Task 3: SASRec model

**Files:**
- Create: `stage_2/models/sasrec.py`
- Create: `stage_2/tests/test_sasrec.py`

- [ ] **Step 1: Write failing tests**

Create `stage_2/tests/test_sasrec.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest stage_2/tests/test_sasrec.py -v`
Expected: `ImportError: cannot import name 'SASRec' from 'stage_2.models.sasrec'` (module doesn't exist yet).

- [ ] **Step 3: Implement `stage_2/models/sasrec.py`**

Create `stage_2/models/sasrec.py`:

```python
"""SASRec-original (Kang & McAuley, 2018) — small causal-attention transformer.

Initialized from Stage 1's ALS item embeddings; fine-tuned end-to-end with
causal LM training. See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §3.
"""

import torch
import torch.nn as nn


class _Block(nn.Module):
    """One pre-LN Transformer block: causal MHA + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=False,            # we pass attn_mask explicitly
        )
        x = x + self.drop(a)
        x = x + self.ffn(self.ln2(x))
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        ffn_dim: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Causal mask cached at construction time
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].zero_()
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def load_als(self, als_factors: torch.Tensor):
        """Load ALS factors into rows [1:]. Row 0 (PAD) stays zero.

        als_factors: shape (vocab_size - 1, d_model), float32.
        """
        v_minus_1, d = als_factors.shape
        assert v_minus_1 == self.item_emb.weight.shape[0] - 1, \
            f"ALS rows ({v_minus_1}) must equal vocab_size - 1 ({self.item_emb.weight.shape[0] - 1})"
        assert d == self.item_emb.weight.shape[1], \
            f"ALS dim ({d}) must equal d_model ({self.item_emb.weight.shape[1]})"
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.item_emb.weight[1:].copy_(als_factors)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) int64 padded with pad_id=0. Returns (B, L, d_model)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.input_drop(self.input_ln(x))

        pad_mask = (seq == self.pad_id)        # (B, L) — True where PAD
        attn_mask = self.causal_mask[:L, :L]   # (L, L) — True where blocked

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        return self.final_ln(x)

    def param_groups(self):
        """Split trainable params into (embedding, others) for discriminative fine-tuning."""
        embed_params = [self.item_emb.weight]
        other_params = [
            p for n, p in self.named_parameters()
            if n != "item_emb.weight" and p.requires_grad
        ]
        return embed_params, other_params
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest stage_2/tests/test_sasrec.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stage_2/models/sasrec.py stage_2/tests/test_sasrec.py
git commit -m "Add SASRec model with ALS init and discriminative param groups"
```

---

## Task 4: Dataset + padding helpers

**Files:**
- Create: `stage_2/data/dataset.py`
- Create: `stage_2/tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

Create `stage_2/tests/test_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest stage_2/tests/test_dataset.py -v`
Expected: `ImportError: cannot import name 'pad_left' from 'stage_2.data.dataset'`.

- [ ] **Step 3: Implement `stage_2/data/dataset.py`**

```python
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

        # positives[i] = seq[i+1]; last position is unused (set to 0)
        positives = torch.zeros_like(seq)
        positives[:-1] = seq[1:]

        # positive_mask: True where both seq[i] and seq[i+1] are real (non-PAD)
        # For left-padded sequences of length=length, real positions are L-length..L-1.
        # Valid prediction positions are L-length..L-2 (last real position has no next).
        L = self.max_seq
        pos_mask = torch.zeros(L, dtype=torch.bool)
        if length >= 2:
            pos_mask[L - length : L - 1] = True

        cand = torch.from_numpy(self.candidates[idx].astype(np.int64))

        return seq, positives, pos_mask, cand
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest stage_2/tests/test_dataset.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stage_2/data/dataset.py stage_2/tests/test_dataset.py
git commit -m "Add SASRecDataset + padding helpers"
```

---

## Task 5: Loss function and negative pool

**Files:**
- Create: `stage_2/training/loss.py`
- Create: `stage_2/tests/test_loss.py`

- [ ] **Step 1: Write failing tests**

Create `stage_2/tests/test_loss.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest stage_2/tests/test_loss.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `stage_2/training/loss.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest stage_2/tests/test_loss.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stage_2/training/loss.py stage_2/tests/test_loss.py
git commit -m "Add negative pool sampling + masked softmax CE loss"
```

---

## Task 6: Preprocessing (Pass 1 padded playlists + Pass 2 candidate cache)

**Files:**
- Create: `stage_2/training/preprocess.py`
- Create: `stage_2/tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

Create `stage_2/tests/test_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest stage_2/tests/test_preprocess.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `stage_2/training/preprocess.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest stage_2/tests/test_preprocess.py -v`
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stage_2/training/preprocess.py stage_2/tests/test_preprocess.py
git commit -m "Add preprocessing: Pass 1 padding + Pass 2 Stage 1 candidate cache"
```

---

## Task 7: Training loop

**Files:**
- Create: `stage_2/training/train.py`
- Modify: `stage_2/tests/test_loss.py` (add an integration test for one training step)

- [ ] **Step 1: Add an end-to-end smoke test**

Append to `stage_2/tests/test_loss.py`:

```python
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
```

- [ ] **Step 2: Run the new test to verify it fails meaningfully** (it shouldn't fail at import time, only at the assertion — but the training loop hasn't been written yet so we'll catch any imports here)

Run: `pytest stage_2/tests/test_loss.py::test_one_training_step_decreases_loss -v`
Expected: PASS (this test doesn't need `train.py`; it directly exercises the building blocks).

- [ ] **Step 3: Implement `stage_2/training/train.py`**

```python
"""Causal-LM training loop for Stage 2 SASRec.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §5.
"""

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage_2.data.dataset import SASRecDataset
from stage_2.models.sasrec import SASRec
from stage_2.training.evaluate import evaluate_pipeline_mode
from stage_2.training.loss import (
    build_logits,
    masked_ce_loss,
    sample_hard_negatives,
    sample_random_negatives,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms — slows things slightly, removes nondeterminism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)


def save_checkpoint(run_dir: Path, tag: str, model: SASRec):
    """Save a checkpoint as a directory with model.pt + item_embeddings.npy."""
    ckpt_dir = run_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Item embedding table as .npy (large, separate)
    embed = model.item_emb.weight.detach().cpu().float().numpy()
    np.save(ckpt_dir / "item_embeddings.npy", embed)

    # Everything else as state_dict
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k != "item_emb.weight"}
    torch.save(state, ckpt_dir / "model.pt")


def load_checkpoint(run_dir: Path, tag: str, model: SASRec):
    """Load a checkpoint produced by save_checkpoint."""
    ckpt_dir = run_dir / tag
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    embed = np.load(ckpt_dir / "item_embeddings.npy")
    with torch.no_grad():
        model.item_emb.weight.copy_(torch.from_numpy(embed))
    model.load_state_dict(state, strict=False)


def train_main(cfg: Dict, run_id: str, smoke: bool = False) -> Dict:
    """Full training loop. Returns a dict of summary metrics.

    If smoke=True: 1 epoch, no early-stop, exits after final val eval.
    """
    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_dir = Path(cfg["paths"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading derived data...")
    padded = np.load(cfg["paths"]["playlists_padded"])     # (N, L) int32
    lengths = np.load(cfg["paths"]["playlists_lengths"])   # (N,)   int16
    candidates = np.load(cfg["paths"]["candidates"])       # (N, 1000) int32

    n_train = cfg["data"]["train_size"]
    n_val = cfg["data"]["val_size"]
    train_idx = np.arange(n_train)
    # Filter train only: length >= min_train_length
    min_train = cfg["data"]["min_train_length"]
    train_idx = train_idx[lengths[train_idx] >= min_train]
    print(f"Train (filtered len>={min_train}): {len(train_idx):,}  Val: {n_val:,}")

    train_ds = SASRecDataset(
        playlists_padded=padded[train_idx],
        playlists_lengths=lengths[train_idx],
        candidates=candidates[train_idx],
        max_seq=cfg["model"]["max_seq_len"],
    )

    # Val subset
    val_start = n_train
    val_subset_size = cfg["data"]["val_eval_subset"]
    val_idx = np.arange(val_start, val_start + val_subset_size)
    val_padded = padded[val_idx]
    val_lengths = lengths[val_idx]
    val_candidates = candidates[val_idx]

    # ── Load ALS factors into model ──────────────────────────────────────────
    print("Loading ALS factors into model...")
    als_raw = np.load(cfg["paths"]["als_factors"]).astype(np.float32)
    model = SASRec(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_als(torch.from_numpy(als_raw).to(device))

    # ── Optimizer with two parameter groups ──────────────────────────────────
    embed_params, other_params = model.param_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": cfg["training"]["peak_lr_embedding"],
             "weight_decay": 0.0},
            {"params": other_params, "lr": cfg["training"]["peak_lr_transformer"],
             "weight_decay": cfg["training"]["weight_decay"]},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    epochs = 1 if smoke else cfg["training"]["epochs"]
    steps_per_epoch = math.ceil(len(train_ds) / cfg["training"]["batch_size"])
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(cfg["training"]["warmup_steps"], total_steps),
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    use_bf16 = (cfg["training"]["precision"] == "bf16") and (device == "cuda")
    history = []
    best_ndcg = -1.0
    patience_left = cfg["training"]["early_stopping_patience_evals"]
    last_ckpt_tag = None
    best_ckpt_tag = None
    t_start = time.time()

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum, epoch_loss_n = 0.0, 0

        epoch_bar = pbar(loader, desc=f"epoch {epoch}/{epochs}")
        for seq, positives, pos_mask, cand in epoch_bar:
            seq = seq.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            pos_mask = pos_mask.to(device, non_blocking=True)
            cand = cand.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                queries = model(seq)
                hn = sample_hard_negatives(cand, k=cfg["training"]["num_hard_negs"])
                rn = sample_random_negatives(
                    k=cfg["training"]["num_random_negs"],
                    vocab_size=cfg["model"]["vocab_size"],
                    device=seq.device,
                )
                logits = build_logits(queries, positives, hn, rn, model.item_emb)
                loss = masked_ce_loss(logits, pos_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += loss.item()
            epoch_loss_n += 1
            epoch_bar.set_postfix({
                "loss": f"{epoch_loss_sum / epoch_loss_n:.3f}",
                "lr": f"{scheduler.get_last_lr()[1]:.2e}",
            })

        avg_train_loss = epoch_loss_sum / max(1, epoch_loss_n)

        # ── Validation (every val_every_n_epochs OR always for smoke) ────────
        should_eval = smoke or (epoch % cfg["training"]["val_every_n_epochs"] == 0)
        if should_eval:
            val_metrics = evaluate_pipeline_mode(
                model=model,
                padded=val_padded,
                lengths=val_lengths,
                candidates=val_candidates,
                batch_size=cfg["training"]["batch_size"],
                device=device,
                use_bf16=use_bf16,
            )
            wall_min = (time.time() - t_start) / 60
            print(
                f"[epoch {epoch}/{epochs}] "
                f"train_loss={avg_train_loss:.3f}  "
                f"val_R@10={val_metrics['R@10']:.3f}  "
                f"val_R@100={val_metrics['R@100']:.3f}  "
                f"val_NDCG@10={val_metrics['NDCG@10']:.3f}  "
                f"wall={wall_min:.1f}m",
                flush=True,
            )
            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val": val_metrics,
                "wall_min": wall_min,
            })

            # Checkpoint management
            new_tag = f"ckpt_epoch_{epoch}"
            save_checkpoint(run_dir, new_tag, model)

            if val_metrics["NDCG@10"] > best_ndcg:
                best_ndcg = val_metrics["NDCG@10"]
                # Promote new ckpt to "best"
                if best_ckpt_tag is not None and best_ckpt_tag != last_ckpt_tag:
                    shutil.rmtree(run_dir / best_ckpt_tag, ignore_errors=True)
                best_ckpt_tag = new_tag
                patience_left = cfg["training"]["early_stopping_patience_evals"]
            else:
                # Delete the non-best, non-last older ckpt
                if last_ckpt_tag is not None and last_ckpt_tag != best_ckpt_tag:
                    shutil.rmtree(run_dir / last_ckpt_tag, ignore_errors=True)
                patience_left -= 1

            last_ckpt_tag = new_tag

            if (not smoke) and patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val improvement).")
                break

    # ── Final: save best + last as their final names ─────────────────────────
    if best_ckpt_tag is not None:
        shutil.copytree(run_dir / best_ckpt_tag, run_dir / "best", dirs_exist_ok=True)
    if last_ckpt_tag is not None:
        shutil.copytree(run_dir / last_ckpt_tag, run_dir / "final", dirs_exist_ok=True)

    # Save history
    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"best_ndcg": best_ndcg, "history": history, "run_dir": str(run_dir)}
```

- [ ] **Step 4: Run the full test suite to verify nothing regressed**

Run: `pytest stage_2/tests/ -v`
Expected: all tests still pass. `train.py` itself has no new tests — it's validated by Tier 2/3.

- [ ] **Step 5: Commit**

```bash
git add stage_2/training/train.py stage_2/tests/test_loss.py
git commit -m "Add SASRec training loop with discriminative LR, early stopping, checkpointing"
```

---

## Task 8: Evaluation (pipeline-mode + unconstrained)

**Files:**
- Create: `stage_2/training/evaluate.py`

- [ ] **Step 1: Write `stage_2/training/evaluate.py`**

```python
"""Pipeline-mode and unconstrained evaluation for Stage 2.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §6.
"""

import math
import sys
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from stage_2.models.sasrec import SASRec


def _pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)


def _recall_at_k(top_k_ids: np.ndarray, targets: np.ndarray, k: int) -> float:
    """top_k_ids: (B, k), targets: (B,) → fraction where target in top_k."""
    return float((top_k_ids[:, :k] == targets[:, None]).any(axis=1).mean())


def _ndcg_at_k(top_k_ids: np.ndarray, targets: np.ndarray, k: int) -> float:
    """NDCG@K with binary relevance (single target). top_k_ids must be sorted best-first."""
    match = (top_k_ids[:, :k] == targets[:, None])             # (B, k) bool
    found = match.any(axis=1)
    ranks = match.argmax(axis=1) + 1                            # 1-indexed
    gain = np.where(found, 1.0 / np.log2(ranks + 1), 0.0)
    return float(gain.mean())


def evaluate_pipeline_mode(
    model: SASRec,
    padded: np.ndarray,
    lengths: np.ndarray,
    candidates: np.ndarray,
    batch_size: int = 256,
    device: str = "cuda",
    use_bf16: bool = False,
) -> Dict[str, float]:
    """Pipeline-mode eval: score this playlist's 1000 Stage 1 candidates.

    Returns dict with R@10, R@50, R@100, NDCG@10. Skips playlists with length < 2.
    """
    model.eval()
    N, L = padded.shape

    r_at = {10: 0, 50: 0, 100: 0}
    ndcg_total = 0.0
    counted = 0

    with torch.no_grad():
        for start in _pbar(range(0, N, batch_size), desc="val pipeline"):
            end = min(start + batch_size, N)
            B = end - start

            seqs_np = padded[start:end].copy()
            lens = lengths[start:end]
            cands_np = candidates[start:end]

            # Build seq = playlist[:-1] padded (zero out the last real token)
            seqs = seqs_np.copy()
            valid = lens >= 2
            for j in range(B):
                if not valid[j]:
                    continue
                # Position of the last real token = L - 1 (left-padded), so zero it out
                seqs[j, L - 1] = 0

            # Targets = last real token of original seq
            targets = np.zeros(B, dtype=np.int64)
            targets[valid] = seqs_np[valid, L - 1]

            seq_t = torch.from_numpy(seqs.astype(np.int64)).to(device)
            cand_t = torch.from_numpy(cands_np.astype(np.int64)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(seq_t)                              # (B, L, D)
                # Query = output at position L-2 (the new "last" after zeroing L-1)
                # But for left-padded sequences, the query should be at the position
                # right before the held-out token, which after zeroing is L-2.
                # Equivalent: use position L-1 of the zeroed sequence — but that's PAD now,
                # so we must use L-2.
                queries = out[:, L - 2, :]                      # (B, D)
                cand_emb = model.item_emb(cand_t)               # (B, 1000, D)
                scores = (queries.unsqueeze(1) * cand_emb).sum(-1)   # (B, 1000)
                # Mask context tokens (seqs after zeroing): set their scores to -inf
                for j in range(B):
                    if not valid[j]:
                        continue
                    real_seq = seqs_np[j][seqs_np[j] != 0]
                    ctx = real_seq[:-1]
                    if ctx.size == 0:
                        continue
                    ctx_t = torch.from_numpy(ctx.astype(np.int64)).to(device)
                    # Build a mask over candidates (vectorized)
                    isin = (cand_t[j].unsqueeze(0) == ctx_t.unsqueeze(1)).any(0)
                    scores[j, isin] = float("-inf")
                # Mask PAD candidates (id 0) defensively
                scores[cand_t == 0] = float("-inf")

            # Top-100 sorted, then trim for R@10/R@50/R@100/NDCG@10
            top_scores, top_idx = torch.topk(scores, k=100, dim=1)
            top_ids = torch.gather(cand_t, 1, top_idx).cpu().numpy()  # (B, 100)

            v = valid
            if v.sum() == 0:
                continue
            for k in (10, 50, 100):
                r_at[k] += int(((top_ids[v, :k]) == targets[v, None]).any(axis=1).sum())
            ndcg_total += _ndcg_at_k(top_ids[v], targets[v], k=10) * v.sum()
            counted += int(v.sum())

    metrics = {
        "R@10":   r_at[10] / max(1, counted),
        "R@50":   r_at[50] / max(1, counted),
        "R@100":  r_at[100] / max(1, counted),
        "NDCG@10": ndcg_total / max(1, counted),
        "n_evaluated": counted,
    }
    return metrics


def evaluate_unconstrained(
    model: SASRec,
    padded: np.ndarray,
    lengths: np.ndarray,
    vocab_size: int,
    batch_size: int = 256,
    device: str = "cuda",
    use_bf16: bool = False,
) -> Dict[str, float]:
    """Unconstrained eval: score against full vocab. ~10 min on A10G for 150K playlists."""
    model.eval()
    N, L = padded.shape
    r_at = {10: 0, 50: 0, 100: 0, 1000: 0}
    ndcg_total = 0.0
    counted = 0

    with torch.no_grad():
        item_emb_w = model.item_emb.weight                       # (vocab, D)
        for start in _pbar(range(0, N, batch_size), desc="test unconstrained"):
            end = min(start + batch_size, N)
            B = end - start

            seqs_np = padded[start:end].copy()
            lens = lengths[start:end]
            seqs = seqs_np.copy()
            valid = lens >= 2
            for j in range(B):
                if not valid[j]:
                    continue
                seqs[j, L - 1] = 0
            targets = np.zeros(B, dtype=np.int64)
            targets[valid] = seqs_np[valid, L - 1]

            seq_t = torch.from_numpy(seqs.astype(np.int64)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(seq_t)                               # (B, L, D)
                queries = out[:, L - 2, :]                       # (B, D)
                scores = queries @ item_emb_w.T                  # (B, vocab)
                scores[:, 0] = float("-inf")                     # mask PAD
                # Mask context tokens
                for j in range(B):
                    if not valid[j]:
                        continue
                    real_seq = seqs_np[j][seqs_np[j] != 0]
                    ctx = real_seq[:-1]
                    if ctx.size == 0:
                        continue
                    ctx_t = torch.from_numpy(ctx.astype(np.int64)).to(device)
                    scores[j, ctx_t] = float("-inf")

            top1000 = torch.topk(scores, k=1000, dim=1).indices.cpu().numpy()  # (B, 1000)

            v = valid
            if v.sum() == 0:
                continue
            for k in (10, 50, 100, 1000):
                r_at[k] += int(((top1000[v, :k]) == targets[v, None]).any(axis=1).sum())
            ndcg_total += _ndcg_at_k(top1000[v], targets[v], k=10) * v.sum()
            counted += int(v.sum())

    return {
        "R@10":    r_at[10] / max(1, counted),
        "R@50":    r_at[50] / max(1, counted),
        "R@100":   r_at[100] / max(1, counted),
        "R@1000":  r_at[1000] / max(1, counted),
        "NDCG@10": ndcg_total / max(1, counted),
        "n_evaluated": counted,
    }


def derived_metrics(
    pipeline: Dict[str, float],
    unconstrained: Dict[str, float],
    stage1_recall_at_100: float = 0.151,
    stage1_recall_at_1000: float = 0.407,
) -> Dict[str, float]:
    """Pipeline lift, headroom gap, Stage-1-recoverable rate (spec §6.4)."""
    return {
        "pipeline_lift_R@100": pipeline["R@100"] / max(1e-9, stage1_recall_at_100),
        "headroom_gap_R@100": unconstrained["R@100"] - pipeline["R@100"],
        "stage1_recoverable_rate": pipeline["R@10"] / max(1e-9, stage1_recall_at_1000),
    }
```

- [ ] **Step 2: Quick smoke check that the eval functions import cleanly**

Run: `python -c "from stage_2.training.evaluate import evaluate_pipeline_mode, evaluate_unconstrained, derived_metrics; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add stage_2/training/evaluate.py
git commit -m "Add pipeline-mode + unconstrained eval and derived metrics"
```

---

## Task 9: Inference (predict + precompute_test_top100)

**Files:**
- Create: `stage_2/inference/predict.py`

- [ ] **Step 1: Write `stage_2/inference/predict.py`**

```python
"""Stage 2 inference.

- `predict_top_100`: single-playlist API for Stage 3 / Stage 4 UI.
- `precompute_test_top100`: bulk-precompute Stage 2 top-100 for every test playlist.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §9 (Stage 3 handoff).
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from stage_2.data.dataset import pad_left
from stage_2.models.sasrec import SASRec


def _pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)


def predict_top_100(
    model: SASRec,
    playlist_shifted_ids: list,           # length-N list in +1-shifted space
    stage1_candidates: np.ndarray,        # (1000,) int32 — Stage 1's top-1000 for this playlist
    device: str = "cuda",
    max_seq_len: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Stage 2 on a single playlist. Returns (top_ids, top_scores) both shape (100,)."""
    model.eval()
    padded = np.array(pad_left(playlist_shifted_ids, L=max_seq_len), dtype=np.int64)
    seq = torch.from_numpy(padded).unsqueeze(0).to(device)        # (1, L)

    with torch.no_grad():
        out = model(seq)                                          # (1, L, D)
        query = out[0, -1, :]                                     # (D,) — last position
        cand_t = torch.from_numpy(stage1_candidates.astype(np.int64)).to(device)  # (1000,)
        cand_emb = model.item_emb(cand_t)                         # (1000, D)
        scores = cand_emb @ query                                 # (1000,)
        # Mask any candidate that's already in the context
        ctx = set(int(x) for x in playlist_shifted_ids if x != 0)
        if ctx:
            mask = torch.tensor([int(c) in ctx for c in stage1_candidates],
                                device=device)
            scores[mask] = float("-inf")
        scores[cand_t == 0] = float("-inf")

        top_scores, top_idx = torch.topk(scores, k=100)
        top_ids = cand_t[top_idx].cpu().numpy().astype(np.int32)
        top_scores = top_scores.cpu().numpy().astype(np.float32)

    return top_ids, top_scores


def precompute_test_top100(
    model: SASRec,
    padded: np.ndarray,           # (N, L) int32 — test set padded sequences
    lengths: np.ndarray,          # (N,)   int16
    candidates: np.ndarray,       # (N, 1000) int32 — Stage 1 candidates for the test set
    out_dir: Path,
    batch_size: int = 256,
    device: str = "cuda",
    use_bf16: bool = False,
) -> None:
    """Run Stage 2 on every test playlist, save top-100 IDs + scores."""
    model.eval()
    N, L = padded.shape
    top_ids_out = np.zeros((N, 100), dtype=np.int32)
    top_scores_out = np.zeros((N, 100), dtype=np.float32)

    with torch.no_grad():
        for start in _pbar(range(0, N, batch_size), desc="inferring test top100"):
            end = min(start + batch_size, N)
            B = end - start
            seqs_np = padded[start:end].copy()
            lens = lengths[start:end]
            cands_np = candidates[start:end]

            # As in evaluate: zero out the last real token so the query at L-2 predicts it
            seqs = seqs_np.copy()
            valid = lens >= 2
            for j in range(B):
                if valid[j]:
                    seqs[j, L - 1] = 0

            seq_t = torch.from_numpy(seqs.astype(np.int64)).to(device)
            cand_t = torch.from_numpy(cands_np.astype(np.int64)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(seq_t)
                queries = out[:, L - 2, :]                             # (B, D)
                cand_emb = model.item_emb(cand_t)                      # (B, 1000, D)
                scores = (queries.unsqueeze(1) * cand_emb).sum(-1)     # (B, 1000)

                # Mask context tokens and PAD candidates
                for j in range(B):
                    if not valid[j]:
                        continue
                    real_seq = seqs_np[j][seqs_np[j] != 0]
                    ctx = real_seq[:-1]
                    if ctx.size == 0:
                        continue
                    ctx_t = torch.from_numpy(ctx.astype(np.int64)).to(device)
                    isin = (cand_t[j].unsqueeze(0) == ctx_t.unsqueeze(1)).any(0)
                    scores[j, isin] = float("-inf")
                scores[cand_t == 0] = float("-inf")

            # Sort full top-100 (we need scores sorted, not just argpartition)
            top_scores, top_idx = torch.topk(scores, k=100, dim=1)     # (B, 100)
            top_ids = torch.gather(cand_t, 1, top_idx)                  # (B, 100)

            top_ids_out[start:end] = top_ids.cpu().numpy().astype(np.int32)
            top_scores_out[start:end] = top_scores.cpu().float().numpy().astype(np.float32)

    np.save(out_dir / "test_top100.npy", top_ids_out)
    np.save(out_dir / "test_top100_scores.npy", top_scores_out)
    print(f"Wrote test_top100.npy {top_ids_out.shape} and test_top100_scores.npy {top_scores_out.shape}")
```

- [ ] **Step 2: Smoke check the imports**

Run: `python -c "from stage_2.inference.predict import predict_top_100, precompute_test_top100; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add stage_2/inference/predict.py
git commit -m "Add single-playlist inference and precompute_test_top100"
```

---

## Task 10: Modal app

**Files:**
- Create: `stage_2/modal_app.py`

- [ ] **Step 1: Write `stage_2/modal_app.py`**

```python
"""Modal entry point for Stage 2 SASRec.

Local usage:
    modal run stage_2/modal_app.py --cmd cache
    modal run stage_2/modal_app.py --cmd train --run-id smoke --smoke
    modal run stage_2/modal_app.py --cmd train --run-id main
    modal run stage_2/modal_app.py --cmd eval --run-id main
    modal run stage_2/modal_app.py --cmd infer --run-id main

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §7.
"""

import json
from pathlib import Path

import modal
import yaml

CONFIG_PATH = "/repo/stage_2/config.yaml"

with open(Path(__file__).parent / "config.yaml") as f:
    CFG_LOCAL = yaml.safe_load(f)

stub = modal.App(CFG_LOCAL["modal"]["app_name"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy>=1.24",
        "tqdm>=4.67",
        "pyyaml>=6.0",
    )
    .add_local_dir(".", "/repo")
)

volume = modal.Volume.from_name(CFG_LOCAL["modal"]["volume_name"], create_if_missing=True)

VOL_PATH = {"vol": volume}
GPU = CFG_LOCAL["modal"]["gpu"]
TIMEOUT = CFG_LOCAL["modal"]["function_timeout_sec"]


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def cache_stage1_candidates():
    """Pass 1 (pad playlists) + Pass 2 (Stage 1 candidate top-1000)."""
    import sys
    sys.path.insert(0, "/repo")
    from stage_2.training.preprocess import run_preprocessing
    cfg = _load_cfg()
    run_preprocessing(
        cfg=cfg,
        raw_playlists_path=cfg["paths"]["playlists_raw"],
        als_factors_path=cfg["paths"]["als_factors"],
        output_dir="/vol/derived",
    )
    volume.commit()


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def train(run_id: str = "main", smoke: bool = False):
    import sys
    sys.path.insert(0, "/repo")
    from stage_2.training.train import train_main
    cfg = _load_cfg()
    out = train_main(cfg, run_id=run_id, smoke=smoke)
    print(json.dumps({"best_ndcg": out["best_ndcg"], "run_dir": out["run_dir"]}, indent=2))
    volume.commit()


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def evaluate(run_id: str = "main", mode: str = "both"):
    """mode ∈ {'pipeline', 'unconstrained', 'both'}."""
    import sys
    sys.path.insert(0, "/repo")
    import numpy as np
    import torch
    from stage_2.models.sasrec import SASRec
    from stage_2.training.train import load_checkpoint
    from stage_2.training.evaluate import (
        evaluate_pipeline_mode, evaluate_unconstrained, derived_metrics,
    )

    cfg = _load_cfg()
    run_dir = Path(cfg["paths"]["runs_dir"]) / run_id

    padded = np.load(cfg["paths"]["playlists_padded"])
    lengths = np.load(cfg["paths"]["playlists_lengths"])
    candidates = np.load(cfg["paths"]["candidates"])

    n_train = cfg["data"]["train_size"]
    n_val = cfg["data"]["val_size"]
    test_padded = padded[n_train + n_val:]
    test_lengths = lengths[n_train + n_val:]
    test_candidates = candidates[n_train + n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SASRec(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(run_dir, "best", model)

    results = {}
    if mode in ("pipeline", "both"):
        results["pipeline"] = evaluate_pipeline_mode(
            model, test_padded, test_lengths, test_candidates,
            batch_size=cfg["training"]["batch_size"],
            device=device, use_bf16=(device == "cuda"),
        )
    if mode in ("unconstrained", "both"):
        results["unconstrained"] = evaluate_unconstrained(
            model, test_padded, test_lengths,
            vocab_size=cfg["model"]["vocab_size"],
            batch_size=cfg["training"]["batch_size"],
            device=device, use_bf16=(device == "cuda"),
        )
    if "pipeline" in results and "unconstrained" in results:
        results["derived"] = derived_metrics(results["pipeline"], results["unconstrained"])

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    volume.commit()


@stub.function(image=image, volumes={"/vol": volume}, gpu=GPU, timeout=TIMEOUT)
def precompute(run_id: str = "main"):
    import sys
    sys.path.insert(0, "/repo")
    import numpy as np
    import torch
    from stage_2.models.sasrec import SASRec
    from stage_2.training.train import load_checkpoint
    from stage_2.inference.predict import precompute_test_top100

    cfg = _load_cfg()
    run_dir = Path(cfg["paths"]["runs_dir"]) / run_id

    padded = np.load(cfg["paths"]["playlists_padded"])
    lengths = np.load(cfg["paths"]["playlists_lengths"])
    candidates = np.load(cfg["paths"]["candidates"])

    n_train = cfg["data"]["train_size"]
    n_val = cfg["data"]["val_size"]
    test_padded = padded[n_train + n_val:]
    test_lengths = lengths[n_train + n_val:]
    test_candidates = candidates[n_train + n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SASRec(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        ffn_dim=cfg["model"]["ffn_dim"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(run_dir, "best", model)

    precompute_test_top100(
        model=model,
        padded=test_padded,
        lengths=test_lengths,
        candidates=test_candidates,
        out_dir=run_dir,
        batch_size=cfg["training"]["batch_size"],
        device=device,
        use_bf16=(device == "cuda"),
    )
    volume.commit()


@stub.local_entrypoint()
def main(
    cmd: str,
    run_id: str = "main",
    smoke: bool = False,
    mode: str = "both",
):
    if cmd == "cache":
        cache_stage1_candidates.remote()
    elif cmd == "train":
        train.remote(run_id=run_id, smoke=smoke)
    elif cmd == "eval":
        evaluate.remote(run_id=run_id, mode=mode)
    elif cmd == "infer":
        precompute.remote(run_id=run_id)
    else:
        raise ValueError(f"Unknown cmd: {cmd}. Use cache|train|eval|infer.")
```

- [ ] **Step 2: Smoke check the Modal app loads (does not deploy)**

Run: `python -c "import sys; sys.path.insert(0, '.'); from stage_2 import modal_app; print('ok')"`
Expected: `ok` (Modal validates the App definition lazily; this just checks imports).

- [ ] **Step 3: Commit**

```bash
git add stage_2/modal_app.py
git commit -m "Add Modal app: cache, train, eval, precompute entry points"
```

---

## Task 11: Tier 1 validation — full unit-test pass

**Files:** none (verification step only)

- [ ] **Step 1: Run the full unit-test suite**

Run: `pytest stage_2/tests/ -v`

Expected: all tests pass. Specifically:
- `test_sasrec.py` — 5 tests
- `test_dataset.py` — 7 tests
- `test_loss.py` — 5 tests (4 + 1 integration)
- `test_preprocess.py` — 3 tests

Total: 20 tests, all green.

- [ ] **Step 2: If any test fails, stop and debug**

Do NOT proceed to Tier 2 with red tests. Fix the failures locally before moving on.

- [ ] **Step 3: No commit (no code changes)**

---

## Task 12: Tier 2 validation — tiny local training

**Files:** none (verification step only)

This tier validates the full training loop end-to-end on a tiny synthetic dataset, locally, no Modal involvement.

- [ ] **Step 1: Set up Modal authentication if not done**

```bash
modal token new
```

This opens a browser to authenticate. Follow the prompts.

Expected: `Token created successfully`.

- [ ] **Step 2: Verify Modal sees the user**

```bash
modal profile current
```

Expected: prints the active profile.

- [ ] **Step 3: Upload Stage 1 checkpoints to the Modal Volume**

From the repo root, run:

```bash
modal volume create stage2-data 2>/dev/null || true
modal volume put stage2-data stage_1/checkpoints/als_item_factors.npy /inputs/als_item_factors.npy
modal volume put stage2-data stage_1/checkpoints/uri_to_id.json       /inputs/uri_to_id.json
modal volume put stage2-data stage_1/checkpoints/playlists.npy        /inputs/playlists.npy
modal volume ls stage2-data inputs/
```

Expected: `modal volume ls` lists three files totaling ~1.4 GB.

If you don't have the Stage 1 checkpoints locally, download them from Google Drive first (see `stage_1/STAGE_1_SUMMARY.md`).

- [ ] **Step 4: Create a tiny synthetic dataset locally for the dry run**

Create `stage_2/tests/tier2_dry_run.py`:

```python
"""Tier 2: Run a tiny synthetic training pass locally to validate the loop.

Synthesizes 1000 playlists × random Stage 1 candidates, runs 3 mini-epochs at
batch=32, max_seq=20. Pass if train loss decreases.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from stage_2.training.train import train_main


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tier2_"))
    print(f"Tier 2 sandbox: {tmp}")

    # ── Synthesize data ──────────────────────────────────────────────────────
    np.random.seed(0)
    vocab = 200            # tiny vocab
    N = 1000               # 1000 playlists
    L = 20
    train_n = 700
    val_n = 150

    # Random lengths in [10, 20] for train, [5, 20] for val/test
    lengths = np.zeros(N, dtype=np.int16)
    lengths[:train_n] = np.random.randint(10, 21, size=train_n)
    lengths[train_n:] = np.random.randint(5, 21, size=N - train_n)

    padded = np.zeros((N, L), dtype=np.int32)
    for i in range(N):
        seq_ids = np.random.randint(1, vocab, size=int(lengths[i]))
        padded[i, L - int(lengths[i]):] = seq_ids

    # Synthetic Stage 1 candidates
    candidates = np.random.randint(1, vocab, size=(N, 64), dtype=np.int32)

    # Save into the sandbox
    np.save(tmp / "playlists_padded.npy", padded)
    np.save(tmp / "playlists_lengths.npy", lengths)
    np.save(tmp / "candidates.npy", candidates)

    # Synthetic ALS factors
    als = np.random.randn(vocab - 1, 32).astype(np.float32)
    np.save(tmp / "als_item_factors.npy", als)

    cfg = {
        "model": {
            "d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_dim": 64,
            "max_seq_len": L, "dropout": 0.0, "vocab_size": vocab,
        },
        "data": {
            "min_train_length": 10,
            "train_size": train_n, "val_size": val_n, "test_size": N - train_n - val_n,
            "val_eval_subset": 100,
        },
        "training": {
            "batch_size": 32, "epochs": 3, "warmup_steps": 10,
            "peak_lr_transformer": 1e-3, "peak_lr_embedding": 1e-4,
            "weight_decay": 0.01, "grad_clip": 1.0, "precision": "fp32",
            "seed": 42,
            "num_hard_negs": 16, "num_random_negs": 16,
            "val_every_n_epochs": 1, "early_stopping_patience_evals": 100,
            "keep_best_and_last_only": True,
        },
        "eval": {"candidates_k": 64, "ranked_top_k": 32,
                 "metrics_recall_k": [10, 50, 100], "ndcg_k": 10},
        "modal": {"gpu": "a10g", "volume_name": "ignore", "app_name": "ignore",
                  "function_timeout_sec": 1},
        "paths": {
            "vol_root": str(tmp),
            "als_factors": str(tmp / "als_item_factors.npy"),
            "uri_to_id": "",
            "playlists_raw": "",
            "playlists_padded": str(tmp / "playlists_padded.npy"),
            "playlists_lengths": str(tmp / "playlists_lengths.npy"),
            "candidates": str(tmp / "candidates.npy"),
            "runs_dir": str(tmp / "runs"),
        },
    }

    out = train_main(cfg, run_id="tier2", smoke=False)
    print(f"Result: best_ndcg={out['best_ndcg']}")
    print(f"History entries: {len(out['history'])}")

    # Pass criteria
    history = out["history"]
    assert len(history) >= 2, "Need at least 2 val evals to compare"
    losses = [h["train_loss"] for h in history]
    print(f"Train losses: {losses}")
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    recalls = [h["val"]["R@100"] for h in history]
    print(f"Val R@100 over epochs: {recalls}")
    assert max(recalls) > 0.02, f"Val R@100 never above 0.02 (best={max(recalls)})"

    print("TIER 2 PASS")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the Tier 2 script**

Run: `python stage_2/tests/tier2_dry_run.py`

Expected: ends with `TIER 2 PASS`. Total runtime ~3-5 minutes on CPU, faster on MPS/CUDA.

- [ ] **Step 6: If Tier 2 fails, stop and debug**

Common failure modes (see spec §10):
- Loss is NaN: try precision="fp32" instead of bf16 — Tier 2 already uses fp32. If still NaN, suspect a masking bug.
- Loss doesn't decrease: check that the positive is at index 0 in `build_logits`, and that the LR isn't too low.
- Val R@100 stays 0: check that `evaluate_pipeline_mode` builds the right query position (L-2 after zeroing L-1).

Do NOT proceed to Tier 3 with a failing Tier 2.

- [ ] **Step 7: Commit the Tier 2 script for future re-runs**

```bash
git add stage_2/tests/tier2_dry_run.py
git commit -m "Add tier 2 dry-run script for local end-to-end validation"
```

---

## Task 13: Tier 3 validation — Modal smoke run

**Files:** none (verification step only)

- [ ] **Step 1: Run candidate caching on Modal (one-time prep)**

```bash
modal run stage_2/modal_app.py --cmd cache
```

Expected logs:
- "Loading playlists from /vol/inputs/playlists.npy..."
- "loaded 1,000,000 playlists"
- "Pass 1: padding playlists..."
- "wrote playlists_padded.npy (1000000, 50)..."
- "Pass 2: computing Stage 1 candidates..."
- tqdm bar "caching candidates" running for ~5 minutes
- "wrote candidates.npy (1000000, 1000)"

Wall-clock: ~5-7 min. Cost: ~$0.10. Verify with `modal volume ls stage2-data derived/` that all three files exist.

- [ ] **Step 2: Run the smoke training (1 epoch)**

```bash
modal run stage_2/modal_app.py --cmd train --run-id smoke --smoke
```

Expected logs (in order):
- "Device: cuda"
- "Loading derived data..."
- "Train (filtered len>=10): ~600,000  Val: 150,000"
- "Loading ALS factors into model..."
- tqdm bar "epoch 1/1" running for ~10-15 min
- Final summary line: `[epoch 1/1] train_loss=...  val_R@10=...  val_R@100=...  val_NDCG@10=...  wall=...m`

Wall-clock: ~10-15 min including data load. Cost: ~$0.30.

- [ ] **Step 3: Verify Tier 3 pass criteria**

Pull the smoke run history locally:

```bash
modal volume get stage2-data runs/smoke/train_history.json ./tmp_smoke_history.json
cat tmp_smoke_history.json
```

Expected (loosely):
- `train_loss` is a finite number (not NaN/Inf), substantially below `log(257)` ≈ 5.55 (random baseline)
- `val.R@100` > 0.05 (sanity — even 1 epoch should beat random by a healthy margin)
- `val.NDCG@10` > 0.01
- `wall_min` < 20 (within 1.5× of the 10-min estimate)

If any of these fail: **STOP. Do not launch the full run.** See spec §10 for diagnosis. Common: epoch wall-clock 3× estimate usually means the DataLoader is CPU-bound — increase `num_workers` from 2 to 4 in `train.py`.

- [ ] **Step 4: Verify the unconstrained eval pipeline also runs (sanity)**

```bash
modal run stage_2/modal_app.py --cmd eval --run-id smoke --mode unconstrained
```

Expected: prints metrics JSON. R@1000 should be > Stage 1's 0.407 if the model is actually doing better than ALS.

- [ ] **Step 5: Clean up the smoke artifacts to free Volume space**

```bash
modal volume rm stage2-data runs/smoke -r
```

- [ ] **Step 6: No commit (verification only). Record the smoke result**

Record the result locally (e.g., in a scratch note): "Smoke run on YYYY-MM-DD: train_loss=..., val_R@100=..., val_NDCG@10=..., wall=...m. PASS."

---

## Task 14: Full Modal training run

**Files:** none (execution step only)

- [ ] **Step 1: Launch the full 15-epoch run**

```bash
modal run --detach stage_2/modal_app.py --cmd train --run-id main
```

The `--detach` flag lets the function run even if your local terminal disconnects. Modal will keep streaming logs to its web UI.

Wall-clock: ~2.5-3.5 hours. Cost: ~$3-4. Monitor in Modal dashboard or pull logs with `modal app logs stage2-sasrec`.

- [ ] **Step 2: Monitor progress**

Either watch the live terminal or `modal app logs stage2-sasrec --tail`. Expected pattern:

```
epoch 1/15: 100%|███| 2340/2340 [12:00<00:00, loss=4.52, lr=9.97e-04]
val pipeline: 100%|███| 40/40 [01:50<00:00, R@100=0.18]
[epoch 2/15] train_loss=4.21  val_R@10=0.07  val_R@100=0.21  val_NDCG@10=0.04  wall=27.5m
...
```

- [ ] **Step 3: If anything goes wrong mid-run**

- **Loss is NaN**: function will likely error. Re-run with `--smoke` to reproduce locally on a 1-epoch run with bf16 disabled (edit `config.yaml` to `precision: fp32`).
- **Early stopping triggered**: this is fine. Best checkpoint will be at `runs/main/best/`.
- **OOM**: shouldn't happen with `batch_size=256` on A10G 24 GB, but if it does, reduce batch_size in config to 128 and re-run.

- [ ] **Step 4: Verify completion**

When the function returns, check:

```bash
modal volume ls stage2-data runs/main/
```

Expected files:
- `best/model.pt`, `best/item_embeddings.npy`
- `final/model.pt`, `final/item_embeddings.npy`
- `train_history.json`
- `config.yaml`

If `best/` is missing but `final/` exists, training completed but no val improvement was ever recorded — investigate.

- [ ] **Step 5: No commit (data on Modal only)**

---

## Task 15: Final evaluation and artifact precompute

**Files:** none (execution step only)

- [ ] **Step 1: Run final test eval (pipeline + unconstrained)**

```bash
modal run stage_2/modal_app.py --cmd eval --run-id main --mode both
```

Expected: prints final metrics, writes `runs/main/test_metrics.json`. Wall-clock ~15 min.

Sanity checks:
- `pipeline.R@1000` should be exactly 0.407 (= Stage 1's R@1000, locked)
- `pipeline.R@100` should be > 0.18 (better than Stage 1's 0.151)
- `pipeline.NDCG@10` should be > 0.05 (Stage 1 was 0.018)
- `unconstrained.R@100` should be ≥ `pipeline.R@100` (always, by construction)

If `pipeline.R@1000` ≠ 0.407: there is a bug in the eval loop (the candidate set was somehow different from Stage 1's).

- [ ] **Step 2: Precompute test top-100 for Stage 3 handoff**

```bash
modal run stage_2/modal_app.py --cmd infer --run-id main
```

Expected: writes `runs/main/test_top100.npy` and `runs/main/test_top100_scores.npy`. Wall-clock ~5 min.

- [ ] **Step 3: No commit (data on Modal only)**

---

## Task 16: Pull artifacts locally and verify

**Files:** updates to `stage_2/checkpoints/` (gitignored)

- [ ] **Step 1: Pull small artifacts locally**

```bash
mkdir -p stage_2/checkpoints
modal volume get stage2-data runs/main/test_top100.npy           ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_top100_scores.npy    ./stage_2/checkpoints/
modal volume get stage2-data runs/main/train_history.json        ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_metrics.json         ./stage_2/checkpoints/
modal volume get stage2-data runs/main/best/model.pt             ./stage_2/checkpoints/best_model.pt
modal volume get stage2-data runs/main/config.yaml               ./stage_2/checkpoints/run_config.yaml
```

Optional (1.1 GB, only if you want local inference): pull the item embeddings:

```bash
modal volume get stage2-data runs/main/best/item_embeddings.npy  ./stage_2/checkpoints/best_item_embeddings.npy
```

- [ ] **Step 2: Verify the artifacts load correctly**

Run:

```bash
python - <<'EOF'
import json
import numpy as np

t100 = np.load("stage_2/checkpoints/test_top100.npy")
scores = np.load("stage_2/checkpoints/test_top100_scores.npy")
hist = json.load(open("stage_2/checkpoints/train_history.json"))
metrics = json.load(open("stage_2/checkpoints/test_metrics.json"))

assert t100.shape == (150_000, 100), f"unexpected shape {t100.shape}"
assert t100.dtype == np.int32
assert scores.shape == (150_000, 100)
assert (t100 >= 1).all() and (t100 < 2_262_293).all(), "IDs out of +1-shifted range"

print(f"test_top100.npy        OK  {t100.shape} {t100.dtype}")
print(f"test_top100_scores.npy OK  {scores.shape} {scores.dtype}")
print(f"train_history          OK  {len(hist)} epochs")
print(f"test_metrics           OK")
print(json.dumps(metrics, indent=2))
EOF
```

Expected: prints OK for each file and the final metrics JSON.

- [ ] **Step 3: Update `.gitignore` if not already present**

Confirm `stage_2/checkpoints/` is in `.gitignore` (Task 1 added it). Do NOT commit these files — they're large and reproducible from the Modal Volume.

- [ ] **Step 4: Final commit — mark Stage 2 complete**

This is the final commit. Update the project README to reflect Stage 2 status:

Modify `README.md`. Change the Stage 2 row from:

```
| Stage 2 — SASRec ranking | — | [stage_2/README.md](stage_2/README.md) |
```

to:

```
| Stage 2 — SASRec ranking | DONE | [stage_2/README.md](stage_2/README.md) |
```

And update the pipeline diagram at the top from `Stage 2 · SASRec sequential ranking → top 100 [TODO]` to `[DONE]`.

Then commit:

```bash
git add README.md
git commit -m "Mark Stage 2 SASRec as complete"
```

---

## Self-Review Output

After writing the plan, walked through the spec section-by-section:

| Spec section | Covered by task(s) |
|---|---|
| §1 Context, goals, success criteria | All tasks (success criteria checked in Task 15 Step 1) |
| §2.1 Inputs | Task 12 Step 3 (upload to Volume) |
| §2.2 Outputs | Tasks 7 (checkpoints), 8 (metrics), 9 (test_top100), 16 (local pull) |
| §2.3 ID convention | Task 4 (`shift_ids_plus_one`), Task 6 (`build_shifted_als_factors`) |
| §3 Model | Task 3 (SASRec) |
| §4 Data pipeline | Task 6 (preprocess Pass 1 + Pass 2) |
| §5 Training | Tasks 5 (loss), 7 (train loop), 2 (hyperparameters in config) |
| §6 Evaluation | Task 8 (pipeline + unconstrained + derived) |
| §7 Modal infrastructure | Task 10 (`modal_app.py`) |
| §8 Pre-launch validation tiers | Tasks 11 (Tier 1), 12 (Tier 2), 13 (Tier 3) |
| §9 Stage 3 handoff | Task 9 (`predict.py`), Task 15 (precompute), Task 16 (local pull) |
| §10 Failure modes | Documented inline in Tasks 12, 13, 14 |
| §11 Reproducibility | Task 2 (config.yaml), Task 7 (`set_seed`, config snapshot) |
| §12 Compute budget | Tasks 13, 14, 15 wall-clock+cost annotations |

**Placeholder scan:** No TBDs, no "implement later," no "similar to Task N." All code blocks are complete. All commands have explicit expected output.

**Type consistency:** `pad_left`, `shift_ids_plus_one`, `SASRecDataset`, `SASRec`, `build_logits`, `masked_ce_loss`, `sample_hard_negatives`, `sample_random_negatives`, `run_preprocessing`, `train_main`, `evaluate_pipeline_mode`, `evaluate_unconstrained`, `derived_metrics`, `predict_top_100`, `precompute_test_top100`, `save_checkpoint`, `load_checkpoint` — all named consistently across tasks. Modal function names (`cache_stage1_candidates`, `train`, `evaluate`, `precompute`) match between Task 10 (definition) and Tasks 12-16 (invocation).

**Scope:** Focused on Stage 2 only. Stage 3 (MMR) is out of scope and only appears as a handoff contract (Task 9).

No fixes needed.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-10-stage2-sasrec-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Use this if you want me to drive the implementation while you review each task before the next one starts.

**2. Inline Execution** — Execute tasks in this session sequentially with checkpoints. Use this if you'd rather watch and intervene in real time.

Which approach?
