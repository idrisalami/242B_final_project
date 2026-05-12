# Stage 2 — Sequential Re-ranking (SASRec)

**Status: DONE.**

Causal-attention transformer that re-ranks Stage 1's top-1000 candidates per playlist down to a top-100, using the playlist's ordered sequence. Trained on Modal (free tier), ~67 min wall-clock, ~$3 total Modal spend.

Full design rationale: [`../docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md`](../docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md)
Implementation plan: [`../docs/superpowers/plans/2026-05-10-stage2-sasrec-implementation.md`](../docs/superpowers/plans/2026-05-10-stage2-sasrec-implementation.md)

---

## Results (150K test playlists)

| Metric | Pipeline-mode | Unconstrained (full 2.26M vocab) | Stage 1 baseline |
|---|---|---|---|
| R@10 | **0.082** | 0.022 | 0.035 |
| R@50 | **0.180** | 0.050 | — |
| R@100 | **0.236** | 0.070 | 0.151 |
| R@1000 | (= Stage 1's R@1000, locked) | 0.188 | 0.407 |
| NDCG@10 | **0.046** | 0.013 | 0.018 |

**Key finding:** pipeline R@100 (0.236) >> unconstrained R@100 (0.070). Stage 2 alone is *worse* at general retrieval than the full pipeline — Stage 1's candidate filtering is essential, Stage 2's value is the ranking refinement within those candidates. This is the cleanest possible justification for the two-stage architecture.

Versus Stage 1 alone: **+56% Recall@100, +156% NDCG@10**.

Training plateaued at epoch 6, early-stopped at epoch 14/15 (no val improvement for 6 epochs).

---

## What this stage does

Input: a playlist as a sequence of integer-encoded track IDs (in the +1-shifted space where ID 0 = PAD).

Output: a top-100 re-ranked subset of Stage 1's top-1000 candidates for that playlist, with relevance scores.

Algorithm: a small causal Transformer (2 layers, 2 heads, d=128) reads the playlist sequence, produces a query vector at the last non-pad position, and dot-products that query against item embeddings to score candidates.

---

## Architecture

- **Model**: SASRec-original (Kang & McAuley 2018), 2 layers / 2 heads / d=128 / FFN=256 / max_seq=50 / dropout 0.2. Pre-LayerNorm.
- **Embedding init**: Stage 1's ALS item factors loaded into the embedding table; row 0 reserved as PAD (zero).
- **Training task**: causal LM — at every position predict the next song. Maximally efficient use of signal: ~19M training positions per epoch from ~600K training playlists.
- **Loss**: softmax cross-entropy with target=0 (positive at index 0), negatives = 128 hard negatives from Stage 1 candidates + 128 random negatives. Masked to valid positions only.
- **Optimizer**: AdamW with two parameter groups — embedding table at LR 1e-4, transformer at LR 1e-3 (discriminative fine-tuning, 10× lower for the pre-initialized embeddings).
- **Schedule**: linear warmup over 1000 steps → cosine decay to 0.
- **Precision**: bf16 autocast on CUDA. Gradient clipping max-norm 1.0.
- **Eval task**: held-out last song of each playlist. Two modes — pipeline (within Stage 1's 1000) and unconstrained (full vocab).

---

## File structure

```
stage_2/
├── config.yaml                      Single source of hyperparameters
├── modal_app.py                     Modal entry points (cache / train / eval / infer)
├── models/sasrec.py                 SASRec model class
├── data/dataset.py                  PyTorch Dataset + padding helpers
├── training/preprocess.py           Pass 1 (pad playlists) + Pass 2 (Stage 1 candidate cache)
├── training/loss.py                 Negative pool sampling + masked CE loss
├── training/train.py                Training loop (early stopping, checkpointing)
├── training/evaluate.py             Pipeline + unconstrained eval metrics
├── inference/predict.py             Single-playlist inference + bulk precompute_test_top100
├── tests/                           20 unit tests (test_sasrec/test_dataset/test_loss/test_preprocess)
├── tests/tier2_dry_run.py           Local synthetic-data smoke test
└── checkpoints/                     Pulled artifacts (gitignored)
```

---

## How to reproduce on Modal

**Prerequisites (one-time):**
- Modal account + `modal token new` set up
- AICrowd account with the MPD challenge license accepted
- `aicrowd-key` Modal Secret containing `AICROWD_API_KEY=<your-key>`

**Step 1 — Stage 1 retrain** (regenerates `als_item_factors.npy`, `playlists.npy`, `uri_to_id.json`):

```bash
modal run stage_1/modal_app.py
```

Downloads MPD (~5.8 GB), extracts it, trains ALS (15 iter, factors=128). ~20-30 min wall-clock, ~$0.10 on Modal CPU.

**Step 2 — Stage 2 candidate cache:**

```bash
modal run stage_2/modal_app.py --cmd cache
```

Pass 1 + Pass 2: pad all 1M playlists, then GPU-batched matmul to compute Stage 1's top-1000 candidates per playlist. ~5 min, ~$0.10.

**Step 3 — Pre-launch validation (optional but recommended):**

```bash
pytest stage_2/tests/                                                 # Tier 1: 20 unit tests
PYTHONPATH=. python stage_2/tests/tier2_dry_run.py                    # Tier 2: local synthetic
modal run stage_2/modal_app.py --cmd train --run-id smoke --smoke     # Tier 3: 1-epoch Modal smoke
```

If any tier fails, do not launch the full run.

**Step 4 — Full training (15 epochs causal LM, A10G):**

```bash
modal run --detach stage_2/modal_app.py --cmd train --run-id main
```

~67 min wall-clock (was ~$2-3 on A10G). Watches val NDCG@10 and stops early if no improvement for 6 epochs.

**Step 5 — Test eval + precompute Stage 3 handoff file:**

```bash
modal run stage_2/modal_app.py --cmd eval --run-id main --mode both
modal run stage_2/modal_app.py --cmd infer --run-id main
```

**Step 6 — Pull artifacts locally for Stage 3:**

```bash
modal volume get stage2-data runs/main/test_top100.npy        ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_top100_scores.npy ./stage_2/checkpoints/
modal volume get stage2-data runs/main/best/model.pt          ./stage_2/checkpoints/best_model.pt
modal volume get stage2-data runs/main/train_history.json     ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_metrics.json      ./stage_2/checkpoints/
```

Optional (only if you want to do local Stage 2 inference or run Stage 3 locally):

```bash
modal volume get stage2-data runs/main/best/item_embeddings.npy \
    ./stage_2/checkpoints/best_item_embeddings.npy
```

---

## Artifacts produced (on Modal Volume `stage2-data`)

```
/vol/runs/main/
├── best/                              Best checkpoint by val NDCG@10
│   ├── model.pt                       Transformer + positional embeddings (~2 MB)
│   └── item_embeddings.npy            Fine-tuned item embedding table (1.1 GB)
├── final/                             Last-epoch checkpoint (same shape as best/)
├── train_history.json                 Per-epoch loss/val metrics + wall-clock
├── test_metrics.json                  Pipeline + unconstrained + derived metrics (final)
├── test_top100.npy                    (150_000, 100) int32 — Stage 3 input
├── test_top100_scores.npy             (150_000, 100) float32 — Stage 3 input
├── config.yaml                        Snapshot of hyperparameters used
└── run_metadata.json                  git SHA + timestamps (not currently emitted)
```

---

## ID convention (load-bearing)

All Stage 2 outputs use **+1-shifted track IDs**: ID 0 is reserved as PAD, original Stage 1 IDs are shifted by +1. So Stage 1's track ID `42` is Stage 2's ID `43`. Stage 3 consumes these shifted IDs directly. To map back to Spotify URIs, subtract 1 and look up in `uri_to_id.json`.

---

## What didn't work / known limitations

- Spec targets were aspirational: R@100 > 0.30 and NDCG@10 > 0.15. Actual 0.236 and 0.046. Honest interpretation: the 2-layer / 2-head architecture saturates quickly on this dataset; train loss kept dropping but val plateaued at epoch 6. Larger architectures (4 layers, 8 heads — Option B in the design spec) would likely close the gap but were skipped to stay within the user's compute budget.
- Two bugs caught during execution:
  - Modal container resolves `__file__` to `/root/modal_app.py`, not `/repo/stage_2/modal_app.py`, so the module-level config load needs a path fallback. Fixed in commit `afa6dcc`.
  - With left-padding, position 0 is PAD; causal attention restricts its keys to position 0 itself; `key_padding_mask=True` then yields all-masked softmax → NaN that propagates via residuals. Fixed by dropping `key_padding_mask` in the attention call (commit `aaa3cc5`).
