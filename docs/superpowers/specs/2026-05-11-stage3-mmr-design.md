# Stage 3 — MMR Re-ranking (embeddings-only) — Design Spec

**Date**: 2026-05-11
**Status**: Approved, ready for implementation
**Scope**: Stage 3 of the music recommendation pipeline. Stage 4 (Streamlit UI) is out of scope.

---

## 1. Context

Stage 2 produced per-playlist top-100 candidates with relevance scores. Stage 3 trims these to a final top-20 that balances relevance with intra-list diversity. The original Stage 3 spec relied on per-track audio features (tempo, energy, valence, danceability) and artist/genre metadata, plus rule-based filters (artist cap, tempo continuity). **None of that data is available.** This spec drops all audio-rule-based filtering and replaces the audio-feature similarity with cosine similarity in Stage 2's fine-tuned 128-dim embedding space.

## 2. Inputs

| File | Source | Shape | Purpose |
|---|---|---|---|
| `stage_2/checkpoints/test_top100.npy` | Stage 2 (local) | `(150_000, 100) int32` | Top-100 candidate IDs per test playlist, in +1-shifted ID space |
| `stage_2/checkpoints/test_top100_scores.npy` | Stage 2 (local) | `(150_000, 100) float32` | Per-candidate relevance scores from Stage 2's transformer |
| `stage_2/checkpoints/best_item_embeddings.npy` | Stage 2 (Modal Volume) | `(2_262_293, 128) float32` | Fine-tuned item embeddings; row 0 is PAD (zero), rows [1:] are real items |

Pulling `best_item_embeddings.npy` from Modal Volume is a one-time `modal volume get` step (~1.1 GB).

## 3. Outputs

Written to `stage_3/checkpoints/` (gitignored):

| File | Shape / format | Purpose |
|---|---|---|
| `test_final20.npy` | `(150_000, 20) int32` | Final top-20 IDs per playlist (+1-shifted space) |
| `test_final20_scores.npy` | `(150_000, 20) float32` | MMR composite scores aligned with the IDs |
| `test_metrics.json` | dict | Per-λ recall@20 + avg intra-list cosine distance; recall@20 baseline from "Stage 2 raw top-20" for comparison |
| `lambda_sweep.json` | dict | Detailed per-λ metrics for the report (relevance/diversity tradeoff curve) |

## 4. Algorithm

Standard greedy MMR. For each test playlist:

```
candidates = test_top100[p]               # (100,) int32 IDs
rel        = minmax(test_top100_scores[p])  # (100,) ∈ [0, 1]
E          = item_embeddings[candidates]    # (100, 128) float32
S = []                                      # selected indices into candidates
while len(S) < K:
    available = [i for i in range(100) if i not in S]
    if not S:
        idx = argmax(rel over available)    # first pick: pure relevance
    else:
        sim_to_S = for each available i: max over s in S of cos(E[i], E[s])
        score    = λ * rel[i] - (1-λ) * sim_to_S[i]
        idx      = argmax(score over available)
    S.append(idx)
output_ids[p]    = candidates[S]
output_scores[p] = mmr_score_at_pick_time   # score the item achieved when it was picked
```

Notes:
- **K = 20** (final playlist size).
- **λ sweep**: run for λ ∈ {0.3, 0.5, 0.7}. Default reported configuration: **λ = 0.5** (balanced).
- **Cosine similarity** is computed against L2-normalized embedding rows. The embeddings are not normalized at save time; Stage 3 normalizes on load.
- **Relevance normalization**: min-max within each playlist's 100 candidates. Stage 2's raw scores are dot products with arbitrary scale; normalizing makes λ comparable across playlists.
- **First-pick convention**: when `S` is empty, the diversity term is undefined. Define `sim_to_S = 0` for all candidates (equivalently, just pick argmax(rel)).
- **Output scores**: store the MMR composite score at the moment of selection (not the raw relevance), so downstream consumers can see the tradeoff.

## 5. Vectorized implementation

Naive per-playlist loop = ~1 sec/playlist × 150K = 40 hours. Vectorize over playlist batches.

Batched algorithm (B playlists at a time, B = 1024):

```python
E_batch  = item_embeddings[test_top100[batch]]   # (B, 100, 128)
E_norm   = E_batch / norm(E_batch, axis=-1, keepdims=True)   # (B, 100, 128)
rel      = minmax_normalize(test_top100_scores[batch])       # (B, 100)
selected_mask = zeros((B, 100), bool)                         # tracks picks per playlist
out_ids       = empty((B, K), int32)
out_scores    = empty((B, K), float32)
running_max_sim = zeros((B, 100))    # max cosine sim to any already-selected, per playlist

for step in range(K):
    if step == 0:
        score = rel.clone()
    else:
        score = λ * rel - (1-λ) * running_max_sim
    score = score.masked_fill(selected_mask, -inf)
    pick = score.argmax(dim=-1)                      # (B,)
    out_ids[:, step]    = test_top100[batch][b, pick[b]] for b in batch
    out_scores[:, step] = score[b, pick[b]]
    selected_mask.scatter_(1, pick.unsqueeze(1), True)
    # Update running max sim: new sim row = cos(picked_emb, all_embs)
    picked_emb = E_norm[b, pick[b]] for b                       # (B, 128)
    new_sim    = einsum("bd,bnd->bn", picked_emb, E_norm)        # (B, 100)
    running_max_sim = max(running_max_sim, new_sim)
```

This runs entirely in NumPy (or PyTorch CPU). Expected wall-clock: <1 min for all 150K playlists.

## 6. Evaluation

Two metrics, both computed on the final 150K × 20 outputs:

1. **Recall@20** — fraction of test playlists where the held-out true next song is in the final 20.
   - Compute over the same 150K test playlists used in Stage 2 evaluation.
   - Compare against the **Stage 2 raw top-20** baseline (= take `test_top100[:, :20]`, no MMR). This tells us how much recall we sacrificed for diversity.

2. **Intra-list diversity (ILD)** — average pairwise cosine *distance* (1 − cos_sim) among the 20 selected items in each playlist, then averaged across all 150K playlists.
   - Higher is better (more diverse). Compare against the Stage 2 raw top-20 baseline for diversity gain.

For each λ ∈ {0.3, 0.5, 0.7}, report (recall@20, ILD). The trade-off curve is the report's headline figure.

**Ground-truth target source**: same as Stage 2 — the last song of each test playlist (`padded[850_000:, L-1]` in the shifted-ID space). The padded array is on Modal Volume; pulling it (~200 MB) is the only other download needed.

To avoid pulling 200 MB just for targets, we can extract them remotely (one Modal function call) and save a tiny `test_targets.npy` of shape `(150_000,) int32`. Or pull `playlists_padded.npy` once locally.

## 7. Code structure

```
stage_3/
├── __init__.py
├── mmr.py                  # Vectorized greedy MMR (one function)
├── evaluate.py             # recall@20, ILD, baseline comparison
├── run_stage3.py           # Main entry: load → sweep λ → save → eval → write JSON
├── tests/
│   ├── __init__.py
│   └── test_mmr.py         # Toy-data unit tests
└── checkpoints/            # gitignored outputs
```

## 8. Pre-implementation: pull Stage 2 embeddings + test targets

One-time data prep before any Stage 3 code runs:

```bash
modal volume get stage2-data runs/main/best/item_embeddings.npy \
    ./stage_2/checkpoints/best_item_embeddings.npy

# Either pull all padded playlists (200 MB) or just slice the targets:
modal volume get stage2-data derived/playlists_padded.npy \
    ./stage_2/checkpoints/playlists_padded.npy
```

Test targets are derived locally:
```python
padded = np.load("stage_2/checkpoints/playlists_padded.npy")
test_targets = padded[850_000:, -1]   # (150_000,) int32 in +1-shifted space
```

## 9. Reproducibility

- Pure deterministic algorithm; no randomness.
- λ sweep grid is fixed at {0.3, 0.5, 0.7} (configurable via CLI flag).
- One `config.yaml`-free design: hyperparameters (K, λ-grid, batch size) live as constants at the top of `run_stage3.py` and are echoed into `test_metrics.json` for reproducibility.

## 10. Compute / cost / time

- All compute is local Python (NumPy + maybe PyTorch CPU). No Modal needed.
- Expected wall-clock: ~1 min for 150K playlists × 3 λ values = ~3 min total.
- Cost: **$0**.

## 11. Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| MMR scores all -inf | All candidates already selected (impossible if K < 100) | Inspect `selected_mask`; should never trigger at K=20 |
| Recall@20 drops by >50% vs Stage 2 top-20 baseline | λ too low — over-prioritizing diversity | Increase λ closer to 1.0 |
| ILD ≈ 0 (no diversity) | λ too high or embeddings too clustered | Lower λ; or inspect embedding norms |
| `item_embeddings[0]` not zero | PAD row got polluted during Stage 2 fine-tuning | Verify with `np.allclose(emb[0], 0)`; if not, replace row 0 with zeros before MMR |
| Out-of-memory loading 1.1 GB embeddings | Laptop has <2 GB free RAM | Memory-map: `np.load(path, mmap_mode="r")` |

## 12. Out of scope

- Audio-feature-based rules (artist cap, tempo continuity, valence smoothing) — data unavailable.
- Tuning λ beyond the three-point sweep.
- Other diversity algorithms (DPP, clustering-based) — MMR is sufficient for this project.
- Stage 4 (Streamlit UI).
