# Stage 3 — Diversity Re-ranking (MMR, embeddings-only)

**Status: DONE.**

Greedy Maximal Marginal Relevance (MMR) over Stage 2's top-100 per playlist, producing a final top-20. The diversity metric is cosine similarity in Stage 2's fine-tuned 128-dim embedding space — no audio features needed. Pure NumPy, runs locally in ~25 seconds for the full λ sweep on all 150K test playlists. $0 Modal cost.

Full design rationale: [`../docs/superpowers/specs/2026-05-11-stage3-mmr-design.md`](../docs/superpowers/specs/2026-05-11-stage3-mmr-design.md)

---

## Results (150K test playlists, K=20)

| Config | Recall@20 | Intra-list diversity (ILD) | vs Stage 2 raw top-20 |
|---|---|---|---|
| Stage 2 raw top-20 (no MMR) | 0.1186 | 0.3682 | — (baseline) |
| **MMR λ=0.5 (reported)** | **0.1135** | **0.4327** | **−4.3% recall, +17.5% diversity** |
| MMR λ=0.7 (relevance-leaning) | 0.1173 | 0.3925 | −1.1% recall, +6.6% diversity |
| MMR λ=0.3 (diversity-leaning) | 0.0975 | 0.4958 | −17.8% recall, +34.7% diversity |

The λ knob behaves as designed: clean monotonic tradeoff between recall and diversity. `ILD` is the mean pairwise cosine *distance* (1 − cosine_similarity) among the final 20 songs, averaged over all 150K playlists. Higher ILD = more variety in the final playlist.

---

## What this stage does

Input: Stage 2's top-100 candidate songs per playlist + their relevance scores + Stage 2's fine-tuned item embeddings.

Output: a final top-20 per playlist that balances relevance with intra-list diversity.

Algorithm: greedy MMR, applied iteratively K=20 times per playlist:

```
score(candidate) = λ · relevance(candidate)
                 − (1 − λ) · max_similarity_to_already_selected(candidate)
```

- **First pick**: the candidate with highest relevance (no diversity term applies).
- **Subsequent picks**: maximize the score above. High relevance AND not too similar to any song already selected.
- **Similarity**: cosine in Stage 2's 128-dim embedding space.
- **λ ∈ [0, 1]**: the knob. λ=1 → pure Stage 2 ordering; λ=0 → pure diversity.

---

## Why no audio features

The original Stage 3 design (in `final_project.md`) used audio features (tempo, energy, valence, danceability) + artist/genre metadata to define similarity and to apply rule-based filters (artist cap, tempo continuity, energy continuity, valence smoothing). **None of that data was available in this project's scope** — accessing Spotify's Web API for 2.26M tracks was infeasible.

The clean reduction: replace audio-feature similarity with **embedding similarity in Stage 2's learned representation space**. Item embeddings encode co-occurrence and sequential structure, so songs that are close in embedding space tend to substitute for each other in playlists. That's exactly the property MMR wants from its similarity metric. The audio-rule-based filters were dropped entirely.

---

## File structure

```
stage_3/
├── mmr.py                  Vectorized greedy MMR + helpers (130 lines)
├── evaluate.py             Recall@K + intra-list diversity + Stage 2 baseline (70 lines)
├── run_stage3.py           Main driver: load → λ sweep → save → eval → write JSON (150 lines)
├── tests/test_mmr.py       10 unit tests (all green)
└── checkpoints/            Outputs (gitignored)
```

No training, no Modal, no GPU. Pure NumPy.

---

## How to run

**Prerequisites (one-time):** Stage 2 outputs must be locally available at `stage_2/checkpoints/`:

```bash
# Required
modal volume get stage2-data runs/main/test_top100.npy            ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_top100_scores.npy     ./stage_2/checkpoints/
# Required for the diversity metric (1.1 GB)
modal volume get stage2-data runs/main/best/item_embeddings.npy   ./stage_2/checkpoints/best_item_embeddings.npy
# Required to derive the held-out test targets (200 MB)
modal volume get stage2-data derived/playlists_padded.npy         ./stage_2/checkpoints/
```

**Run unit tests:**

```bash
uv run pytest stage_3/tests/ -v
```

10 tests, ~0.2 s.

**Run the full pipeline (λ sweep + save final top-20 for λ=0.5):**

```bash
PYTHONPATH=. uv run python stage_3/run_stage3.py
```

Wall-clock: ~25 sec for the full sweep on all 150K test playlists (3 λ values + baseline + eval). RAM peak: ~3 GB (loads the 1.1 GB embedding table once).

---

## Artifacts produced (in `stage_3/checkpoints/`, gitignored)

```
test_final20.npy             (150_000, 20) int32   — final per-playlist IDs (λ=0.5)
test_final20_scores.npy      (150_000, 20) float32 — MMR composite score at pick time
test_metrics.json            Baseline + reported-λ metrics
lambda_sweep.json            Full per-λ tradeoff (used for the report figure)
```

IDs are in the **+1-shifted space** (same as Stage 2's output). To convert back to Spotify URIs, subtract 1 and look up in `stage_1/checkpoints/uri_to_id.json`.

---

## Hyperparameters (constants at top of `run_stage3.py`)

| Parameter | Value | Why |
|---|---|---|
| `K` | 20 | Standard playlist length; matches the original spec's lower bound |
| `LAMBDA_GRID` | `[0.3, 0.5, 0.7]` | Three-point sweep gives a tradeoff curve for the report |
| `REPORTED_LAMBDA` | 0.5 | Balanced point; recall loss ~4%, diversity gain ~18% |
| `BATCH_SIZE` | 1024 | Vectorizes over playlists; 1024 fits in RAM comfortably |

---

## Vectorization

Naive per-playlist Python loop = ~1 sec/playlist × 150K = ~40 hours, infeasible. The actual implementation processes B=1024 playlists at a time:

1. Gather all candidate embeddings into a `(B, 100, 128)` tensor.
2. Maintain a `running_max_sim` of shape `(B, 100)` — the maximum cosine similarity of each candidate to *any* already-selected candidate, per playlist.
3. At each of K=20 steps:
   - Compute the MMR score: `λ · rel − (1-λ) · running_max_sim`.
   - Mask already-picked candidates to `-inf` (no duplicate selection).
   - `argmax` along the candidate axis → one pick per playlist.
   - Update `running_max_sim` with the new picks' similarities (one `einsum` per step).
4. Total work: O(B × K × C × D) per batch — fully vectorized, runs in NumPy without any GPU.

End-to-end: ~7 sec per λ value on a laptop.

---

## Evaluation

Two metrics computed on all 150K test playlists:

1. **Recall@20** — fraction of playlists where the held-out true next song appears in the final 20. Compared against the Stage 2 raw top-20 baseline (just take Stage 2's top-20 without re-ranking) to measure how much recall we sacrificed for diversity.

2. **Intra-list diversity (ILD)** — mean pairwise cosine *distance* among the final 20 songs (averaged across all playlists). Bounded in [0, 2]; 0 = all identical, 1 = orthogonal. Higher = more diverse.

The λ sweep gives the tradeoff curve. Pick λ based on whether you care more about recall (high λ) or playlist variety (low λ).

---

## Out of scope

- Audio-feature-based rules (artist cap, tempo continuity, valence smoothing) — data not available.
- Fine-grained λ tuning beyond the three-point sweep.
- Other diversity algorithms (DPP, clustering-based) — MMR is sufficient.
- Per-playlist λ adaptation (e.g. tighter λ for short context, broader for long).
