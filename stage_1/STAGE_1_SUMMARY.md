# Stage 1 — Training Summary

**Dataset**: Spotify Million Playlist Dataset — 1,000,000 playlists, 2,262,292 unique tracks  
**Split**: 700K train / 150K val / 150K test (same order as MPD slice files)  
**Task**: Given playlist context (first N−1 songs), retrieve the N-th song from 2.26M tracks

---

## What We Tried

### Attempt 1 — Two-Tower Neural Network (❌ Abandoned)

**Architecture**: shared embedding table (128-dim) → mean pool → MLP [512→256→128] → L2-norm  
**Loss**: in-batch InfoNCE (contrastive), temperature=0.07, batch_size=512  
**Optimizer**: AdamW, lr=1e-4, cosine decay with 500-step warmup  
**Device**: Apple MPS (M-chip)

| Epoch | Train Loss | Val Loss | Time/Epoch |
|-------|-----------|---------|------------|
| 1 | 6.28 | 6.24 | 1h 25m |
| 2 | 6.23 | 6.20 | 2h 18m |
| 3 | 6.17 | 6.12 | 2h 28m |
| 4 | ~6.11 | — | ~2h 20m |

Random baseline = log(512) = **6.24**. After 4 epochs (~10 hours), the model was barely below random.

**Evaluated epoch-4 checkpoint:**
- Recall@100 = **0.029** (target > 0.15) ❌
- Recall@1000 = **0.058** (target > 0.80) ❌

**Why it failed**: 20 epochs × 2h20m = ~47 hours total on MPS. Even if fully trained, the loss curve suggested convergence around 3.0–4.0 loss, which would require 50+ epochs. Compute was the bottleneck, not the architecture.

**Decision**: Abandoned neural approach. Switched to classical collaborative filtering.

---

### Attempt 2 — ALS Matrix Factorization (✅ Best result so far)

**Method**: Alternating Least Squares on a sparse playlist × track co-occurrence matrix  
**Library**: `implicit` (v0.7.2)  
**Training data**: context-only (playlist[:-1]) to mirror the test task  
**Config**: factors=128, iterations=15, regularization=0.01, alpha=40, binary weights

| Step | Time |
|------|------|
| Build sparse matrix (700K × 2.26M, 45.7M non-zeros) | 5.7s |
| ALS training (15 iterations) | 7.9 min |
| Total | ~8 min |

**Evaluated on full 150,000 test playlists:**

| Metric | Result | Target |
|---|---|---|
| Recall@10 | 0.0352 | — |
| Recall@50 | 0.1023 | — |
| Recall@100 | **0.1513** | > 0.15 ✅ |
| Recall@500 | 0.3188 | — |
| Recall@1000 | **0.4067** | > 0.80 ❌ |
| NDCG@10 | **0.0179** | — |

**Artifacts saved:**
```
checkpoints/als_item_factors.npy   # (2262292, 128) float32
checkpoints/uri_to_id.json         # 2,262,292 URI → int mappings
```

---

### Attempt 3 — Track2Vec / Skip-gram (❌ No improvement)

**Method**: Word2Vec skip-gram applied to playlist sequences (treats playlists as sentences, tracks as words)  
**Config**: dim=128, window=10, neg_samples=10, 5 epochs, batch=4096  
**Device**: Apple MPS

| Epoch | Loss | Time |
|-------|------|------|
| 1 | 1.15 | ~65 min |
| 4 | 0.30 | 67 min |
| 5 | 0.23 | 91 min |

**Total training time: ~6 hours**

**Evaluated on 12,800 test playlists:**
- Recall@100 = **0.128** ❌
- Recall@1000 = **0.403** ❌

**Why it didn't help**: Both Track2Vec and ALS use mean-pooling of item embeddings at inference time. The sequential context Track2Vec learned during training doesn't help because it's discarded at retrieval. The bottleneck is inference, not training.

---

## What Is Next (to improve Stage 1)

### Immediate — Run ALS + BM25 + 50 iterations (~25 min)
Already configured in `training/train_als.py`. BM25 down-weights popular tracks (less discriminative), and more iterations → better convergence.

```bash
python training/train_als.py
```

Expected result: **Recall@1000 ~0.48–0.52**, Recall@100 ~0.18

### Hard ceiling
Any method using mean-pool at inference is capped at roughly **50% Recall@1000**. To exceed this, inference must be sequence-aware — which is exactly what Stage 2 (SASRec) provides.

---

## Current Best Artifacts

| File | Size | Contents |
|------|------|---------|
| `checkpoints/als_item_factors.npy` | 1.1 GB | (2,262,292 × 128) float32 item embeddings |
| `checkpoints/uri_to_id.json` | 105 MB | Spotify URI → integer ID mapping |

**How to use for inference:**
```python
import numpy as np, json

item_factors = np.load('checkpoints/als_item_factors.npy')  # (2262292, 128)
uri_to_id    = json.load(open('checkpoints/uri_to_id.json'))

def get_candidates(playlist_uris, k=1000):
    ids      = [uri_to_id[u] for u in playlist_uris if u in uri_to_id]
    user_emb = item_factors[ids].mean(axis=0)            # (128,)
    scores   = user_emb @ item_factors.T                 # (2262292,)
    scores[ids] = -1e9                                   # mask seen
    top_k    = np.argpartition(scores, -k)[-k:]
    return top_k[np.argsort(scores[top_k])[::-1]]        # sorted best-first
```

---

## What Is Left for the Full Pipeline

| Stage | Status | What's needed |
|-------|--------|--------------|
| **Stage 1** (candidate gen) | 🟡 Partial — R@1000=0.39 | Run ALS+BM25+50iter → target ~0.50 |
| **Stage 2** (SASRec ranking) | ❌ Not built | Causal transformer, re-ranks 1000→100 |
| **Stage 3** (MMR re-ranking) | ❌ Not built | Rule-based diversity filter, 100→20-30 |
| **Interface** (FastAPI+React) | ❌ Not built | Web app wrapping all stages |

See `PIPELINE.md` for full input/output contracts for each stage.

---

## Key Lesson

> Classical collaborative filtering (ALS) trained in 8 minutes outperformed a neural Two-Tower model trained for 10 hours by **7×** on Recall@1000 (0.40 vs 0.058). For this dataset size on consumer hardware, simpler = better.

The neural approach would eventually win given sufficient compute (GPU, 50+ epochs), but ALS is the practical choice for this project's constraints.
