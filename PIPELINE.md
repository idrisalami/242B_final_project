# Music Recommendation Pipeline — Handoff Document

Full pipeline: **Spotify Million Playlist Dataset → 20–30 recommended songs**

---

## Overview

```
Playlist (N songs)
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1 · ALS Matrix Factorization │  2,262,292 tracks → top 1,000 candidates
│  STATUS: ✅ Complete                 │  Recall@100=0.15  Recall@1000=0.42
└─────────────────────────────────────┘
      │  top-1000 song IDs + scores
      ▼
┌─────────────────────────────────────┐
│  Stage 2 · SASRec Ranking           │  1,000 candidates → top 100
│  STATUS: ❌ Not built               │
└─────────────────────────────────────┘
      │  top-100 song IDs + scores
      ▼
┌─────────────────────────────────────┐
│  Stage 3 · MMR + Rules              │  100 → final 20–30
│  STATUS: ❌ Not built               │
└─────────────────────────────────────┘
      │  final playlist
      ▼
┌─────────────────────────────────────┐
│  Interface · FastAPI + React        │
│  STATUS: ❌ Not built               │
└─────────────────────────────────────┘
```

---

## Stage 1 — Candidate Generation (COMPLETE)

### What it does
Fast brute-force retrieval. Scans all 2.26M tracks and returns the 1,000 most
likely next songs for a given playlist. Optimised for recall, not precision.

### Approach — ALS (Alternating Least Squares)
We factorize the playlist × track co-occurrence matrix using ALS from the
`implicit` library. This produces a 128-dimensional embedding for every track.
At inference, the user embedding is the mean of item embeddings for context songs,
then scored against all 2.26M item embeddings.

> **Why not a neural model?** A Two-Tower MLP was attempted first (InfoNCE
> contrastive loss, in-batch negatives, AdamW, cosine-decay LR). After 4 epochs
> (~10 hours on Apple MPS) the model achieved only Recall@1000 = 0.058 —
> essentially random. ALS achieved Recall@1000 = 0.42 in ~30 minutes.

### Input
```python
playlist: List[int]   # sequence of song IDs (integer-encoded)
                      # IDs are the internal integer mapping produced by
                      # load_spotify_mpd_slices() — NOT Spotify URIs
```

### Output
```python
candidate_ids:    np.ndarray  # shape (1000,) — top-1000 song IDs, sorted best-first
candidate_scores: np.ndarray  # shape (1000,) — dot-product scores (higher = better)
```

### Performance (test set — 150,000 held-out playlists, never seen during training)
| Metric | Result | Target |
|---|---|---|
| Recall@10 | 0.035 | — |
| Recall@100 | **0.151** ✅ | > 0.15 |
| Recall@1000 | **0.407** | > 0.80 |
| NDCG@10 | 0.018 | — |

Recall@1000 is below the 80% target but well above random (0.004%). Stage 2
can still demonstrate meaningful re-ranking within these 40% coverage cases.
Improving Stage 1 recall further (e.g. more ALS iterations, larger factors)
would directly raise the ceiling for Stage 2.

### Training details
- **Dataset**: Spotify MPD — 1M playlists, 2.26M unique tracks
- **Split**: 70% train / 15% val / 15% test (indices, not shuffled)
- **Matrix**: sparse `(700K playlists) × (2.26M tracks)`, binary co-occurrence
- **Algorithm**: ALS, factors=128, iterations=15, regularization=0.01, alpha=40
- **Runtime**: ~30 min on Apple M-chip CPU

### Artifacts produced
```
stage_1/checkpoints/
    als_item_factors.npy      # shape (2262292, 128), float32 — item embeddings
```

### How to call it (Python API)
```python
import numpy as np

# Load once at startup
item_factors = np.load('stage_1/checkpoints/als_item_factors.npy')
# item_factors.shape == (2262292, 128)

def get_candidates(playlist_ids: list[int], k: int = 1000) -> tuple:
    """
    playlist_ids: integer song IDs (from the training-time URI mapping)
    returns: (candidate_ids, scores) — both np.ndarray of shape (k,)
    """
    # User embedding = mean of item factors for context songs
    user_emb = item_factors[playlist_ids].mean(axis=0)   # (128,)

    # Score all 2.26M items
    scores = user_emb @ item_factors.T                    # (2262292,)

    # Mask songs already in the playlist
    scores[playlist_ids] = -1e9

    # Top-k (unsorted)
    top_k_idx = np.argpartition(scores, -k)[-k:]

    # Sort top-k by score descending
    top_k_sorted = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    return top_k_sorted, scores[top_k_sorted]

# Example
candidate_ids, candidate_scores = get_candidates([1042, 837, 291, 4405], k=1000)
# candidate_ids.shape  == (1000,)
# candidate_scores.shape == (1000,)
```

### Important: ID mapping
Track URIs (`spotify:track:abc123`) are mapped to integers by the data loader.
The mapping is built by `load_spotify_mpd_slices()` in order of first appearance
across the 1,000 MPD slice files. **Stage 2 must use the same integer IDs.**

Save the mapping from the data loader and reuse it everywhere:
```python
# In the data loader (stage_1/data/data_loader.py), after building uri_to_id:
import json
json.dump(uri_to_id, open('stage_1/checkpoints/uri_to_id.json', 'w'))

# In Stage 2 / interface:
uri_to_id = json.load(open('stage_1/checkpoints/uri_to_id.json'))
id_to_uri  = {v: k for k, v in uri_to_id.items()}
```

> **Note**: `uri_to_id.json` is not yet saved automatically — add this save
> step to `load_spotify_mpd_slices()` before integrating Stage 2.

---

## Stage 2 — Ranking (NOT BUILT)

### What it does
Re-ranks the 1,000 candidates from Stage 1 to a tighter top-100, using the full
ordered sequence of the playlist (not just mean-pooling). Song *order* carries
information: recent songs matter more than older ones.

### Input
```python
playlist:      List[int]   # full ordered playlist context (same integer IDs as Stage 1)
candidate_ids: List[int]   # the 1,000 IDs returned by Stage 1
```

### Output
```python
ranked_ids:    List[int]    # top-100 song IDs, sorted best-first
ranked_scores: List[float]  # score for each
```

### Suggested model — SASRec style
- Causal Transformer encoder (masked self-attention, no peeking at future)
- Input: sequence of item embeddings + positional encoding
- Output at last position: a 128-dim query vector
- Score each of the 1,000 candidates: `query · item_emb`
- Item embeddings: **can be initialised from Stage 1's ALS factors** (`als_item_factors.npy`)
  and fine-tuned, or learned from scratch
- Architecture ref: [SASRec paper](https://arxiv.org/abs/1808.09781)

```
playlist [s1, s2, ..., sN]
      ↓  ALS item embeddings (128d) + positional encoding
Transformer (2 layers, 4 heads, d=128, causal mask)
      ↓  output at position N  →  query (128d)
dot product with each of the 1,000 candidate ALS embeddings
      ↓
scores (1000,)  →  top-100
```

### Training
- **Same dataset**: MPD 1M playlists, same 70/15/15 train/val/test split
- **Loss**: in-batch InfoNCE (same as Stage 1 attempted), or BPR against the 1,000 Stage-1 candidates
- **Key difference from Stage 1**: causal attention preserves song order

### Evaluation (same test set)
| Metric | Target |
|---|---|
| Recall@100 | > 30% — must improve on Stage 1's 15% |
| NDCG@100 | > 0.35 — ranking quality within the 100 |

Evaluate two ways:
1. **Unconstrained** — score against full 2.26M vocab (upper bound on model quality)
2. **Pipeline** — score only within Stage 1's 1,000 candidates (real-world performance)

### Artifacts to produce
```
stage_2/checkpoints/
    final_model.pt
    config.yaml
```

---

## Stage 3 — Re-ranking / MMR (NOT BUILT)

### What it does
Takes the top-100 from Stage 2 and selects a final playlist of 20–30 songs that
is both relevant *and* diverse — no artist spam, smooth audio transitions, varied
genres. Fully rule-based, no training required.

### Input
```python
candidate_ids:    List[int]    # top-100 IDs from Stage 2
candidate_scores: List[float]  # relevance scores from Stage 2
audio_features:   Dict[int, Dict]  # per-song audio features (see below)
```

### Audio features needed (from Spotify API or pre-fetched)
```python
{
    song_id: {
        "tempo":        float,   # BPM
        "energy":       float,   # 0–1
        "valence":      float,   # 0–1
        "danceability": float,   # 0–1
        "artist_id":    str,
        "genre":        str      # optional
    }
}
```

### Output
```python
final_ids:    List[int]    # 20–30 song IDs, ordered for listening
final_scores: List[float]  # final MMR scores
```

### Algorithm — Maximal Marginal Relevance (MMR)
At each step pick the next song that maximises:
```
MMR = λ · relevance_score  −  (1 − λ) · max_similarity_to_already_selected
```
where `similarity_to_already_selected` = Euclidean distance in
tempo/energy/valence/danceability space.

### Rules applied on top of MMR
| Rule | Implementation |
|---|---|
| Artist cap | No artist appears more than 2× in the final 20–30 |
| Tempo continuity | Consecutive songs differ by < 20 BPM |
| Energy continuity | Consecutive songs differ by < 0.3 in energy |
| Valence smoothing | Avoid abrupt mood swings (valence diff < 0.4) |

Tune `λ` and rule thresholds empirically on the val set.

### Evaluation
- **Diversity score**: average pairwise audio-feature distance across final songs
- **Artist coverage**: number of unique artists in the final playlist
- **Transition smoothness**: average consecutive tempo/energy delta

---

## Interface (NOT BUILT)

### FastAPI backend
```
POST /recommend
Body: { "playlist": ["spotify:track:uri1", "spotify:track:uri2", ...] }

Response: {
    "recommendations": [
        { "track_uri": str, "score": float },
        ...
    ]
}
```

Internally:
1. Map Spotify URIs → integer IDs (load `uri_to_id.json` from Stage 1)
2. Stage 1 → 1,000 candidates (numpy, ~10 ms)
3. Stage 2 → top-100 (PyTorch, ~50 ms)
4. Stage 3 → final 20–30 (pure Python, ~5 ms)
5. Map integer IDs back → Spotify URIs
6. Return

### React frontend
- Playlist input: search/add songs by name
- Recommendations panel: display final 20–30 songs with preview player

---

## Shared Data Contract

All stages use the **same integer ID space** built by the Stage 1 data loader.

```
stage_1/checkpoints/
    als_item_factors.npy    # (2262292, 128) float32 — item embeddings
    uri_to_id.json          # {"spotify:track:abc": 0, ...}  ← save this!
```

The vocab size is **2,262,292** unique tracks across all 1,000 MPD slice files.

---

## Summary Table

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| **Input** | playlist (N songs) | playlist + 1,000 candidates | 100 candidates + audio features |
| **Output** | top-1,000 IDs + scores | top-100 IDs + scores | 20–30 IDs + scores |
| **Model** | ALS matrix factorization | SASRec Transformer | MMR + rules (no model) |
| **Trained** | ✅ Yes | ✅ Yes (needs building) | ❌ No |
| **Train data** | MPD 1M playlists | MPD 1M playlists (same split) | — |
| **Key metric** | Recall@1000 | Recall@100 | Diversity + smoothness |
| **Result** | R@100=0.151 / R@1000=0.407 / NDCG@10=0.018 | — | — |
| **Status** | ✅ Complete | ❌ Not built | ❌ Not built |
