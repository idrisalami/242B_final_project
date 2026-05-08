# Stage 2 — Sequential Re-ranking (SASRec)

**Status: NOT BUILT — this is your task.**

---

## What This Stage Does

Takes the 1,000 candidates from Stage 1 and re-ranks them to a top 100 using the full ordered sequence of the playlist. Song order carries signal that Stage 1's mean-pooling discards — recent songs matter more than older ones.

---

## Input / Output

```
Input:
  playlist_ids  — full ordered context as integer IDs (same ID space as Stage 1)
  candidate_ids — top-1000 IDs returned by Stage 1

Output:
  ranked_ids    — top-100 IDs, sorted best-first
  ranked_scores — score for each
```

---

## What You Need (from Google Drive)

Place these in `stage_1/checkpoints/` before starting:

| File | Size | What it is |
|---|---|---|
| `als_item_factors.npy` | 1.1 GB | (2,262,292 × 128) track embeddings — use to initialize SASRec |
| `uri_to_id.json` | 105 MB | Spotify URI → integer ID mapping |
| `playlists.npy` | ~200 MB | All 1M playlists as integer sequences — load this instead of downloading raw data |

`playlists.npy` was produced by `stage_1/training/save_playlists.py`. Load it with:
```python
playlists = np.load('stage_1/checkpoints/playlists.npy', allow_pickle=True).tolist()
```

Use the same 70/15/15 split as Stage 1 (indices, not shuffled).

---

## Model — SASRec-Style Transformer

A causal self-attention transformer that reads the playlist sequence and outputs a query vector, which is then dot-producted against the 1,000 candidate embeddings to produce scores.

Key details:
- **Embedding dim**: 128 (matches ALS factors)
- **Layers**: 1–2, **Heads**: 2–4, **Dropout**: 0.2
- **Max sequence length**: 50 (truncate older songs, left-pad shorter ones)
- **Causal mask**: position `i` cannot attend to positions `j > i`
- **Initialize** `item_embedding` from `als_item_factors.npy` — strong head start, avoids learning from scratch

---

## Training

- **Data**: `playlists.npy` — each playlist's context is `playlist[:-1]`, target is `playlist[-1]`
- **Loss**: cross-entropy with in-batch negatives (1 positive + ~99 random negatives per step)
- **Do NOT** score against all 2.26M tracks per step — too expensive; sample negatives instead
- **Optimizer**: Adam, lr=1e-3, 5–10 epochs

---

## Evaluation

Evaluate the full pipeline: Stage 1 retrieves 1,000 candidates → Stage 2 re-ranks → check if true next song is in top-10 / top-100.

| Metric | Stage 1 (ALS) | Stage 2 target |
|---|---|---|
| Recall@10 | 0.035 | > 0.18 |
| Recall@100 | 0.151 | > 0.30 |
| NDCG@10 | 0.018 | > 0.15 |

---

## Artifacts to Produce

```
stage_2/checkpoints/
    sasrec_model.pt   — model weights (torch.save)
    config.yaml       — hyperparameters used
```

---

## Suggested File Structure

```
stage_2/
├── models/sasrec.py          SASRec transformer class
├── data/dataset.py           PyTorch Dataset (padding, ID shift for pad token)
├── training/train_sasrec.py  training entry point
├── training/evaluate.py      pipeline evaluation (Stage 1 + Stage 2 together)
└── checkpoints/
```
