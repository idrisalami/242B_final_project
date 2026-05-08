# Stage 3 — Diversity Re-ranking (MMR)

**Status: NOT BUILT — this is your task.**

No training required. This is a rule-based algorithm.

---

## What This Stage Does

Takes the top-100 from Stage 2 and selects a final playlist of **20–30 songs** that is both relevant *and* diverse — no artist spam, smooth audio transitions, varied mood.

---

## Input

```python
candidate_ids:    List[int]    # top-100 IDs from Stage 2 (integer IDs)
candidate_scores: List[float]  # relevance scores from Stage 2 (higher = better)
audio_features:   Dict[int, Dict]  # per-song audio features (see below)
```

### Audio features format

```python
{
    song_id: {
        "tempo":        float,   # BPM (e.g. 120.0)
        "energy":       float,   # 0–1
        "valence":      float,   # 0–1 (mood: sad → happy)
        "danceability": float,   # 0–1
        "artist_id":    str,     # Spotify artist URI
    }
}
```

These can be fetched from the Spotify Web API (requires credentials) or pre-computed offline.
If the Spotify API is unavailable, use random uniform features as a stub — the algorithm logic still demonstrates correctly.

---

## Output

```python
final_ids:    List[int]    # 20–30 song IDs, ordered for listening
final_scores: List[float]  # final MMR scores
```

---

## Algorithm — Maximal Marginal Relevance

At each step pick the next song that maximises:

```
MMR(s) = λ · relevance(s)  −  (1 − λ) · max_sim(s, already_selected)
```

where `max_sim` is the maximum cosine similarity in the audio feature space between candidate `s` and any already-selected song.

```python
import numpy as np

def mmr_select(candidate_ids, scores, audio_features,
               n=25, lam=0.7):
    """
    candidate_ids: List[int] of length 100
    scores:        np.ndarray of length 100 (higher = better, already normalized)
    audio_features: Dict[int, Dict]
    returns: List[int] of length n
    """
    # Build feature matrix (100, 4)
    keys = ['tempo', 'energy', 'valence', 'danceability']
    feat = np.array([[audio_features[i].get(k, 0.5) for k in keys]
                     for i in candidate_ids], dtype=np.float32)
    feat /= feat.max(axis=0) + 1e-8   # normalize each feature to [0,1]

    rel = np.array(scores, dtype=np.float32)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-8)  # normalize to [0,1]

    selected = []
    remaining = list(range(len(candidate_ids)))

    for _ in range(n):
        if not remaining:
            break

        if not selected:
            # First pick: highest relevance
            pick = max(remaining, key=lambda i: rel[i])
        else:
            sel_feat = feat[selected]   # (k, 4)
            best_score = -np.inf
            pick = remaining[0]
            for i in remaining:
                sim = np.max(
                    np.dot(sel_feat, feat[i]) /
                    (np.linalg.norm(sel_feat, axis=1) * np.linalg.norm(feat[i]) + 1e-8)
                )
                s = lam * rel[i] - (1 - lam) * sim
                if s > best_score:
                    best_score, pick = s, i

        selected.append(pick)
        remaining.remove(pick)

    return [candidate_ids[i] for i in selected]
```

---

## Hard Rules (applied after MMR)

These rules override MMR if violated. Apply as a post-filter:

| Rule | Implementation |
|---|---|
| Artist cap | Remove duplicates so no artist appears more than **2×** in final 20–30 |
| Tempo continuity | Reorder: consecutive songs differ by < 20 BPM |
| Energy continuity | Reorder: consecutive songs differ by < 0.3 in energy |

```python
def apply_rules(song_ids, audio_features, max_per_artist=2):
    artist_count = {}
    filtered = []
    for sid in song_ids:
        artist = audio_features[sid].get('artist_id', sid)
        if artist_count.get(artist, 0) < max_per_artist:
            artist_count[artist] = artist_count.get(artist, 0) + 1
            filtered.append(sid)
    return filtered
```

---

## Evaluation

There is no single ground-truth for diversity — evaluate with:

| Metric | What it measures |
|---|---|
| Artist coverage | # unique artists in final 20–30 |
| Avg pairwise audio distance | diversity across tempo/energy/valence/danceability |
| Avg consecutive tempo delta | transition smoothness |
| Avg consecutive energy delta | transition smoothness |

```python
def diversity_score(song_ids, audio_features):
    keys = ['tempo', 'energy', 'valence', 'danceability']
    feats = np.array([[audio_features[s].get(k, 0.5) for k in keys]
                      for s in song_ids])
    n = len(feats)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(feats[i] - feats[j])
    return total / (n * (n - 1) / 2)
```

---

## Suggested File Structure

```
stage_3/
├── README.md          (this file)
├── mmr.py             MMR selection algorithm
├── rules.py           hard rule post-filters
├── evaluate_stage3.py diversity metrics
└── demo.py            end-to-end demo with stub audio features
```

---

## Integration with Stages 1 and 2

```python
# Full pipeline
import numpy as np, json

item_factors = np.load('../stage_1/checkpoints/als_item_factors.npy')
uri_to_id    = json.load(open('../stage_1/checkpoints/uri_to_id.json'))
id_to_uri    = {v: k for k, v in uri_to_id.items()}

# Stage 1
candidate_ids = stage1_get_candidates(playlist_ids, k=1000)

# Stage 2
ranked_ids, ranked_scores = stage2_rerank(playlist_ids, candidate_ids, k=100)

# Stage 3
final_ids = mmr_select(ranked_ids, ranked_scores, audio_features, n=25)
final_ids = apply_rules(final_ids, audio_features)

# Convert back to URIs
final_uris = [id_to_uri[i] for i in final_ids]
```

---

## Tuning λ

- λ = 1.0 → pure relevance (no diversity)
- λ = 0.0 → pure diversity (ignores relevance)
- λ = 0.7 is a reasonable default; tune on the validation set
