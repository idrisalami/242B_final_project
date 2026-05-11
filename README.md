# Spotify recommendation system

Multi-stage music recommendation pipeline on the Spotify Million Playlist Dataset (MDP).

```
Playlist (N songs)
      ↓
Stage 1 · ALS candidate generation   → top 1,000 candidates   [DONE]
      ↓
Stage 2 · SASRec sequential ranking  → top 100                [DONE]
      ↓
Stage 3 · MMR diversity re-ranking   → final 20               [DONE]
      ↓
Stage 4 · Streamlit interface                                  [TODO]
```

---

## Repository layout

```
project/
├── stage_1/                      ALS candidate generation — COMPLETE
│   ├── checkpoints/              ← download from Google Drive
│   │   ├── als_item_factors.npy  (1.1 GB) item embeddings (2262292 × 128)
│   │   ├── uri_to_id.json        (105 MB) Spotify URI → integer ID
│   │   └── playlists.npy         (~200 MB) all 1M playlists as integer sequences
│   ├── data/
│   │   └── data_loader.py        loads MPD slices, builds integer IDs
│   ├── training/
│   │   ├── train_als.py          trains ALS (run this to retrain)
│   │   └── evaluate_als.py       evaluates on full 150K test set
│   ├── utils/helpers.py
│   ├── config/config.yaml
│   ├── failed_experiments/       Two-Tower neural + Track2Vec (abandoned)
│   │   ├── models/two_tower.py
│   │   ├── inference/
│   │   ├── evaluation/
│   │   └── training/             train.py, trainer.py, train_track2vec.py
│   └── STAGE_1_SUMMARY.md        what was tried and why ALS won
│
├── stage_2/                      SASRec ranking — NOT BUILT YET
│   └── README.md
│
├── stage_3/                      MMR re-ranking — NOT BUILT YET
│   └── README.md
│
├── stage_4/                      Streamlit interface — NOT BUILT YET
│   └── README.md
│
├── PIPELINE.md                   full input/output contracts for every stage
├── final_project.md              course project specification
└── README.md                     this file
```

---

## Stage 1 

**Method**: ALS matrix factorization 

| Metric | Result | Target |
|---|---|---|
| Recall@100 | **0.154** | > 0.15 |
| Recall@1000 | **0.42** | > 0.45 |

Training: 8 minutes on Apple M-chip CPU. Three approaches were tried — see `STAGE_1_SUMMARY.md` for the full story.

**Artifacts ready for Stage 2** (shared via Google Drive — no data download needed):
- `stage_1/checkpoints/als_item_factors.npy` — (2,262,292 × 128) float32 track embeddings
- `stage_1/checkpoints/uri_to_id.json` — maps `"spotify:track:abc"` → integer ID
- `stage_1/checkpoints/playlists.npy` — all 1M playlists as integer ID sequences (~200 MB)

**Quick inference from Stage 1:**
```python
import numpy as np, json

item_factors = np.load('stage_1/checkpoints/als_item_factors.npy')
uri_to_id    = json.load(open('stage_1/checkpoints/uri_to_id.json'))
id_to_uri    = {v: k for k, v in uri_to_id.items()}

def get_candidates(playlist_uris, k=1000):
    ids      = [uri_to_id[u] for u in playlist_uris if u in uri_to_id]
    user_emb = item_factors[ids].mean(axis=0)       # (128,)
    scores   = user_emb @ item_factors.T             # (2262292,)
    scores[ids] = -1e9                               # mask seen
    top_k    = np.argpartition(scores, -k)[-k:]
    return top_k[np.argsort(scores[top_k])[::-1]]   # sorted best-first
```

---

## What is missing

| Stage | Status | README |
|---|---|---|
| Stage 2 — SASRec ranking | DONE — pipeline R@100=0.236, NDCG@10=0.046 | [stage_2/README.md](stage_2/README.md) |
| Stage 3 — MMR re-ranking | DONE — λ=0.5: recall@20=0.114, ILD=0.433 (+18% diversity for −4% recall vs Stage 2 top-20) | [stage_3/README.md](stage_3/README.md) |
| Stage 4 — Streamlit interface | TODO | [stage_4/README.md](stage_4/README.md) |

Each stage is fully independent — see its README for what it needs and what it produces. Full input/output contracts: [PIPELINE.md](PIPELINE.md).

---

## Setup

```bash
# Install dependencies
uv sync        # or: pip install -r requirements.txt
```

---

## Getting the checkpoints

All three checkpoint files are shared via Google Drive. Download them and place them in `stage_1/checkpoints/`:

```
stage_1/checkpoints/
    als_item_factors.npy   ← from Drive
    uri_to_id.json         ← from Drive
    playlists.npy          ← from Drive
```

**That's all you need** to build Stages 2 and 3. No raw data download required.

Load the playlist sequences in one line:

```python
import numpy as np
playlists = np.load('stage_1/checkpoints/playlists.npy', allow_pickle=True).tolist()
# playlists: List[List[int]] — 1,000,000 playlists, each a list of integer track IDs
# Same 70/15/15 train/val/test split as Stage 1:
n = len(playlists)
train_playlists = playlists[:int(n * 0.70)]
val_playlists   = playlists[int(n * 0.70):int(n * 0.85)]
test_playlists  = playlists[int(n * 0.85):]
```

---

## Raw data (only needed to retrain Stage 1)

The raw Spotify Million Playlist Dataset is 31 GB uncompressed. You only need it if you want to retrain the ALS model from scratch.

### 1. Request access

Go to the AICrowd challenge page and accept the license:
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

### 2. Download

```bash
pip install aicrowd-cli
aicrowd login
aicrowd dataset download --challenge spotify-million-playlist-dataset-challenge
```

### 3. Place the files

```
stage_1/data/raw/data/
    mpd.slice.0-999.json
    ...
    mpd.slice.999000-999999.json
```

### 4. Retrain

```bash
cd stage_1
python training/train_als.py
# ~30 min on CPU. Overwrites checkpoints/als_item_factors.npy and playlists.npy
```
