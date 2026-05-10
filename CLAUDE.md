# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A 4-stage music recommender on the Spotify Million Playlist Dataset (MPD): given a partial playlist, predict the next songs.

```
Playlist → Stage 1 ALS (1000 candidates) → Stage 2 SASRec (top 100)
        → Stage 3 MMR diversity (top 20–30) → Stage 4 Streamlit UI
```

**Only Stage 1 is built.** Stages 2–4 are spec-only — see each `stage_N/README.md` and `PIPELINE.md` for the full input/output contracts. `final_project.md` is the course brief; `README.md` is the user-facing overview; `stage_1/STAGE_1_SUMMARY.md` documents the three approaches tried (Two-Tower neural and Track2Vec were abandoned in favor of ALS).

## Environment

```bash
uv sync                       # preferred (uv.lock pinned)
# or: pip install -r requirements.txt
```

Python ≥ 3.11. Key deps: `implicit` (ALS), `torch`, `numpy`, `scipy`, `pyyaml`.

## Common commands

All Stage 1 scripts assume you `cd stage_1/` first — they read `config/config.yaml` via a relative path and use `sys.path.insert(...)` to import sibling modules.

```bash
cd stage_1

python training/train_als.py        # train ALS, ~8 min CPU, writes checkpoints/als_item_factors.npy + uri_to_id.json
python training/evaluate_als.py     # evaluate saved factors on full 150K test set, ~25 min (includes ~15 min data load)
python training/save_playlists.py   # export integer-encoded playlists to checkpoints/playlists.npy for Stage 2 handoff
```

There is no test suite, linter, or CI in this repo. Don't invent one.

## Architecture — what a future Claude actually needs to know

### The shared integer ID space is the load-bearing invariant

Every stage operates on the same `int` track IDs assigned by `DataLoader_TwoTower.load_spotify_mpd_slices()` in `stage_1/data/data_loader.py`. IDs are assigned in order of first appearance across MPD slice files, so they are deterministic only if all 1000 slices are loaded in filename order. Vocab size is **2,262,292**.

The mapping is persisted as `stage_1/checkpoints/uri_to_id.json` and must be reused by every downstream stage. **Never re-derive IDs in a downstream stage.**

### Three checkpoint files are the public API between stages

```
stage_1/checkpoints/
  als_item_factors.npy  (1.1 GB) — (2262292, 128) float32 item embeddings
  uri_to_id.json        (105 MB) — Spotify URI → int ID
  playlists.npy         (~200 MB) — object array of List[int], all 1M playlists
```

These are **gitignored** (see `.gitignore`) and shared via Google Drive. Stage 2/3/4 work consumes these files and does **not** need the raw 31 GB MPD download. The raw data, if present, lives under `stage_1/data/raw/data/mpd.slice.*.json` (also gitignored).

### `stage_1/data/` is gitignored — `data_loader.py` is not in the repo

`train_als.py` and `evaluate_als.py` import from `data.data_loader`, but `stage_1/data/` is excluded by `.gitignore`. If you need to run training and the loader is missing, that's why — it lives on the original author's machine alongside the raw MPD. For Stage 2+ work, prefer loading `playlists.npy` directly instead of rebuilding from MPD.

### Data split convention — 70/15/15 by index, never shuffled

```python
n = len(playlists)
train = playlists[:int(n*0.70)]
val   = playlists[int(n*0.70):int(n*0.85)]
test  = playlists[int(n*0.85):]
```

Every stage must use this exact split, in this exact order, so test playlists stay unseen end-to-end. Configured in `stage_1/config/config.yaml` under `data.train_val_test_split`.

### Task framing: context = `playlist[:-1]`, target = `playlist[-1]`

All training and evaluation use the last song as the held-out target and everything before it as context. Mirror this in any new stage. Recall@K is computed by checking whether `playlist[-1]` is in the top-K scored items, with context songs masked to `-1e9` before top-K selection.

### Stage 1 inference recipe (already validated)

User embedding is the **mean** of context item embeddings, scored by dot product against all 2.26M items. This mean-pool is the fundamental bottleneck — it discards sequence order, which caps Recall@1000 around 0.50 regardless of training quality. Stage 2 (SASRec) exists specifically to break this ceiling with causal self-attention.

### Stage 1 results (current)

Recall@100 = 0.151 (target > 0.15 ✅), Recall@1000 = 0.407 (target > 0.45 ❌), NDCG@10 = 0.018. Measured on all 150K test playlists. See `stage_1/STAGE_1_SUMMARY.md` for the comparison against Two-Tower (R@1000 = 0.058) and Track2Vec (R@1000 = 0.403).

### `stage_1/failed_experiments/` is gitignored

Past dead-ends (Two-Tower MLP, Track2Vec). Don't resurrect these without reading `STAGE_1_SUMMARY.md` first — the conclusion was that mean-pool inference is the bottleneck, not the training method.

## When working on Stage 2+

The intended SASRec design (per `stage_2/README.md` and `PIPELINE.md`):
- Causal Transformer (1–2 layers, 2–4 heads, d=128, dropout 0.2, max seq len 50)
- Initialize item embeddings from `als_item_factors.npy` rather than from scratch
- Sampled-softmax / BPR loss against ~99 negatives — **never** score all 2.26M items per step
- Evaluate both unconstrained (full vocab) and pipeline-mode (only within Stage 1's 1000 candidates)

Stage 3 is rule-based MMR (no training); Stage 4 is a Streamlit front-end. See their respective READMEs.
