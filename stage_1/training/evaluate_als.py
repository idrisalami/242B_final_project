"""
Evaluate saved ALS item factors on the full test split.

Usage (from stage_1/):
    python training/evaluate_als.py

Evaluates all 150,000 test playlists (not just 12,800).
Runtime: ~10 min (15 min data load + eval).
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader_TwoTower
from utils.helpers import load_config

cfg = load_config("config/config.yaml")

FACTORS_PATH = Path(cfg["checkpoint"]["save_path"]) / "als_item_factors.npy"
EVAL_BS      = 256

# ── Load item factors ─────────────────────────────────────────────────────────
print(f"Loading item factors from {FACTORS_PATH}…")
item_factors   = np.load(FACTORS_PATH)          # (vocab_size, D)
item_factors_T = item_factors.T.copy()          # (D, vocab_size) contiguous
print(f"Item factors: {item_factors.shape}")

# ── Load data (same split as training) ───────────────────────────────────────
print("Loading dataset (~15 min)…")
raw_dir = Path(cfg["data"]["data_path"]) / "data"
dataset    = DataLoader_TwoTower.load_spotify_mpd_slices(
    str(raw_dir),
    max_files=cfg["data"].get("max_files"),
    min_playlist_length=cfg["data"]["min_playlist_length"],
)
playlists  = dataset["playlists"]
vocab_size = dataset["num_unique_tracks"]
print(f"Playlists: {len(playlists):,}   Vocab: {vocab_size:,}")

n          = len(playlists)
train_size = int(n * cfg["data"]["train_val_test_split"][0])
val_size   = int(n * cfg["data"]["train_val_test_split"][1])
test_playlists = playlists[train_size + val_size:]
print(f"Test playlists: {len(test_playlists):,}")

# ── Eval loop ─────────────────────────────────────────────────────────────────
hits  = {10: 0, 50: 0, 100: 0, 500: 0, 1000: 0}
ndcg  = {10: 0.0}
total = 0
t0    = time.time()

print(f"\nEvaluating all {len(test_playlists):,} test playlists…")
for batch_start in range(0, len(test_playlists), EVAL_BS):
    batch = test_playlists[batch_start : batch_start + EVAL_BS]

    ctx_embs, targets, contexts = [], [], []
    for pl in batch:
        if len(pl) < 2:
            continue
        ctx_embs.append(item_factors[pl[:-1]].mean(axis=0))
        targets.append(pl[-1])
        contexts.append(pl[:-1])

    if not ctx_embs:
        continue

    ctx_embs = np.stack(ctx_embs)           # (B, D)
    scores   = ctx_embs @ item_factors_T    # (B, vocab_size)

    for j, ctx in enumerate(contexts):
        scores[j, ctx] = -1e9

    top1000     = np.argpartition(scores, -1000, axis=1)[:, -1000:]
    rows_idx    = np.arange(len(ctx_embs))[:, None]
    top1000_sc  = scores[rows_idx, top1000]

    targets_arr = np.array(targets)[:, None]
    hits[1000] += (top1000 == targets_arr).any(axis=1).sum()

    for k in [10, 50, 100, 500]:
        top_k = top1000[rows_idx, np.argpartition(top1000_sc, -k, axis=1)[:, -k:]]
        hits[k] += (top_k == targets_arr).any(axis=1).sum()

    # NDCG@10: sort just the top-10 to find rank of the target
    B           = len(ctx_embs)
    top10_pos   = np.argpartition(top1000_sc, -10, axis=1)[:, -10:]       # (B, 10) indices into top1000
    top10_ids   = top1000[rows_idx, top10_pos]                             # (B, 10) track IDs
    top10_sc    = top1000_sc[np.arange(B)[:, None], top10_pos]            # (B, 10) scores
    sort10      = np.argsort(top10_sc, axis=1)[:, ::-1]                   # descending
    top10_sorted = top10_ids[np.arange(B)[:, None], sort10]               # (B, 10) sorted IDs
    match       = (top10_sorted == targets_arr)                            # (B, 10)
    found       = match.any(axis=1)                                        # (B,)
    ranks       = (np.argmax(match, axis=1) + 1).astype(float)            # 1-indexed
    ndcg[10]   += (found * (1.0 / np.log2(ranks + 1))).sum()

    total += len(ctx_embs)

    done = batch_start // EVAL_BS + 1
    total_batches = (len(test_playlists) + EVAL_BS - 1) // EVAL_BS
    if done % 50 == 0:
        print(f"  batch {done:>3}/{total_batches} | "
              f"R@10={hits[10]/total:.4f}  R@100={hits[100]/total:.4f}  "
              f"NDCG@10={ndcg[10]/total:.4f}")

elapsed = time.time() - t0
print(f"\n{'='*55}")
print(f"Test playlists : {total:,}  |  Eval time: {elapsed/60:.1f} min")
print(f"{'='*55}")
for k in [10, 50, 100, 500, 1000]:
    print(f"  Recall@{k:<5} = {hits[k]/total:.4f}")
print(f"  NDCG@10   = {ndcg[10]/total:.4f}")
print(f"{'='*55}")
