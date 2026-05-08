"""
ALS-based candidate generation for Stage 1.
Replaces the Two-Tower neural model — trains in 30-60 min vs 47 hrs.

Usage (from stage_1/):
    pip install implicit
    python training/train_als.py

Output: checkpoints/als_item_factors.npy  — shape (vocab_size, 128)
        Same shape as the Two-Tower item embeddings, so Stage 2 is unchanged.

Typical results: Recall@1000 ~30-50% after 15 iterations.
"""

import sys
import json
import time
import numpy as np
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader_TwoTower
from utils.helpers import load_config, setup_logging

try:
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight
except ImportError:
    print("Run:  pip install implicit")
    sys.exit(1)

setup_logging(log_file="als_training.log")
cfg = load_config("config/config.yaml")

FACTORS        = cfg["model"]["embedding_dim"]   # 128 — matches Two-Tower
ITERATIONS     = 15      # 15 iterations → R@100=0.154, R@1000=0.42 (known good)
REGULARIZATION = 0.01
EVAL_BS        = 64
MAX_EVAL       = 12_800  # ~200 batches, enough for a stable estimate

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading dataset (~15 min for 1000 files)…")
raw_dir = Path(cfg["data"]["data_path"]) / "data"
dataset    = DataLoader_TwoTower.load_spotify_mpd_slices(
    str(raw_dir),
    max_files=cfg["data"].get("max_files"),
    min_playlist_length=cfg["data"]["min_playlist_length"],
)
playlists  = dataset["playlists"]
vocab_size = dataset["num_unique_tracks"]
uri_to_id  = dataset["uri_to_id"]
print(f"Playlists: {len(playlists):,}   Vocab: {vocab_size:,}")

# Same split as neural training
n          = len(playlists)
train_size = int(n * cfg["data"]["train_val_test_split"][0])
val_size   = int(n * cfg["data"]["train_val_test_split"][1])
train_playlists = playlists[:train_size]
test_playlists  = playlists[train_size + val_size:]
print(f"Train: {len(train_playlists):,}   Test: {len(test_playlists):,}")

# ── 2. Build sparse playlist × track matrix ────────────────────────────────────
print("\nBuilding sparse matrix…")
t0 = time.time()

rows, cols = [], []
for pid, pl in enumerate(train_playlists):
    # Use context only (all-but-last) — mirrors the test task
    context = pl[:-1] if len(pl) > 1 else pl
    for tid in context:
        rows.append(pid)
        cols.append(tid)

user_item = sp.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(len(train_playlists), vocab_size),
    dtype=np.float32,
)
print(f"Matrix: {user_item.shape}   nnz: {user_item.nnz:,}   ({time.time()-t0:.1f}s)")

# ── 3. Train ALS ───────────────────────────────────────────────────────────────
print(f"\nTraining ALS  factors={FACTORS}  iter={ITERATIONS}  reg={REGULARIZATION}")
model = AlternatingLeastSquares(
    factors=FACTORS,
    iterations=ITERATIONS,
    regularization=REGULARIZATION,
    use_gpu=False,
)
t0 = time.time()
# Binary weights × alpha=40 — outperformed BM25 in practice for this dataset
user_item_weighted = (user_item * 40).tocsr()
model.fit(user_item_weighted)
elapsed = time.time() - t0
print(f"Training done in {elapsed/60:.1f} min")

item_factors = np.array(model.item_factors, dtype=np.float32)   # (vocab_size, D)
item_factors_T = item_factors.T.copy()                           # (D, vocab_size) — contiguous
print(f"Item factors: {item_factors.shape}")

# ── 4. Evaluate on test split ──────────────────────────────────────────────────
print(f"\nEvaluating on up to {MAX_EVAL:,} test playlists…")
hits  = {100: 0, 1000: 0}
total = 0
t0    = time.time()

for batch_start in range(0, min(MAX_EVAL, len(test_playlists)), EVAL_BS):
    batch = test_playlists[batch_start : batch_start + EVAL_BS]

    ctx_embs, targets, contexts = [], [], []
    for pl in batch:
        if len(pl) < 2:
            continue
        context = pl[:-1]
        target  = pl[-1]
        # User embedding = mean of item factors for context songs
        ctx_embs.append(item_factors[context].mean(axis=0))
        targets.append(target)
        contexts.append(context)

    if not ctx_embs:
        continue

    ctx_embs = np.stack(ctx_embs)                    # (B, D)
    scores   = ctx_embs @ item_factors_T             # (B, vocab_size)

    # Mask songs already in the context
    for j, context in enumerate(contexts):
        scores[j, context] = -1e9

    # Top-1000 via argpartition (O(vocab) — much faster than argsort)
    top1000 = np.argpartition(scores, -1000, axis=1)[:, -1000:]   # (B, 1000)

    # Top-100: best 100 of the top-1000
    rows_idx = np.arange(len(ctx_embs))[:, None]
    top1000_scores = scores[rows_idx, top1000]                     # (B, 1000)
    top100_of_1000 = np.argpartition(top1000_scores, -100, axis=1)[:, -100:]
    top100  = top1000[rows_idx, top100_of_1000]                    # (B, 100)

    targets_arr = np.array(targets)[:, None]                       # (B, 1)
    hits[1000] += (top1000 == targets_arr).any(axis=1).sum()
    hits[100]  += (top100  == targets_arr).any(axis=1).sum()
    total      += len(ctx_embs)

    done = batch_start // EVAL_BS + 1
    if done % 50 == 0:
        print(f"  batch {done:>3} | "
              f"R@100={hits[100]/total:.4f}  R@1000={hits[1000]/total:.4f}")

print(f"\n{'='*50}")
print(f"Test playlists evaluated : {total:,}")
print(f"Recall@100  = {hits[100]  / total:.4f}  (target > 0.15)")
print(f"Recall@1000 = {hits[1000] / total:.4f}  (target > 0.80)")
print(f"Eval time   : {(time.time()-t0)/60:.1f} min")
print(f"{'='*50}")

# ── 5. Save ────────────────────────────────────────────────────────────────────
save_dir = Path(cfg["checkpoint"]["save_path"])
save_dir.mkdir(exist_ok=True, parents=True)

# Item embeddings — used by Stage 2 for scoring
factors_path = save_dir / "als_item_factors.npy"
np.save(factors_path, item_factors)

# URI → integer ID mapping — Stage 2 and the API must use the same IDs
mapping_path = save_dir / "uri_to_id.json"
with open(mapping_path, "w") as f:
    json.dump(uri_to_id, f)

print(f"\nSaved item factors : {factors_path}  {item_factors.shape}")
print(f"Saved URI mapping  : {mapping_path}  ({len(uri_to_id):,} entries)")
print("Load in Stage 2:")
print("  item_factors = np.load('checkpoints/als_item_factors.npy')")
print("  uri_to_id    = json.load(open('checkpoints/uri_to_id.json'))")
