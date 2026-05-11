"""Tier 2: Run a tiny synthetic training pass locally to validate the loop.

Synthesizes 1000 playlists × random Stage 1 candidates, runs 3 mini-epochs at
batch=32, max_seq=20. Pass if train loss decreases.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from stage_2.training.train import train_main


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tier2_"))
    print(f"Tier 2 sandbox: {tmp}")

    # ── Synthesize data ──────────────────────────────────────────────────────
    np.random.seed(0)
    vocab = 200            # tiny vocab
    N = 1000               # 1000 playlists
    L = 20
    train_n = 700
    val_n = 150

    # Random lengths in [10, 20] for train, [5, 20] for val/test
    lengths = np.zeros(N, dtype=np.int16)
    lengths[:train_n] = np.random.randint(10, 21, size=train_n)
    lengths[train_n:] = np.random.randint(5, 21, size=N - train_n)

    padded = np.zeros((N, L), dtype=np.int32)
    for i in range(N):
        seq_ids = np.random.randint(1, vocab, size=int(lengths[i]))
        padded[i, L - int(lengths[i]):] = seq_ids

    # Synthetic Stage 1 candidates — must be >= 100 because evaluate_pipeline_mode
    # calls torch.topk(scores, k=100, dim=1) unconditionally.
    candidates = np.random.randint(1, vocab, size=(N, 100), dtype=np.int32)

    # Save into the sandbox
    np.save(tmp / "playlists_padded.npy", padded)
    np.save(tmp / "playlists_lengths.npy", lengths)
    np.save(tmp / "candidates.npy", candidates)

    # Synthetic ALS factors
    als = np.random.randn(vocab - 1, 32).astype(np.float32)
    np.save(tmp / "als_item_factors.npy", als)

    cfg = {
        "model": {
            "d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_dim": 64,
            "max_seq_len": L, "dropout": 0.0, "vocab_size": vocab,
        },
        "data": {
            "min_train_length": 10,
            "train_size": train_n, "val_size": val_n, "test_size": N - train_n - val_n,
            "val_eval_subset": 100,
        },
        "training": {
            "batch_size": 32, "epochs": 3, "warmup_steps": 10,
            "peak_lr_transformer": 1e-3, "peak_lr_embedding": 1e-4,
            "weight_decay": 0.01, "grad_clip": 1.0, "precision": "fp32",
            "seed": 42,
            "num_hard_negs": 16, "num_random_negs": 16,
            "val_every_n_epochs": 1, "early_stopping_patience_evals": 100,
            "keep_best_and_last_only": True,
        },
        "eval": {"candidates_k": 100, "ranked_top_k": 32,
                 "metrics_recall_k": [10, 50, 100], "ndcg_k": 10},
        "modal": {"gpu": "a10g", "volume_name": "ignore", "app_name": "ignore",
                  "function_timeout_sec": 1},
        "paths": {
            "vol_root": str(tmp),
            "als_factors": str(tmp / "als_item_factors.npy"),
            "uri_to_id": "",
            "playlists_raw": "",
            "playlists_padded": str(tmp / "playlists_padded.npy"),
            "playlists_lengths": str(tmp / "playlists_lengths.npy"),
            "candidates": str(tmp / "candidates.npy"),
            "runs_dir": str(tmp / "runs"),
        },
    }

    out = train_main(cfg, run_id="tier2", smoke=False)
    print(f"Result: best_ndcg={out['best_ndcg']}")
    print(f"History entries: {len(out['history'])}")

    # Pass criteria
    history = out["history"]
    assert len(history) >= 2, "Need at least 2 val evals to compare"
    losses = [h["train_loss"] for h in history]
    print(f"Train losses: {losses}")
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    recalls = [h["val"]["R@100"] for h in history]
    print(f"Val R@100 over epochs: {recalls}")
    assert max(recalls) > 0.02, f"Val R@100 never above 0.02 (best={max(recalls)})"

    print("TIER 2 PASS")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
