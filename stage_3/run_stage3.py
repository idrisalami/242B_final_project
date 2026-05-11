"""Main driver for Stage 3 MMR re-ranking.

Loads Stage 2 outputs, sweeps λ over {0.3, 0.5, 0.7}, writes final
top-20 outputs + per-λ metrics, and computes the Stage 2 raw top-20
baseline for comparison.

See docs/superpowers/specs/2026-05-11-stage3-mmr-design.md.
"""

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from stage_3.evaluate import baseline_top_k, evaluate_run
from stage_3.mmr import l2_normalize_rows, mmr_full

# ── Hyperparameters (echoed into metrics for reproducibility) ─────────────────
K: int = 20
LAMBDA_GRID: List[float] = [0.3, 0.5, 0.7]
REPORTED_LAMBDA: float = 0.5
BATCH_SIZE: int = 1024

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE2_CKPT  = PROJECT_ROOT / "stage_2/checkpoints"
STAGE3_CKPT  = PROJECT_ROOT / "stage_3/checkpoints"

PATH_TOP100        = STAGE2_CKPT / "test_top100.npy"
PATH_TOP100_SCORES = STAGE2_CKPT / "test_top100_scores.npy"
PATH_EMBEDDINGS    = STAGE2_CKPT / "best_item_embeddings.npy"
PATH_PADDED        = STAGE2_CKPT / "playlists_padded.npy"


def load_test_targets(padded_path: Path) -> np.ndarray:
    """Held-out test targets = last token of the last 150K padded playlists.

    The test split is `padded[train_size + val_size :]` = `padded[850000:]`.
    The held-out target is the last column (position L-1) of the padded array.
    """
    padded = np.load(padded_path, mmap_mode="r")
    return padded[850_000:, -1].astype(np.int32)            # (150000,)


def main():
    STAGE3_CKPT.mkdir(parents=True, exist_ok=True)
    print("Loading Stage 2 outputs...")
    test_top100        = np.load(PATH_TOP100)                # (150000, 100) int32
    test_top100_scores = np.load(PATH_TOP100_SCORES)         # (150000, 100) float32
    item_embeddings    = np.load(PATH_EMBEDDINGS)            # (2262293, 128) float32
    targets            = load_test_targets(PATH_PADDED)      # (150000,) int32
    print(
        f"  test_top100={test_top100.shape}  scores={test_top100_scores.shape}  "
        f"embeddings={item_embeddings.shape}  targets={targets.shape}"
    )

    # Pre-normalize once for both MMR and ILD evaluation
    print("L2-normalizing embeddings...")
    item_emb_normed = l2_normalize_rows(item_embeddings)

    # ── Baseline: Stage 2 raw top-K ──────────────────────────────────────────
    print("\nBaseline: Stage 2 raw top-20")
    baseline_ids = baseline_top_k(test_top100, test_top100_scores, K=K)
    baseline_metrics = evaluate_run(baseline_ids, targets, item_emb_normed)
    print(
        f"  recall@{K}={baseline_metrics['recall_at_K']:.4f}  "
        f"ILD={baseline_metrics['intra_list_diversity']:.4f}"
    )

    # ── λ sweep ───────────────────────────────────────────────────────────────
    sweep = {}
    reported_run_ids = None
    reported_run_scores = None
    for lam in LAMBDA_GRID:
        print(f"\nMMR sweep λ={lam}")
        t0 = time.time()
        out_ids, out_scores = mmr_full(
            test_top100=test_top100,
            test_top100_scores=test_top100_scores,
            item_embeddings=item_embeddings,    # mmr_full will normalize internally — but we already have
            K=K,
            lam=lam,
            batch_size=BATCH_SIZE,
            progress=True,
        )
        wall = (time.time() - t0) / 60
        metrics = evaluate_run(out_ids, targets, item_emb_normed)
        metrics["wall_min"] = wall
        sweep[f"lambda_{lam}"] = metrics
        print(
            f"  recall@{K}={metrics['recall_at_K']:.4f}  "
            f"ILD={metrics['intra_list_diversity']:.4f}  wall={wall:.1f}m"
        )
        if lam == REPORTED_LAMBDA:
            reported_run_ids = out_ids
            reported_run_scores = out_scores

    # ── Save outputs for the reported λ ───────────────────────────────────────
    if reported_run_ids is None:
        raise RuntimeError(f"REPORTED_LAMBDA={REPORTED_LAMBDA} not in grid {LAMBDA_GRID}")
    print(f"\nSaving final outputs (λ={REPORTED_LAMBDA}) ...")
    np.save(STAGE3_CKPT / "test_final20.npy", reported_run_ids)
    np.save(STAGE3_CKPT / "test_final20_scores.npy", reported_run_scores)

    # ── Save metrics ──────────────────────────────────────────────────────────
    test_metrics = {
        "reported_lambda": REPORTED_LAMBDA,
        "K": K,
        "n_evaluated": int(test_top100.shape[0]),
        "baseline_stage2_top20": baseline_metrics,
        "mmr_reported_lambda": sweep[f"lambda_{REPORTED_LAMBDA}"],
    }
    with open(STAGE3_CKPT / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    lambda_sweep = {
        "K": K,
        "lambda_grid": LAMBDA_GRID,
        "baseline_stage2_top20": baseline_metrics,
        "sweep": sweep,
    }
    with open(STAGE3_CKPT / "lambda_sweep.json", "w") as f:
        json.dump(lambda_sweep, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'config':<15} {'recall@20':>10} {'ILD':>10}")
    print("-" * 60)
    print(
        f"{'Stage 2 top20':<15} "
        f"{baseline_metrics['recall_at_K']:>10.4f} "
        f"{baseline_metrics['intra_list_diversity']:>10.4f}"
    )
    for lam in LAMBDA_GRID:
        m = sweep[f"lambda_{lam}"]
        print(
            f"{'λ='+str(lam):<15} "
            f"{m['recall_at_K']:>10.4f} "
            f"{m['intra_list_diversity']:>10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
