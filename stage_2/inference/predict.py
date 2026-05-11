"""Stage 2 inference.

- `predict_top_100`: single-playlist API for Stage 3 / Stage 4 UI.
- `precompute_test_top100`: bulk-precompute Stage 2 top-100 for every test playlist.

See docs/superpowers/specs/2026-05-10-stage2-sasrec-design.md §9 (Stage 3 handoff).
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from stage_2.data.dataset import pad_left
from stage_2.models.sasrec import SASRec


def _pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)


def predict_top_100(
    model: SASRec,
    playlist_shifted_ids: list,           # length-N list in +1-shifted space
    stage1_candidates: np.ndarray,        # (1000,) int32 — Stage 1's top-1000 for this playlist
    device: str = "cuda",
    max_seq_len: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Stage 2 on a single playlist. Returns (top_ids, top_scores) both shape (100,)."""
    model.eval()
    padded = np.array(pad_left(playlist_shifted_ids, L=max_seq_len), dtype=np.int64)
    seq = torch.from_numpy(padded).unsqueeze(0).to(device)        # (1, L)

    with torch.no_grad():
        out = model(seq)                                          # (1, L, D)
        query = out[0, -1, :]                                     # (D,) — last position
        cand_t = torch.from_numpy(stage1_candidates.astype(np.int64)).to(device)  # (1000,)
        cand_emb = model.item_emb(cand_t)                         # (1000, D)
        scores = cand_emb @ query                                 # (1000,)
        # Mask any candidate that's already in the context
        ctx = set(int(x) for x in playlist_shifted_ids if x != 0)
        if ctx:
            mask = torch.tensor([int(c) in ctx for c in stage1_candidates],
                                device=device)
            scores[mask] = float("-inf")
        scores[cand_t == 0] = float("-inf")

        top_scores, top_idx = torch.topk(scores, k=100)
        top_ids = cand_t[top_idx].cpu().numpy().astype(np.int32)
        top_scores = top_scores.cpu().numpy().astype(np.float32)

    return top_ids, top_scores


def precompute_test_top100(
    model: SASRec,
    padded: np.ndarray,           # (N, L) int32 — test set padded sequences
    lengths: np.ndarray,          # (N,)   int16
    candidates: np.ndarray,       # (N, 1000) int32 — Stage 1 candidates for the test set
    out_dir: Path,
    batch_size: int = 256,
    device: str = "cuda",
    use_bf16: bool = False,
) -> None:
    """Run Stage 2 on every test playlist, save top-100 IDs + scores."""
    model.eval()
    N, L = padded.shape
    top_ids_out = np.zeros((N, 100), dtype=np.int32)
    top_scores_out = np.zeros((N, 100), dtype=np.float32)

    with torch.no_grad():
        for start in _pbar(range(0, N, batch_size), desc="inferring test top100"):
            end = min(start + batch_size, N)
            B = end - start
            seqs_np = padded[start:end].copy()
            lens = lengths[start:end]
            cands_np = candidates[start:end]

            # As in evaluate: zero out the last real token so the query at L-2 predicts it
            seqs = seqs_np.copy()
            valid = lens >= 2
            for j in range(B):
                if valid[j]:
                    seqs[j, L - 1] = 0

            seq_t = torch.from_numpy(seqs.astype(np.int64)).to(device)
            cand_t = torch.from_numpy(cands_np.astype(np.int64)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(seq_t)
                queries = out[:, L - 2, :]                             # (B, D)
                cand_emb = model.item_emb(cand_t)                      # (B, 1000, D)
                scores = (queries.unsqueeze(1) * cand_emb).sum(-1)     # (B, 1000)

                # Mask context tokens and PAD candidates
                for j in range(B):
                    if not valid[j]:
                        continue
                    real_seq = seqs_np[j][seqs_np[j] != 0]
                    ctx = real_seq[:-1]
                    if ctx.size == 0:
                        continue
                    ctx_t = torch.from_numpy(ctx.astype(np.int64)).to(device)
                    isin = (cand_t[j].unsqueeze(0) == ctx_t.unsqueeze(1)).any(0)
                    scores[j, isin] = float("-inf")
                scores[cand_t == 0] = float("-inf")

            # Sort full top-100 (we need scores sorted, not just argpartition)
            top_scores, top_idx = torch.topk(scores, k=100, dim=1)     # (B, 100)
            top_ids = torch.gather(cand_t, 1, top_idx)                  # (B, 100)

            top_ids_out[start:end] = top_ids.cpu().numpy().astype(np.int32)
            top_scores_out[start:end] = top_scores.cpu().float().numpy().astype(np.float32)

    np.save(out_dir / "test_top100.npy", top_ids_out)
    np.save(out_dir / "test_top100_scores.npy", top_scores_out)
    print(f"Wrote test_top100.npy {top_ids_out.shape} and test_top100_scores.npy {top_scores_out.shape}")
