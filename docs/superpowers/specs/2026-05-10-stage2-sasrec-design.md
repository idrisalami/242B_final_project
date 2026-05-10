# Stage 2 — SASRec Sequential Re-ranker — Design Spec

**Date**: 2026-05-10
**Status**: Approved, ready for implementation planning
**Scope**: Stage 2 of the music recommendation pipeline. Stage 3 (MMR) and Stage 4 (Streamlit) are out of scope.

---

## 1. Context and goals

### 1.1 Where Stage 2 fits

```
Playlist
   ↓
Stage 1 · ALS  (R@1000 = 0.407)  →  top 1000 candidates per playlist
   ↓
Stage 2 · SASRec (this spec)     →  top 100, re-ranked using sequence order
   ↓
Stage 3 · MMR                    →  top 20–30
```

Stage 1 mean-pools item embeddings to retrieve candidates, which discards sequence order — this caps Recall@1000 at ~50% in principle and 0.407 in practice. Stage 2 re-ranks Stage 1's 1000 candidates using a causal Transformer that reads the full ordered playlist.

### 1.2 Primary goals

- Train a causal-LM SASRec model that re-ranks Stage 1's 1000 candidates to a top-100 with strong pipeline-mode Recall@100 and NDCG@10.
- Stay within Modal's free tier ($30/month compute credit, ~50 GB Volume).
- Produce artifacts that let Stage 3 development proceed without re-running Stage 2 inference.

### 1.3 Non-goals

- Beating SOTA on MPD. Course project, not a publication.
- Improving Stage 1. Out of scope; Stage 1 is treated as a fixed black box.
- Hyperparameter search beyond the defaults locked in §6.
- Multiple ablation runs. Compute budget supports one main training run only.

### 1.4 Success criteria

| | Target | Note |
|---|---|---|
| Pipeline R@100 | > 0.30 | PIPELINE.md target. Reachable; ceiling is 0.407. |
| Pipeline NDCG@10 | > 0.15 | Stage 1's NDCG@10 is 0.018, so this is a ~8× improvement. |
| Unconstrained R@100 | reported, no hard target | Diagnostic: shows model quality independent of Stage 1's ceiling. |
| Project wall-clock | < 6 hours total | Including data prep, smoke run, full training, and final eval. |
| Modal spend | < $10 | Leaves headroom in the $30 monthly credit. |

---

## 2. Data contracts

### 2.1 Inputs (existing, from Stage 1)

Downloaded from Google Drive into `stage_1/checkpoints/`:

| File | Size | Shape / format |
|---|---|---|
| `als_item_factors.npy` | 1.1 GB | `(2_262_292, 128)` float32 — ALS item embeddings |
| `uri_to_id.json` | 105 MB | `dict[str, int]` — Spotify URI → integer ID, IDs in `[0, 2_262_292)` |
| `playlists.npy` | 200 MB | object array `(1_000_000,)` of `list[int]` — playlists as integer sequences |

### 2.2 Outputs (this spec produces)

Written to Modal Volume `stage2-data` under `/vol/runs/main/`. The small ones get pulled to `stage_2/checkpoints/` locally for downstream stages:

| File | Size | Shape / format | Consumer |
|---|---|---|---|
| `best_model.pt` | ~2 MB | PyTorch state_dict — transformer only (no embedding table) | Future runs, Stage 4 UI |
| `best_item_embeddings.npy` | 1.1 GB | `(2_262_293, 128)` float32 — fine-tuned item embeddings (ID 0 is PAD) | Stage 3 MMR similarity |
| `final_model.pt`, `final_item_embeddings.npy` | ~1.1 GB | Last-epoch checkpoint | Resume / comparison |
| `test_top100.npy` | ~60 MB | `(150_000, 100)` int32 — Stage 2's top-100 for every test playlist, IDs in shifted +1 space | **Stage 3 primary input** |
| `test_top100_scores.npy` | ~60 MB | `(150_000, 100)` float32 — relevance scores aligned with `test_top100.npy` | Stage 3 MMR (λ × relevance term) |
| `train_history.json` | <1 MB | Per-epoch metrics (train loss, val R@K, NDCG@10, wall-clock) | Report figures |
| `test_metrics.json` | <1 KB | Final test R@10/R@50/R@100/R@1000/NDCG@10 in both pipeline and unconstrained mode + derived numbers (§7.4) | Report tables |
| `config.yaml` | <1 KB | All hyperparameters used for this run | Reproducibility |
| `run_metadata.json` | <1 KB | `git rev-parse HEAD`, start/end timestamps, Modal function IDs, total cost | Reproducibility |

### 2.3 ID space convention

Vocab size is `2_262_293 = 2_262_292 + 1` because **ID 0 is reserved as PAD**. All track IDs from `uri_to_id.json` are shifted **+1** at load time. The ALS embedding table is loaded into rows `[1:]`; row `[0]` is a learnable PAD embedding initialized to zero.

Stage 3 MUST operate in this +1-shifted space (or shift back to Stage 1 IDs by subtracting 1). The IDs written to `test_top100.npy` are in the shifted space.

---

## 3. Model architecture

SASRec-original (Kang & McAuley, 2018), small configuration.

```
Input:  (B, L=50) int32 item IDs, left-padded with PAD=0
              ↓
Item embedding lookup (vocab=2_262_293, d=128)   ← init from als_item_factors.npy
        + learned positional embedding (max_pos=50, d=128)
              ↓
Pre-LayerNorm + Dropout(0.2)
              ↓
2× Transformer block:
   ├─ Causal multi-head self-attention (heads=2, d_head=64, d_model=128, dropout=0.2)
   └─ Pre-LN FFN (d=128 → 256 → 128, GELU, dropout=0.2)
              ↓
Final LayerNorm
              ↓
Output: (B, L, 128) — query vector at every position
```

### 3.1 Component rationale

| Choice | Reason |
|---|---|
| Causal mask | Training task is "predict next song from prefix" — future positions must not leak. |
| Pre-LN | More stable than SASRec-original's post-LN; no downside at this scale. |
| Learned positional embeddings | Faithful to SASRec; sufficient for max_seq=50. |
| Dropout 0.2 | SASRec default. |
| Left padding | Lets us read the query from position `[:, -1, :]` regardless of playlist length. |
| Embedding init from ALS | Strong head start — see Stage 1 summary, ALS factors already encode co-occurrence structure. |
| Fine-tune embeddings (not freeze) | Allows transformer to reshape embeddings for sequence task. Frozen would cap performance. |
| Discriminative fine-tuning (10× lower embedding LR) | Standard recipe to avoid destroying the ALS init during early training. |

### 3.2 Parameter count

| Component | Params |
|---|---|
| Item embedding table (2_262_293 × 128) | ~290M |
| Positional embedding (50 × 128) | 6.4K |
| 2 Transformer blocks (attention + FFN + LN) | ~470K |
| **Total trainable** | **~290M** (98% is the embedding table) |

### 3.3 Scoring

**Training**: at every position `i`, query `q_i = transformer_output[i]` is dot-producted against the embeddings of `[positive_i, hard_negs..., random_negs...]`. Softmax CE over this set, with the positive at index 0.

**Inference (pipeline mode)**: query at the last non-pad position is dot-producted against the embeddings of the 1000 Stage 1 candidates for this playlist. Top-100 by score.

**Inference (unconstrained)**: query at the last non-pad position is dot-producted against all 2.26M item embeddings. Top-K from full vocab.

---

## 4. Data pipeline

Two preprocessing passes, both cached on the Modal Volume. Idempotent.

### 4.1 Pass 1 — Padded playlists

Runs once on Modal CPU (~30 seconds). Input: `playlists.npy`. Output: `playlists_padded.npy`, `playlists_lengths.npy`.

```python
playlists = np.load("/vol/inputs/playlists.npy", allow_pickle=True).tolist()

# Shift IDs by +1 (PAD=0 reserved)
playlists = [[t + 1 for t in p] for p in playlists]

# Truncate to max_seq=50, keeping the most recent 50 songs (recent context matters more)
# Left-pad short playlists with 0
def pad(p, L=50):
    p = p[-L:]                       # keep last L
    return [0] * (L - len(p)) + p

padded  = np.array([pad(p) for p in playlists], dtype=np.int32)   # (1_000_000, 50)
lengths = np.array([min(len(p), 50) for p in playlists], dtype=np.int16)  # (1_000_000,)
np.save("/vol/derived/playlists_padded.npy", padded)
np.save("/vol/derived/playlists_lengths.npy", lengths)
```

### 4.2 Pass 2 — Stage 1 candidate cache

Runs once on Modal A10G (~5 minutes). Input: `als_item_factors.npy` + `playlists_padded.npy`. Output: `candidates.npy` shape `(1_000_000, 1000)` int32 (~4 GB), IDs in the +1-shifted space.

First, build a shifted ALS table so everything operates in the same ID space:

```python
als_raw = np.load("/vol/inputs/als_item_factors.npy")                              # (2_262_292, 128)
als = np.vstack([np.zeros((1, 128), dtype=np.float32), als_raw])                   # (2_262_293, 128)
# als[0] = zeros (PAD), als[1..2_262_293) = original ALS factors (shifted)
```

Then for each playlist `p_padded` (length-50 left-padded array, IDs already +1-shifted from Pass 1), recover the non-padded prefix and compute Stage 1's top-1000 over `prefix[:-1]`:

```python
prefix = p_padded[p_padded != 0]                # drop padding
ctx    = prefix[:-1]                            # all but the last real song

user_emb = als[ctx].mean(axis=0)                # (128,)
scores   = user_emb @ als.T                     # (2_262_293,)
scores[ctx] = -1e9                              # mask seen
scores[0]   = -1e9                              # mask PAD
top1000  = np.argpartition(scores, -1000)[-1000:]   # IDs in [1, 2_262_293)
```

Batched at 512 playlists per step on GPU (the GEMM `user_embs @ als.T` is one matmul per batch; the per-playlist seen-item masking is the only per-row work). All 1M playlists processed (train + val + test) since val/test need candidates for pipeline-mode eval.

**Note on the prefix choice**: Stage 1 sees `prefix[:-1]`, i.e., everything except the last real song. This matches what Stage 1 will see at eval time (when the last song is held out as the target). The same cached candidates serve both as training-time negatives and as eval-time candidate sets.

### 4.3 Split

Same indices as Stage 1, applied to the padded array. **Filter is train-only:**

```python
train = padded[:700_000]
train = train[lengths[:700_000] >= 10]   # filter: training only, len ≥ 10 (Section 4.4)
val   = padded[700_000:850_000]          # unchanged from Stage 1 — keeps metrics comparable
test  = padded[850_000:1_000_000]        # unchanged from Stage 1
```

### 4.4 Why filter only train

Causal LM exposes the model to short prefixes anyway (position 0 sees 1 song, position 1 sees 2). Filtering training to len ≥ 10 keeps signal high without forcing us to re-eval Stage 1 on a different test set. Val/test remain unchanged so headline metrics are directly comparable to Stage 1's published R@100=0.151, R@1000=0.407.

After filter: ~600K training playlists (~19M causal-LM positions per epoch).

### 4.5 Modal Volume layout

```
/vol/stage2-data/
├── inputs/
│   ├── als_item_factors.npy        (1.1 GB)
│   ├── uri_to_id.json              (105 MB)
│   └── playlists.npy               (200 MB)
├── derived/
│   ├── playlists_padded.npy        (200 MB)
│   ├── playlists_lengths.npy       (2 MB)
│   └── candidates.npy              (4 GB)
└── runs/
    ├── smoke/                      (smoke run artifacts; ephemeral)
    └── main/                       (full run artifacts; persistent)
```

Total Volume usage: ~7 GB persistent + ~3 GB ephemeral. Well within 50 GB allowance.

---

## 5. Training

### 5.1 One training step

Batch of B=256 playlists. **At training time, `seq` is the full padded playlist** (not `playlist[:-1]`). Every internal position contributes a training signal. At eval time `seq` is `playlist[:-1]` and only the last position's query is used — see §6.

Per playlist:

- `seq`: `(50,)` int32, full padded playlist, left-padded with PAD=0
- `positives`: `(50,)` int32 — at each position `i`, the next-song target (`seq` shifted by 1; last position's target is undefined and masked out)
- `positive_mask`: `(50,)` bool — True where this position has both a real current token and a real next token (not padding, not the very last real position)
- `candidates_p`: `(1000,)` int32 — this playlist's Stage 1 candidates from `candidates.npy`

**Forward pass:**

```python
embed   = item_emb(seq) + pos_emb(arange(50))                    # (B, 50, 128)
queries = transformer(embed, causal_mask, key_padding_mask)      # (B, 50, 128)
```

**Negative pool construction (per position, see §5.2):**

```python
# hard negs: 128 sampled from this playlist's Stage 1 candidates, shared across positions
hard_negs   = sample(candidates_p, k=128)                         # (B, 128)
# random negs: 128 uniform from [1, vocab), shared across the entire batch
random_negs = uniform(1, vocab_size, size=(128,))                 # (128,)

# Concat with positive at index 0
all_items = concat([positives[:, :, None],
                    hard_negs[:, None, :].expand(-1, 50, -1),
                    random_negs[None, None, :].expand(B, 50, -1)],
                   dim=-1)                                        # (B, 50, 257)
```

**Loss:**

```python
item_vecs = item_emb(all_items)                                   # (B, 50, 257, 128)
logits    = (queries[:, :, None, :] * item_vecs).sum(-1)          # (B, 50, 257)
target    = zeros((B, 50), dtype=long)                            # positive is always at index 0
loss      = cross_entropy(logits, target, reduction="none")       # (B, 50)
loss      = (loss * positive_mask).sum() / positive_mask.sum()    # masked mean
```

### 5.2 Why this negative recipe (option B from brainstorming)

| Component | Why |
|---|---|
| 128 hard negs from Stage 1 | Pipeline alignment — train and inference see the same candidate distribution. |
| 128 random negs from full vocab | Unbiased signal; prevents the model from over-specializing to the candidate set's shape. Standard hard-negative-mining recipe. |
| Hard negs shared across positions in a playlist | Stage 1 candidates are playlist-level, not position-level. Approximation is small and saves compute. |
| Random negs shared across the batch | Per-step waste is negligible. Simpler implementation than per-position random sampling. |
| Positive at index 0 | Lets one CE call replace a binary-classification loop. |

### 5.3 Hyperparameters

| | Value | Note |
|---|---|---|
| Optimizer | AdamW(β=(0.9, 0.98), eps=1e-8) | Standard transformer setup |
| Weight decay | 0.01 | Not applied to bias, LayerNorm, embeddings (standard exclusion) |
| Peak LR — transformer | 1e-3 | SASRec / GPT-style |
| Peak LR — embedding table | 1e-4 | Discriminative fine-tuning: 10× lower than transformer |
| LR schedule | Linear warmup over 1000 steps → cosine decay to 0 | |
| Batch size | 256 | Fits A10G 24 GB comfortably (~3 GB peak activations at bf16) |
| Gradient clipping | max norm 1.0 | |
| Mixed precision | bf16 autocast | 2× speedup vs fp32, more stable than fp16 |
| Epochs | 15 (with early stopping) | |
| Seed | 42 | Set on torch, numpy, random; `torch.use_deterministic_algorithms(True)` |

### 5.4 Validation cadence and early stopping

- Every 2 epochs, run pipeline-mode eval on a fixed 10K-playlist val subset (`val[:10_000]`). Compute R@10, R@100, NDCG@10.
- Track best checkpoint by **val pipeline-mode NDCG@10**.
- **Early stopping**: if val NDCG@10 doesn't improve for 3 consecutive eval checkpoints (= 6 epochs), stop training.
- Per val eval: ~2 minutes on A10G.

### 5.5 Checkpoint policy

Each "checkpoint" is a directory `runs/<id>/ckpt_epoch_{N}/` containing two files (matching the final-artifact split from §2.2):

- `model.pt` — `state_dict()` of the transformer + positional embedding (~2 MB)
- `item_embeddings.npy` — the fine-tuned item embedding table (`(2_262_293, 128)` float32, ~1.1 GB)

**Optimizer state is NOT saved per-checkpoint.** Training fits in one ~2.5 hr Modal function call (< 4 hr timeout), so we never resume mid-run. Skipping optimizer state saves ~2.3 GB per checkpoint and removes the largest Volume-space risk.

Cleanup policy: keep **best by val NDCG@10** and the **last** epoch only. Delete intermediate checkpoints as soon as a new "best" is recorded.

Worst-case Volume usage from checkpoints: 2 × 1.1 GB = ~2.2 GB at any time during training.

### 5.6 Progress reporting

Modal streams stdout to the local terminal during `modal run`. tqdm bars work but need throttling.

```python
from tqdm import tqdm
import sys

def pbar(iterable, **kw):
    return tqdm(iterable, mininterval=2.0, miniters=10, dynamic_ncols=True,
                file=sys.stdout, **kw)
```

Bars used:

| Loop | Description | Postfix |
|---|---|---|
| Pass 2 candidate caching | `"caching candidates"` | `{i}/{1_000_000}` |
| Training — epochs | `"epoch {n}/{N}"` | `loss={smoothed_loss:.3f} lr={lr:.2e}` |
| Training — batches per epoch | `"  step"` (nested, `position=1`) | `loss={batch_loss:.3f}` |
| Pipeline-mode eval | `"val pipeline"` | `R@100={running_recall:.3f}` |
| Unconstrained eval | `"test unconstrained"` | `R@100={running_recall:.3f}` |
| Test top-100 precompute | `"inferring test top100"` | `{batch}/{total}` |

After each epoch's val eval, print a plain summary line with `flush=True`:

```
[epoch 4/15] train_loss=4.821  val_R@10=0.082  val_R@100=0.234  val_NDCG@10=0.051  wall=22m18s  ETA=3h44m
```

This is the canonical "what's happening" log line; tqdm is for live monitoring.

---

## 6. Evaluation

Two modes, run at different points in the project. See §7.4 for derived report metrics.

### 6.1 Pipeline-mode (val and test)

For each batch of playlists:

```python
seq = playlist[:-1]  left-padded to (B, 50)
target = playlist[-1]                              # held-out last song
candidates = candidates.npy[playlist_idx]          # (B, 1000)

queries = transformer(seq)[:, -1, :]               # (B, 128) — last non-pad position
cand_emb = item_emb(candidates)                    # (B, 1000, 128)
scores = (queries[:, None, :] * cand_emb).sum(-1)  # (B, 1000)

# Defensive double-mask (Stage 1 already masks seen items)
mask_seen(scores, playlist[:-1])

top100_idx = argpartition(scores, -100)[:, -100:]
top100 = candidates[batch_idx, top100_idx]
```

Compute R@10, R@50, R@100, NDCG@10 relative to `target`.

**Important**: if `target` isn't in `candidates` (~59% of playlists, since Stage 1 R@1000 = 0.407), it contributes 0 to every metric. This is correct — it reflects real pipeline behavior.

### 6.2 Unconstrained mode (test only, once at end)

Same as pipeline-mode but the candidate set is the full vocab. To keep argpartition indices interpretable as IDs directly, score against the full `item_emb.weight` and mask PAD to `-inf`:

```python
queries = transformer(seq)[:, -1, :]               # (B, 128)
scores  = queries @ item_emb.weight.T              # (B, 2_262_293) — ~580 MB bf16 per batch of 256
scores[:, 0] = -inf                                # mask PAD
mask_seen(scores, playlist[:-1])                   # mask context positions to -inf
top1000 = argpartition(scores, -1000, axis=1)[:, -1000:]   # indices ARE IDs in +1-shifted space
```

Compute is heavier (~10 min on A10G for all 150K test playlists). Processed at batch size 256 (one matmul per batch).

### 6.3 Reported metrics

| Metric | Pipeline (within 1000) | Unconstrained (within 2.26M) |
|---|---|---|
| R@10 | x.xxx | x.xxx |
| R@50 | x.xxx | x.xxx |
| R@100 | x.xxx | x.xxx |
| R@1000 | = 0.407 (locked by Stage 1) | x.xxx |
| NDCG@10 | x.xxx | x.xxx |

### 6.4 Derived metrics (for the report's analysis)

- **Pipeline lift** = `Stage 2 pipeline R@100 / Stage 1 R@100` — how much Stage 2 improves over Stage 1 at K=100. Tells the "ranking improves over retrieval" story.
- **Headroom gap** = `unconstrained R@100 − pipeline R@100` — how much Stage 2 is being capped by Stage 1's recall. Motivates future Stage 1 work.
- **Stage-1-recoverable rate** = `pipeline R@10 / Stage 1 R@1000` = `pipeline R@10 / 0.407` — of the playlists where Stage 1 surfaced the target, how often does Stage 2 put it in top-10.

---

## 7. Modal infrastructure

### 7.1 App structure

Single file `stage_2/modal_app.py`:

```python
import modal

stub = modal.App("stage2-sasrec")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy>=1.24", "tqdm", "pyyaml")
    .add_local_dir(".", "/repo")
)

volume = modal.Volume.from_name("stage2-data", create_if_missing=True)

@stub.function(image=image, volumes={"/vol": volume}, gpu="a10g", timeout=3600 * 4)
def cache_stage1_candidates():
    # Pass 1 + Pass 2 from §4. Writes to /vol/derived/.
    ...

@stub.function(image=image, volumes={"/vol": volume}, gpu="a10g", timeout=3600 * 4)
def train(run_id: str, epochs: int = 15, smoke: bool = False):
    # If smoke=True: 1 epoch on full data + both eval modes at end.
    # Reads /vol/inputs/, /vol/derived/. Writes /vol/runs/<run_id>/.
    ...

@stub.function(image=image, volumes={"/vol": volume}, gpu="a10g", timeout=3600 * 2)
def evaluate(run_id: str, mode: str):
    # mode ∈ {"pipeline", "unconstrained"}. Runs on test set.
    ...

@stub.function(image=image, volumes={"/vol": volume}, gpu="a10g", timeout=3600)
def precompute_test_top100(run_id: str):
    # Pipeline-mode top-100 for every test playlist. Saves test_top100.npy + test_top100_scores.npy.
    ...

@stub.local_entrypoint()
def main(cmd: str, run_id: str = "main", epochs: int = 15, smoke: bool = False):
    ...
```

### 7.2 Local repo layout

```
stage_2/
├── modal_app.py              ← Modal entry points (above)
├── models/sasrec.py          ← Pure PyTorch model — testable locally
├── data/dataset.py           ← PyTorch Dataset, padding logic
├── training/train.py         ← train loop body, called by modal_app.train()
├── training/evaluate.py      ← eval loop body, called by modal_app.evaluate()
├── training/preprocess.py    ← Pass 1 + Pass 2 logic, called by modal_app.cache_stage1_candidates()
├── inference/predict.py      ← single-playlist inference (for Stage 3, Stage 4 UI)
├── tests/                    ← unit tests for pre-launch Tier 1 validation
├── config.yaml               ← all hyperparameters (single source of truth)
└── checkpoints/              ← gitignored; populated by `modal volume get`
```

### 7.3 Usage flow

```bash
# One-time data prep (~15 min upload + ~5 min compute)
modal run stage_2/modal_app.py --cmd cache

# Smoke run before committing to full training (~15 min)
modal run stage_2/modal_app.py --cmd train --run-id smoke --smoke

# Full run (~3 hours)
modal run stage_2/modal_app.py --cmd train --run-id main --epochs 15

# Final eval + precompute test outputs (~30 min)
modal run stage_2/modal_app.py --cmd eval --run-id main
modal run stage_2/modal_app.py --cmd infer --run-id main

# Pull small artifacts back to laptop
modal volume get stage2-data runs/main/test_top100.npy            ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_top100_scores.npy     ./stage_2/checkpoints/
modal volume get stage2-data runs/main/train_history.json         ./stage_2/checkpoints/
modal volume get stage2-data runs/main/test_metrics.json          ./stage_2/checkpoints/
modal volume get stage2-data runs/main/best_model.pt              ./stage_2/checkpoints/
```

---

## 8. Pre-launch validation

Three-tier gate before launching the 2.5-hour run.

| Tier | What | Where | Wall-clock | Cost | Pass criteria |
|---|---|---|---|---|---|
| **1. Unit tests** | `pytest stage_2/tests/` — SASRec forward, loss math, dataset shapes, causal-mask correctness, padding masking | Local CPU | ~2 min | $0 | All green |
| **2. Tiny local training** | `python -m stage_2.training.train --debug` on `playlists[:1000]`, 3 epochs, batch 32 | Local CPU/MPS | ~5 min | $0 | Train loss strictly decreasing; eval R@100 on val[:100] > 0.02 |
| **3. Modal smoke run** | `modal run stage_2/modal_app.py --cmd train --run-id smoke --smoke` | Modal A10G | ~15 min | ~$0.30 | Epoch wall-clock < 15 min; no NaN/Inf; pipeline val NDCG@10 > 0.01; unconstrained eval produces a number; all artifacts written |

If Tier 3 passes, optionally use its checkpoint as the resume-point for the full run (saves 1 epoch). Simpler path: discard.

**If any tier fails: stop and debug locally. Do not launch the full run.**

---

## 9. Stage 3 handoff contract

Stage 3 development consumes four files (pulled from Modal Volume to `stage_2/checkpoints/`):

```python
test_top100        = np.load("stage_2/checkpoints/test_top100.npy")           # (150_000, 100) int32
test_top100_scores = np.load("stage_2/checkpoints/test_top100_scores.npy")    # (150_000, 100) float32
item_embeddings    = np.load("stage_2/checkpoints/best_item_embeddings.npy")  # (2_262_293, 128) float32
uri_to_id          = json.load(open("stage_1/checkpoints/uri_to_id.json"))    # same as Stage 1
```

For each test playlist `i`, Stage 3 loops over `test_top100[i]`, applies MMR + audio-feature rules, and writes `test_final20.npy`.

**ID convention**: `test_top100.npy` IDs are in the **+1-shifted space** (track 5 in Stage 1 is ID 6 in Stage 2). Stage 3 must operate in this space, or subtract 1 to get back to Stage 1 IDs. Documented at the top of every Stage 2 artifact.

---

## 10. Failure modes and recovery

| Symptom | Likely cause | Recovery |
|---|---|---|
| Train loss NaN/Inf within first 100 steps | bf16 overflow, bad embedding init, divide-by-zero in masking | Fall back to fp32 autocast; verify ALS init loaded correctly (norm should be ~5–10) |
| Train loss decreases but val NDCG@10 stays near 0 | Eval pipeline bug — usually the query position or candidate indexing | Verify on Tier 2 with a tiny set and a known-good toy model |
| Train loss diverges after 1–2 epochs | LR too high for embedding fine-tuning | Reduce embedding peak LR from 1e-4 to 5e-5; rerun |
| Pipeline NDCG@10 plateaus but unconstrained keeps climbing | Hit pipeline ceiling (this is success) | Stop training; report headroom gap |
| Modal "out of disk" on Volume | Old checkpoints accumulated | `modal volume rm stage2-data runs/<old>/*.pt`; cleanup policy kicks in next run |
| Modal function timeout on candidate caching | Function timeout exceeded | Already set to 4h; if still timing out, batch the caching call |
| Epoch wall-clock 3× estimate | Data loader CPU-bound | Increase `num_workers`; profile with `torch.profiler` |
| Smoke run fails Tier 3 | Various | Do not launch full run. Debug. |

---

## 11. Reproducibility

- Single `stage_2/config.yaml` captures every hyperparameter; copied verbatim into each run's output dir.
- Seeds set on `torch.manual_seed`, `np.random.seed`, `random.seed` at the top of `main()`.
- `torch.use_deterministic_algorithms(True)` enabled.
- Modal Image pins specific package versions (`torch==2.4.0`).
- `git rev-parse HEAD` is written to `run_metadata.json` at the start of each run.
- `candidates.npy` is deterministic given a fixed `als_item_factors.npy` (argpartition order is implementation-defined but the set of top-1000 is deterministic).

---

## 12. Compute and time budget

| Phase | Wall-clock | Cost (A10G @ $1.10/hr) |
|---|---|---|
| Upload inputs to Modal Volume (one-time) | ~10 min (network) | $0 |
| Pass 1 + Pass 2 (candidate caching) | ~5 min | $0.10 |
| Tier 3 smoke run | ~15 min | $0.30 |
| Full training (15 epochs, A10G) | ~2.5 hrs | $2.75 |
| Val evals during training (8 × 2 min) | ~15 min | $0.30 |
| Final pipeline + unconstrained eval | ~15 min | $0.30 |
| `precompute_test_top100` | ~5 min | $0.10 |
| Safety margin (1 retry of training) | — | $3.00 |
| **Expected total** | **~3.5–4 hrs** | **~$7** |
| **Worst-case total (with retry)** | **~7 hrs** | **~$10** |

Modal Volume storage: ~7 GB persistent + ~3 GB ephemeral = ~10 GB. Within 50 GB free allowance.

Local laptop storage: ~70 MB (small artifacts only) to ~2.6 GB (small + Stage 1 inputs + fine-tuned embeddings).

---

## 13. Out of scope and follow-ups

- **Ablations** (frozen vs fine-tuned embeddings; hard vs in-batch negatives). Compute budget excludes them. If a future hour of Modal time is available, the highest-leverage cheap ablation is "in-batch only at 8 epochs."
- **Multi-target loss variants** (predict multiple held-out songs). Single-target is sufficient for the project narrative.
- **Larger model (Option B/C from brainstorming)**. Skipped due to time constraints; Option A is sufficient for course-project metrics.
- **W&B / TensorBoard logging**. `train_history.json` is sufficient.
- **Resuming training across Modal function invocations**. Single training call is < timeout; not needed.
