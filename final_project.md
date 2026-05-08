# Spotify Recommendation System — Full Project Pipeline

## Project Goal

Build a modern multi-stage Spotify recommendation system inspired by:

- RecSys Challenge 2018 approaches
- industrial recommender system pipelines
- sequential recommendation architectures

Task:

```text
Given a playlist context (first N songs),
predict the next likely songs.
```

---

# Final Pipeline Overview

```text
Playlist Context
      ↓
Stage 1: Candidate Generation (ALS)
      ↓
Top 1000 Candidate Songs
      ↓
Stage 2: Sequential Reranking (SASRec)
      ↓
Top 100 Ranked Songs
      ↓
Stage 3: Diversity Re-ranking (MMR)
      ↓
Final Top 20 Recommendations
```

---

# Dataset

## Spotify Million Playlist Dataset (MPD)

- 1,000,000 playlists
- 2,262,292 unique tracks
- ~45M interactions

Use:
- full dataset OR filtered subset

---

# Evaluation Setup

## Task Definition

Given:

```text
[A, B, C]
```

Predict:

```text
[D, E]
```

This is a sequential recommendation problem.

---

# Metrics

| Metric | Purpose |
|---|---|
| Recall@10 | Top recommendation quality |
| Recall@100 | Retrieval quality |
| Recall@1000 | Candidate generation quality |
| NDCG@10 | Ranking quality |
| MRR | Early ranking correctness |

---

# STAGE 0 — Data Processing

## Goals

- Parse MPD JSON files
- Create track ID mappings
- Build playlist sequences
- Construct sparse matrices

---

## Representation

Convert:

```text
spotify:track:abc123
```

to:

```text
integer track IDs
```

Store:
- `track_id ↔ uri`
- playlist sequences
- sparse interaction matrix

---

# STAGE 1 — Candidate Generation

## Goal

Reduce:

```text
2.26M tracks
```

to:

```text
top 1000 likely candidates
```

efficiently.

This stage optimizes:
- high recall
- scalability
- retrieval speed

NOT ranking precision.

---

# Baseline 1 — Popularity Recommender

Recommend globally popular tracks.

## Expected Results

| Metric | Expected |
|---|---|
| Recall@10 | 0.01–0.03 |
| Recall@100 | 0.02–0.05 |

Purpose:
- sanity check
- simple baseline

---

# Baseline 2 — Item Similarity

Use:
- co-occurrence counts
- cosine similarity

## Expected Results

| Metric | Expected |
|---|---|
| Recall@10 | 0.05–0.10 |
| Recall@100 | 0.07–0.12 |

---

# Main Retrieval Model — ALS Matrix Factorization

## Method

Sparse matrix factorization:

```text
X ≈ U Vᵀ
```

Where:
- U = playlist embeddings
- V = track embeddings

---

## Recommended Configuration

```python
factors = 128
iterations = 50
regularization = 0.01
alpha = 40
weighting = BM25
```

Library:
- `implicit`

---

## Inference

Given playlist:

```text
[track1, track2, track3]
```

Compute:

```python
playlist_embedding = mean(track_embeddings)
```

Then retrieve:
- top 1000 nearest tracks

---

## Expected Results

| Metric | Goal |
|---|---|
| Recall@100 | 0.16–0.20 |
| Recall@1000 | 0.45–0.55 |

---

## Strengths

- scalable
- sparse-friendly
- fast training
- strong retrieval performance

---

## Weaknesses

- ignores sequence order
- mean-pooling bottleneck
- no transition modeling

This motivates Stage 2.

---

# STAGE 2 — Sequential Reranking (Core DL Component)

## Goal

Take:
- top 1000 ALS candidates

and rerank using:
- playlist sequence information

---

# Model — SASRec-Style Transformer

Use:
- causal self-attention
- positional embeddings
- sequence masking

---

## Input

```text
[track1, track2, track3, ...]
```

---

## Output

Predict:
- next likely track(s)

---

# Architecture

## Embedding Layer

```python
track_embedding(track_id)
```

Embedding dimension:
- 64 or 128

---

## Positional Embeddings

Adds:
- sequence order information

---

## Transformer Blocks

Recommended:

| Component | Value |
|---|---|
| Layers | 1–2 |
| Heads | 2–4 |
| Hidden dim | 128 |
| Dropout | 0.2 |

Keep the model small for consumer hardware.

---

# Causal Masking

Future songs must be hidden.

This ensures:
- autoregressive prediction
- valid sequential learning

---

# Loss Function

Recommended:
- sampled softmax
- BPR loss
- cross-entropy with negative sampling

Example:
- 1 positive next song
- 99 sampled negatives

---

# Training Strategy

IMPORTANT:

Do NOT score all 2.26M tracks.

Instead:
- train on sampled candidates
- use ALS negatives/candidates

This reduces compute dramatically.

---

# Expected Results

| Metric | Goal |
|---|---|
| Recall@10 | 0.18–0.25 |
| Recall@100 | 0.18–0.22 |
| NDCG@10 | major improvement |

---

# Key Insight

ALS:
- retrieves broadly

Transformer:
- ranks contextually

Retrieval and ranking are different problems.

---

# STAGE 3 — Diversity Re-ranking

## Problem

Without reranking:
- recommendations become repetitive
- same artists dominate
- diversity decreases

---

# Solution — MMR

Maximal Marginal Relevance balances:
- relevance
- diversity

Formula:

```text
score =
λ * relevance
− (1−λ) * similarity_to_previous
```

---

# Similarity

Use:
- cosine similarity
between track embeddings.

---

# Final Output

Return:
- top 20–30 recommendations

---

# STAGE 4 — Experiments & Analysis

This section is critical for grading.

---

# Required Experiments

## 1. Model Comparison

| Model | Recall@10 | Recall@100 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.02 | 0.05 | 0.01 |
| Cosine Similarity | 0.08 | 0.12 | 0.05 |
| ALS | 0.15 | 0.18 | 0.11 |
| SASRec | 0.20 | 0.22 | 0.18 |
| Final Pipeline | 0.22 | 0.23 | 0.20 |

---

# 2. Training Curves

Plot:
- training loss
- validation loss
- Recall@K over epochs

Discuss:
- overfitting
- convergence
- sparsity

---

# 3. Ablation Studies

Test:
- no positional embeddings
- different transformer depths
- embedding dimension
- candidate size

---

# STAGE 5 — Deployment (Optional Bonus)

## Streamlit App

Input:
- playlist songs

Output:
- recommended songs

Potential additions:
- album art
- Spotify links
- similarity explanations

This can help secure bonus points.

---

# Recommended Folder Structure

```text
project/
├── data/
├── preprocessing/
├── baselines/
├── retrieval/
│   └── ALS
├── ranking/
│   └── SASRec
├── reranking/
│   └── MMR
├── evaluation/
├── app/
├── checkpoints/
└── report/
```

---

# Strong Scientific Narrative

The project should emphasize:

1. Neural retrieval at Spotify scale is computationally expensive

2. Classical collaborative filtering remains extremely competitive

3. Mean-pooled retrieval loses sequential information

4. Sequential transformers improve ranking quality by modeling playlist order

---

# Recommended Division of Work (2 People)

## Person A — Retrieval & Systems

Responsible for:
- sparse matrices
- ALS retrieval
- evaluation pipeline
- candidate generation
- FAISS (optional)

---

## Person B — Deep Learning & Ranking

Responsible for:
- SASRec transformer
- sequence modeling
- ranking loss
- reranking pipeline

---

# Recommended Final Targets

| Metric | Strong Target |
|---|---|
| Recall@10 | 0.18–0.25 |
| Recall@100 | 0.18–0.22 |
| Recall@1000 | 0.45–0.55 |
| NDCG@10 | 0.15–0.20 |

---

# Key Final Advice

Do NOT optimize for:
- giant architectures
- state-of-the-art compute

Optimize for:
- complete pipeline
- rigorous evaluation
- thoughtful analysis
- clean methodology

A finished, well-analyzed two-stage recommender system is significantly stronger than an unfinished giant model.
