# Multi-Stage Spotify Recommendation System

This project builds an end-to-end playlist continuation system on the Spotify Million Playlist Dataset (MPD). Given a partial playlist as Spotify track URIs, the app recommends additional tracks that fit the same playlist context.

## How to Use the App

Install dependencies:

```bash
uv sync
```

Create .env file with the following details:
```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

## Checkpoint Files

Checkpoint files are stored in Google Drive due to their large size (>2.5GB total):

📁 **[Download checkpoints](https://drive.google.com/drive/folders/1_QebdOx9KCHu2d9rLRuYK9i0-rZ9DGuv?usp=share_link)**

Download and extract them to their respective stage directories:
- `stage_1/checkpoints/` — ALS item factors and URI mappings
- `stage_2/checkpoints/` — SASRec model and embeddings
- `stage_3/checkpoints/` — MMR sweep results

Run the Streamlit interface:

```bash
PYTHONPATH=. uv run streamlit run stage_4/app.py
```bash


In the app:

1. Paste Spotify track URIs into the text box.
2. Click **Confirm tracks**.
3. Choose the number of recommendations.
4. Adjust the MMR lambda slider if desired.
5. Click **Generate recommendations**.

The app reports MPD coverage, meaning how many pasted tracks exist in the training-time Spotify URI mapping. Recommendations can only be generated from tracks covered by this local MPD catalog. A good demo input should show nonzero coverage, ideally all tracks found.

Example input:

```text
spotify:track:0UaMYEvWZi0ZqiDOoHU3YI
spotify:track:6I9VzXrHxO9rA9A5euc8Ak
spotify:track:0WqIKmW4BTrj3eJFmnCKMv
spotify:track:1AWQoqb9bSvzTjaLralEkT
spotify:track:1lzr43nnXAijIGYnCT8M8H
spotify:track:0XUfyU2QviPAs6bxSpXYG4
spotify:track:68vgtRHr7iZHpzGpon6Jlo
spotify:track:3BxWKCI06eQ5Od8TY2JBeA
spotify:track:7H6ev70Weq6DdpZyyTmUXk
spotify:track:2PpruBYCo4H7WOBJ7Q2EwM
```

For a presentation, `MMR lambda = 0.7` is a good default: it keeps recommendations relevance-focused while still adding some diversity.

## What Was Built

The system uses a three-stage recommendation pipeline:

```text
Pasted Spotify track URIs
        ↓
Stage 1: ALS candidate generation
        ↓
Stage 2: SASRec sequential re-ranking
        ↓
Stage 3: MMR diversity re-ranking
        ↓
Ranked recommendations in the web app
```

Stage 1 retrieves 1000 candidate tracks from the full 2.26M-track catalog using implicit-feedback ALS. Stage 2 uses a SASRec-style Transformer to re-rank those candidates based on playlist order. Stage 3 applies Maximal Marginal Relevance (MMR) to trade off relevance and diversity in the final list.

## Results

| System | Recall@100 | NDCG@10 |
|---|---:|---:|
| Stage 1 ALS | 0.151 | 0.018 |
| Stage 2 pipeline | 0.236 | 0.046 |

Stage 2 improves Recall@100 by about 56% and NDCG@10 by about 156% over ALS alone.

For diversity re-ranking:

| Configuration | Recall@20 | ILD |
|---|---:|---:|
| Stage 2 raw top-20 | 0.119 | 0.368 |
| MMR lambda = 0.5 | 0.114 | 0.433 |
| MMR lambda = 0.7 | 0.117 | 0.393 |

The reported `lambda = 0.5` setting increases intra-list diversity by about 18% with roughly a 4% Recall@20 loss. The app defaults to `lambda = 0.7` because it gives a cleaner live-demo balance between relevance and diversity.

## Required Artifacts

The app requires the Stage 1 artifacts:

```text
stage_1/checkpoints/
  als_item_factors.npy
  uri_to_id.json
```

Optional artifacts enable the full Stage 2 and Stage 3 path:

```text
stage_2/checkpoints/best/model.pt
stage_2/checkpoints/best/item_embeddings.npy
```

or the flat checkpoint layout:

```text
stage_2/checkpoints/best_model.pt
stage_2/checkpoints/best_item_embeddings.npy
```

If Stage 2 or Stage 3 artifacts are missing, the app labels the fallback behavior explicitly.

## Repository Structure

```text
stage_1/    ALS candidate generation and training scripts
stage_2/    SASRec model, training, and inference
stage_3/    MMR diversity re-ranking
stage_4/    Streamlit app and analysis view
docs/       planning notes
report.tex  final project report, if included locally
```

## Tests

Run the Stage 4 tests:

```bash
PYTHONPATH=. uv run pytest stage_4/tests
```

These tests cover URI parsing, artifact detection, raw/shifted ID conversion, input-track masking, Spotify fallback behavior, and the analysis helpers.
