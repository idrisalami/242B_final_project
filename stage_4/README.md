# Stage 4 — Streamlit Interface + Analysis

**Status: BUILT.**

Stage 4 wraps the completed recommendation pipeline in a transparent Streamlit demo and adds a compact experiment-analysis view.

```
Spotify playlist URL or pasted track URIs
      ↓
Stage 1 ALS candidate generation       raw IDs      → top 1,000
      ↓
Stage 2 SASRec sequential re-ranking   shifted IDs  → top 100
      ↓
Stage 3 MMR diversity re-ranking       shifted IDs  → final top 5-25
      ↓
Spotify metadata display
```

The app never overclaims. If a checkpoint is missing, it falls back to the strongest available local stage and shows that status in the UI.

---

## Files

```
stage_4/
├── app.py          Streamlit app
├── pipeline.py     artifact detection + Stage 1/2/3 integration
├── spotify.py      Spotify URL parsing, playlist fetch, metadata fetch
├── tests/          Stage 4 unit tests
└── README.md
```

---

## Required and Optional Artifacts

Required for any recommendations:

```
stage_1/checkpoints/
├── als_item_factors.npy
└── uri_to_id.json
```

Optional for real Stage 2 + Stage 3:

Flat layout from the Stage 2 README:

```
stage_2/checkpoints/
├── best_model.pt
└── best_item_embeddings.npy
```

or nested Modal checkpoint layout:

```
stage_2/checkpoints/best/
├── model.pt
└── item_embeddings.npy
```

Optional for the analysis tab:

```
stage_2/checkpoints/test_metrics.json
stage_2/checkpoints/train_history.json
stage_3/checkpoints/test_metrics.json
stage_3/checkpoints/lambda_sweep.json
```

If the JSON files are missing, the analysis tab uses the documented results from the Stage 2 and Stage 3 READMEs:

- Stage 2: R@10 = 0.082, R@100 = 0.236, NDCG@10 = 0.046.
- Stage 3 λ sweep:
  - λ=0.3: Recall@20 = 0.0975, ILD = 0.4958.
  - λ=0.5: Recall@20 = 0.1135, ILD = 0.4327.
  - λ=0.7: Recall@20 = 0.1173, ILD = 0.3925.

---

## ID Convention

This is the most important integration detail.

- Stage 1 uses raw MPD track IDs: `0, 1, 2, ...`.
- Stage 2 and Stage 3 use shifted IDs: `0` is PAD, so real tracks are `raw_id + 1`.
- Before calling Stage 2/3, Stage 4 shifts raw IDs by `+1`.
- Before mapping final recommendations back to Spotify URIs, Stage 4 subtracts `1`.

The conversion helpers live in `stage_4/pipeline.py`:

```python
raw_to_shifted([0, 41])      # [1, 42]
shifted_to_raw([1, 42])      # [0, 41]
```

---

## Spotify Credentials

Playlist URL mode requires Spotify API credentials.

Create `stage_4/.env`:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

Do not commit `.env`. The repo-level `.gitignore` already ignores it.

If credentials are unavailable, use **Paste track URIs** mode. Recommendations can still run, but metadata will be limited.

---

## Run the App

Install dependencies:

```bash
uv sync
```

Run:

```bash
PYTHONPATH=. uv run streamlit run stage_4/app.py
```

The app shows:

- Stage availability: ALS ready/missing, SASRec ready/fallback, MMR ready/fallback.
- MPD coverage: how many input tracks are known in `uri_to_id.json`.
- Recommendation cards with rank, artist, album art, Spotify link, and score when metadata is available.
- Analysis charts for Stage 2 ranking quality and Stage 3 relevance/diversity tradeoff.

Open the **Analysis** tab to see:

- Model comparison chart: Stage 1 ALS vs Stage 2 SASRec on R@10, R@100, and NDCG@10.
- Stage 2 training curves if `train_history.json` is available locally.
- MMR lambda sweep: λ=0.3, 0.5, 0.7 plotted against Recall@20 and ILD.
- Percent-change chart: recall loss and diversity gain vs the Stage 2 raw top-20 baseline.
- Parameter-variation table explaining which knobs were evaluated offline and which are demo-only.

---

## Export Analysis Tables

To generate concise CSV tables for the report:

```bash
PYTHONPATH=. uv run python -m stage_4.analysis --out-dir stage_4/analysis_exports
```

This writes:

```
stage_4/analysis_exports/
├── stage_comparison.csv       Stage 1 vs Stage 2 ranking metrics
├── lambda_sweep.csv           λ sweep with Recall@20, ILD, and % changes
└── parameter_variations.csv   What parameters were varied and why
```

These exports use local JSON metrics if present. If the JSON files are missing, they use the documented README values so the graphs remain available for the presentation.

---

## Fallback Behavior

| Missing artifact | Behavior |
|---|---|
| Stage 1 ALS factors or URI map | App blocks recommendation generation and shows setup instructions |
| Stage 2 model/embeddings | Uses Stage 1 ALS top-100 as ranking fallback |
| Stage 3 embeddings | Uses top-N ranked items without MMR |
| Spotify credentials | Allows pasted track URIs and displays minimal metadata |

These fallbacks are for demo robustness only. The UI labels them explicitly.

---

## Tests

Run Stage 4 tests:

```bash
PYTHONPATH=. uv run pytest stage_4/tests/ -v
```

The tests cover:

- Spotify playlist URL parsing.
- Spotify track URI parsing.
- raw ID ↔ shifted ID conversion.
- Stage 2 artifact layout detection.
- Stage 1 masking of already-seen input tracks.
