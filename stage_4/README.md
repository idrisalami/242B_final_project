# Stage 4 — Interface

**Status: NOT BUILT.**

A Streamlit app that lets a user paste a Spotify playlist URL, runs the full pipeline, and displays the recommended songs.

---

## What This Stage Does

Wraps all three pipeline stages into a single interactive UI:

```
User pastes Spotify playlist URL
      ↓
Fetch track URIs from Spotify API
      ↓
Stage 1 (ALS)   — 2.26M tracks → top 1,000
      ↓
Stage 2 (SASRec) — 1,000 → top 100          ← stub until Stage 2 is built
      ↓
Stage 3 (MMR)   — 100 → final 20–30         ← stub until Stage 3 is built
      ↓
Display recommended tracks (name, artist, album art)
```

---

## What You Need

From Google Drive (`stage_1/checkpoints/`):

| File | Used for |
|---|---|
| `als_item_factors.npy` | Stage 1 retrieval |
| `uri_to_id.json` | URI ↔ integer ID mapping |

From teammates (once built):
- `stage_2/checkpoints/sasrec_model.pt` — Stage 2 re-ranking
- Stage 3 is pure Python (no checkpoint needed)

From Spotify:
- A **Spotify API client ID + secret** — needed to fetch track metadata (name, artist, album art) from a playlist URL. Free at [developer.spotify.com](https://developer.spotify.com/dashboard).

---

## Step 1 — Install dependencies

```bash
pip install streamlit spotipy numpy
```

`spotipy` is the Python wrapper for the Spotify Web API.

---

## Step 2 — Set up Spotify API credentials

Create a `.env` file in `stage_4/`:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

Then load them in code:

```python
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ['SPOTIFY_CLIENT_ID'],
    client_secret=os.environ['SPOTIFY_CLIENT_SECRET'],
))
```

---

## Step 3 — Fetch tracks from a playlist URL

Given a playlist URL like `https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M`, extract the playlist ID and fetch its tracks:

```python
def fetch_playlist_tracks(playlist_url: str) -> list[dict]:
    """Returns list of {uri, name, artist, album_art} dicts."""
    playlist_id = playlist_url.split('/')[-1].split('?')[0]
    results = sp.playlist_tracks(playlist_id)
    tracks = []
    for item in results['items']:
        t = item['track']
        if t is None:
            continue
        tracks.append({
            'uri':       t['uri'],                          # "spotify:track:abc123"
            'name':      t['name'],
            'artist':    t['artists'][0]['name'],
            'album_art': t['album']['images'][0]['url'] if t['album']['images'] else None,
        })
    return tracks
```

---

## Step 4 — Run the pipeline

```python
import numpy as np
import json

# Load Stage 1 artifacts once at startup
item_factors = np.load('../stage_1/checkpoints/als_item_factors.npy')
uri_to_id    = json.load(open('../stage_1/checkpoints/uri_to_id.json'))
id_to_uri    = {v: k for k, v in uri_to_id.items()}

def stage1(playlist_uris: list[str], k: int = 1000) -> np.ndarray:
    ids = [uri_to_id[u] for u in playlist_uris if u in uri_to_id]
    if not ids:
        return np.array([])
    user_emb = item_factors[ids].mean(axis=0)
    scores   = user_emb @ item_factors.T
    scores[ids] = -1e9
    top_k = np.argpartition(scores, -k)[-k:]
    return top_k[np.argsort(scores[top_k])[::-1]]

def stage2_stub(playlist_uris: list[str], candidate_ids: np.ndarray) -> np.ndarray:
    """Placeholder — returns top-100 of Stage 1 candidates by ALS score."""
    return candidate_ids[:100]

def stage3_stub(candidate_ids: np.ndarray) -> list[str]:
    """Placeholder — returns top-25 as URIs."""
    return [id_to_uri[i] for i in candidate_ids[:25] if i in id_to_uri]

def recommend(playlist_uris: list[str]) -> list[str]:
    candidates = stage1(playlist_uris, k=1000)   # Stage 1
    ranked     = stage2_stub(playlist_uris, candidates)  # Stage 2 (stub)
    final_uris = stage3_stub(ranked)             # Stage 3 (stub)
    return final_uris
```

Replace `stage2_stub` and `stage3_stub` with real implementations once those stages are built.

---

## Step 5 — Build the Streamlit app

Create `app.py`:

```python
import streamlit as st

st.title("Spotify Playlist Recommender")

playlist_url = st.text_input(
    "Paste a Spotify playlist URL",
    placeholder="https://open.spotify.com/playlist/...",
)

if playlist_url:
    with st.spinner("Fetching playlist…"):
        tracks = fetch_playlist_tracks(playlist_url)

    if not tracks:
        st.error("Could not fetch tracks. Check the URL and try again.")
    else:
        st.subheader(f"Input playlist ({len(tracks)} tracks)")
        for t in tracks[:5]:
            st.write(f"- {t['name']} — {t['artist']}")
        if len(tracks) > 5:
            st.caption(f"… and {len(tracks)-5} more")

        with st.spinner("Generating recommendations…"):
            playlist_uris  = [t['uri'] for t in tracks]
            rec_uris       = recommend(playlist_uris)
            rec_meta       = {t['uri']: t for t in tracks}

        st.subheader("Recommendations")
        for uri in rec_uris:
            # Fetch metadata for recommended tracks not already in input
            if uri not in rec_meta:
                track_id = uri.split(':')[-1]
                t = sp.track(track_id)
                rec_meta[uri] = {
                    'name':      t['name'],
                    'artist':    t['artists'][0]['name'],
                    'album_art': t['album']['images'][0]['url'] if t['album']['images'] else None,
                }
            meta = rec_meta[uri]
            col1, col2 = st.columns([1, 5])
            with col1:
                if meta['album_art']:
                    st.image(meta['album_art'], width=60)
            with col2:
                st.write(f"**{meta['name']}** — {meta['artist']}")
```

Run it:

```bash
cd stage_4
streamlit run app.py
```

---

## Step 6 — Swap in real Stage 2 and Stage 3

Once teammates finish:

**Stage 2** — replace `stage2_stub`:
```python
import torch
# from stage_2 import SASRec  (import their model)

model = SASRec(...)
model.load_state_dict(torch.load('../stage_2/checkpoints/sasrec_model.pt'))
model.eval()

def stage2(playlist_uris, candidate_ids):
    # convert URIs → int IDs, run SASRec, return top-100
    ...
```

**Stage 3** — replace `stage3_stub`:
```python
# from stage_3 import mmr_select, apply_rules

def stage3(candidate_ids, candidate_scores, audio_features):
    final = mmr_select(candidate_ids, candidate_scores, audio_features)
    return apply_rules(final, audio_features)
```

---

## File Structure

```
stage_4/
├── README.md       (this file)
├── app.py          Streamlit app
├── pipeline.py     stage1(), stage2(), stage3(), recommend()
├── spotify.py      fetch_playlist_tracks() and Spotify API helpers
└── .env            SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (never commit this)
```

---

## Important

- **Never commit `.env`** — add it to `.gitignore`
- The app works with just Stage 1 from the start — stubs fill in for Stages 2 and 3
- Spotify API rate limit: 30 requests/sec — fine for a demo, no throttling needed
