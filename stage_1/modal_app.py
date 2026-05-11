"""Modal entry point for retraining Stage 1 (ALS) and saving its checkpoints.

This exists because the original Stage 1 outputs (als_item_factors.npy,
uri_to_id.json, playlists.npy) were lost. We download the raw MPD dataset
from AICrowd into a Modal Volume, run ALS, and persist the three artifacts
into /vol/inputs/ for Stage 2 to consume.

Local usage (from repo root):
    modal run stage_1/modal_app.py

Prerequisites:
  1. AICrowd account with accepted MPD challenge license
     (https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
  2. AICrowd API key from https://www.aicrowd.com/participants/me
  3. Modal Secret named 'aicrowd-key' containing env var AICROWD_API_KEY=<key>
     Create via: modal secret create aicrowd-key AICROWD_API_KEY=<key>

Wall-clock: ~30-45 min on Modal CPU (most of it is MPD download/extract).
Cost: ~$0.05-0.15 on Modal CPU pricing.
"""

import modal

APP_NAME = "stage1-retrain"
VOLUME_NAME = "stage2-data"

stub = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install(
        "aicrowd-cli>=0.1.15",
        "implicit>=0.7.2",
        "numpy>=1.24",
        "scipy>=1.10",
        "tqdm>=4.67",
    )
    .add_local_dir(".", "/repo")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@stub.function(
    image=image,
    volumes={"/vol": volume},
    cpu=4.0,
    memory=16384,
    timeout=3600 * 3,                              # 3 hours
    secrets=[modal.Secret.from_name("aicrowd-key")],
)
def retrain():
    """Download MPD, train ALS, save the three Stage 1 checkpoint files."""
    import json
    import os
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    import scipy.sparse as sp
    from implicit.als import AlternatingLeastSquares

    sys.path.insert(0, "/repo")
    from stage_1.data.data_loader import DataLoader_TwoTower

    raw_dir = Path("/vol/raw_mpd")
    inputs_dir = Path("/vol/inputs")
    raw_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Download + extract MPD if not already present ──────────────────────────
    data_dir = raw_dir / "spotify_million_playlist_dataset" / "data"
    if not data_dir.exists():
        # Sometimes the extracted layout differs by mirror. Check a couple of locations.
        alt = raw_dir / "data"
        if alt.exists() and any(alt.glob("mpd.slice.*.json")):
            data_dir = alt

    if not data_dir.exists() or not any(data_dir.glob("mpd.slice.*.json")):
        api_key = os.environ.get("AICROWD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AICROWD_API_KEY env var not set. "
                "Create Modal Secret 'aicrowd-key' via "
                "`modal secret create aicrowd-key AICROWD_API_KEY=<key>`."
            )

        print("Logging in to AICrowd...")
        subprocess.run(["aicrowd", "login", "--api-key", api_key], check=True)

        print("Downloading MPD via aicrowd-cli (~5 GB, ~5-10 min)...")
        t0 = time.time()
        subprocess.run(
            [
                "aicrowd", "dataset", "download",
                "--challenge", "spotify-million-playlist-dataset-challenge",
                "-o", str(raw_dir),
            ],
            check=True,
        )
        print(f"Download done in {(time.time()-t0)/60:.1f} min")

        # Extract all .zip files in raw_dir
        zips = list(raw_dir.glob("*.zip"))
        if not zips:
            raise RuntimeError(f"No .zip found in {raw_dir} after download.")
        for z in zips:
            print(f"Extracting {z.name} ({z.stat().st_size / 1e9:.1f} GB)...")
            t0 = time.time()
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw_dir)
            print(f"  done in {(time.time()-t0)/60:.1f} min")
            z.unlink()                              # free disk
        volume.commit()

        # Re-resolve data_dir post-extraction
        candidates = list(raw_dir.rglob("mpd.slice.0-999.json"))
        if not candidates:
            raise RuntimeError(
                f"Could not find mpd.slice.0-999.json under {raw_dir} after extraction. "
                f"Listing: {[str(p.relative_to(raw_dir)) for p in raw_dir.iterdir()]}"
            )
        data_dir = candidates[0].parent
        print(f"MPD data dir resolved to: {data_dir}")
    else:
        print(f"MPD already extracted at {data_dir} — skipping download.")

    # ── Load slices into integer-ID playlists ─────────────────────────────────
    print("Loading MPD slices (this takes ~15 min for all 1000 files)...")
    t0 = time.time()
    dataset = DataLoader_TwoTower.load_spotify_mpd_slices(
        str(data_dir),
        max_files=1000,
        min_playlist_length=5,
    )
    playlists = dataset["playlists"]
    vocab_size = dataset["num_unique_tracks"]
    uri_to_id = dataset["uri_to_id"]
    print(
        f"  Loaded {len(playlists):,} playlists  "
        f"vocab={vocab_size:,}  ({(time.time()-t0)/60:.1f} min)"
    )

    # ── Build sparse playlist x track matrix (train split = first 70%) ────────
    n_total = len(playlists)
    train_size = int(n_total * 0.70)
    train_playlists = playlists[:train_size]
    print(f"Train split: {train_size:,} playlists (of {n_total:,})")

    print("Building sparse matrix...")
    t0 = time.time()
    rows, cols = [], []
    for pid, pl in enumerate(train_playlists):
        # Use context only (all-but-last) — mirrors the eval task
        context = pl[:-1] if len(pl) > 1 else pl
        for tid in context:
            rows.append(pid)
            cols.append(tid)
    user_item = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(train_playlists), vocab_size),
        dtype=np.float32,
    )
    print(f"  nnz={user_item.nnz:,}  ({time.time()-t0:.1f}s)")

    # ── Train ALS ──────────────────────────────────────────────────────────────
    FACTORS = 128
    ITERATIONS = 15
    REGULARIZATION = 0.01
    ALPHA = 40
    print(f"Training ALS factors={FACTORS} iter={ITERATIONS} reg={REGULARIZATION}")
    t0 = time.time()
    model = AlternatingLeastSquares(
        factors=FACTORS,
        iterations=ITERATIONS,
        regularization=REGULARIZATION,
        use_gpu=False,
    )
    user_item_weighted = (user_item * ALPHA).tocsr()
    model.fit(user_item_weighted)
    print(f"ALS done in {(time.time()-t0)/60:.1f} min")

    item_factors = np.array(model.item_factors, dtype=np.float32)
    print(f"Item factors: {item_factors.shape}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    print(f"Saving outputs to {inputs_dir}/")
    np.save(inputs_dir / "als_item_factors.npy", item_factors)
    with open(inputs_dir / "uri_to_id.json", "w") as f:
        json.dump(uri_to_id, f)
    np.save(inputs_dir / "playlists.npy", np.array(playlists, dtype=object))

    print(f"  als_item_factors.npy: {item_factors.shape}")
    print(f"  uri_to_id.json:       {len(uri_to_id):,} URIs")
    print(f"  playlists.npy:        {len(playlists):,} playlists")
    volume.commit()
    print("Stage 1 retrain complete.")


@stub.local_entrypoint()
def main():
    retrain.remote()
