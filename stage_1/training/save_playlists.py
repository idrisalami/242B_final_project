"""
Save the processed playlist sequences to checkpoints/playlists.npy.

This lets Stage 2 teammates load training data without downloading the 31 GB
raw MPD dataset — they only need the three checkpoint files from Google Drive.

Usage (from stage_1/):
    python training/save_playlists.py

Output:
    checkpoints/playlists.npy  — numpy object array of shape (1000000,)
                                 each element is a List[int] of track IDs
                                 same integer ID space as als_item_factors.npy
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_loader import DataLoader_TwoTower
from utils.helpers import load_config

cfg     = load_config("config/config.yaml")
raw_dir = Path(cfg["data"]["data_path"]) / "data"

print("Loading MPD slices (~15 min)…")
t0 = time.time()
dataset = DataLoader_TwoTower.load_spotify_mpd_slices(
    str(raw_dir),
    max_files=cfg["data"].get("max_files"),
    min_playlist_length=cfg["data"]["min_playlist_length"],
)
playlists = dataset["playlists"]
print(f"Loaded {len(playlists):,} playlists  ({(time.time()-t0)/60:.1f} min)")

save_path = Path(cfg["checkpoint"]["save_path"]) / "playlists.npy"
save_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Saving to {save_path}…")
np.save(save_path, np.array(playlists, dtype=object))

size_mb = save_path.stat().st_size / 1e6
print(f"Saved {len(playlists):,} playlists  →  {save_path}  ({size_mb:.0f} MB)")
print()
print("Load in Stage 2:")
print("  playlists = np.load('stage_1/checkpoints/playlists.npy', allow_pickle=True).tolist()")
