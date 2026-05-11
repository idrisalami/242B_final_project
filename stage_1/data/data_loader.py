"""Reconstructed loader for the Spotify Million Playlist Dataset.

Reads MPD slice files (`mpd.slice.<lo>-<hi>.json`), extracts `track_uri` sequences,
and assigns integer IDs in first-appearance order across files sorted by `<lo>`.

This is the contract used by `stage_1/training/train_als.py`:

    dataset = DataLoader_TwoTower.load_spotify_mpd_slices(
        raw_dir, max_files=..., min_playlist_length=...
    )
    dataset["playlists"]          -> List[List[int]]
    dataset["num_unique_tracks"]  -> int
    dataset["uri_to_id"]          -> dict[str, int]

The original `data_loader.py` was authored prior to this session and is gitignored
(`stage_1/data/`). This reconstruction is force-added.
"""

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional


def _slice_sort_key(path: str) -> int:
    """Sort `mpd.slice.<lo>-<hi>.json` filenames by the integer `lo`."""
    stem = Path(path).stem                           # e.g., "mpd.slice.1000-1999"
    lo_str = stem.split(".")[-1].split("-")[0]       # "1000"
    return int(lo_str)


class DataLoader_TwoTower:
    """Static loader for Spotify MPD slice files."""

    @staticmethod
    def load_spotify_mpd_slices(
        raw_dir: str,
        max_files: Optional[int] = None,
        min_playlist_length: int = 5,
    ) -> Dict:
        """Load MPD slices, return playlists as integer-encoded sequences.

        Args:
            raw_dir: directory containing `mpd.slice.*.json` files
            max_files: if provided, limit to the first N files (sorted by `lo`)
            min_playlist_length: drop playlists with fewer than this many tracks

        Returns:
            dict with keys:
              - "playlists":         List[List[int]]
              - "num_unique_tracks": int
              - "uri_to_id":         dict[str, int]
        """
        files = sorted(
            glob.glob(str(Path(raw_dir) / "mpd.slice.*.json")),
            key=_slice_sort_key,
        )
        if not files:
            raise FileNotFoundError(
                f"No mpd.slice.*.json files found in {raw_dir!r}"
            )
        if max_files is not None:
            files = files[:max_files]

        uri_to_id: Dict[str, int] = {}
        playlists: List[List[int]] = []

        for filepath in files:
            with open(filepath) as f:
                data = json.load(f)
            for playlist in data["playlists"]:
                track_ids: List[int] = []
                for track in playlist["tracks"]:
                    uri = track["track_uri"]
                    tid = uri_to_id.get(uri)
                    if tid is None:
                        tid = len(uri_to_id)
                        uri_to_id[uri] = tid
                    track_ids.append(tid)
                if len(track_ids) >= min_playlist_length:
                    playlists.append(track_ids)

        return {
            "playlists": playlists,
            "num_unique_tracks": len(uri_to_id),
            "uri_to_id": uri_to_id,
        }
