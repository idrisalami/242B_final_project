"""Spotify Web API helpers for the Stage 4 app."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dependency may be absent in minimal test envs
    load_dotenv = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:  # pragma: no cover
    spotipy = None
    SpotifyClientCredentials = None

SPOTIFY_URI_RE = re.compile(r"^spotify:track:[A-Za-z0-9]+$")


def load_stage4_env() -> None:
    """Load project and Stage 4 dotenv files if python-dotenv is installed."""
    if load_dotenv is None:
        return
    stage4_dir = Path(__file__).resolve().parent
    project_root = stage4_dir.parent
    load_dotenv(project_root / ".env")
    load_dotenv(stage4_dir / ".env")


def get_spotify_client():
    """Create a Spotipy client from environment credentials, or return None."""
    load_stage4_env()
    if spotipy is None or SpotifyClientCredentials is None:
        return None
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
    )


def parse_playlist_id(value: str) -> str:
    """Extract a Spotify playlist ID from a URL, URI, or bare ID."""
    text = value.strip()
    if not text:
        raise ValueError("Playlist input is empty.")

    if text.startswith("spotify:playlist:"):
        playlist_id = text.split(":")[-1]
    elif "open.spotify.com/playlist/" in text:
        playlist_id = text.split("open.spotify.com/playlist/", 1)[1].split("?", 1)[0].split("/", 1)[0]
    else:
        playlist_id = text

    if not re.fullmatch(r"[A-Za-z0-9]+", playlist_id):
        raise ValueError(f"Could not parse Spotify playlist ID from: {value}")
    return playlist_id


def parse_track_uris(text: str) -> List[str]:
    """Parse Spotify track URIs from free-form pasted text."""
    uris: List[str] = []
    seen = set()
    for token in re.split(r"[\s,]+", text.strip()):
        token = token.strip()
        if not token:
            continue
        if "open.spotify.com/track/" in token:
            track_id = token.split("open.spotify.com/track/", 1)[1].split("?", 1)[0].split("/", 1)[0]
            token = f"spotify:track:{track_id}"
        if SPOTIFY_URI_RE.fullmatch(token) and token not in seen:
            uris.append(token)
            seen.add(token)
    return uris


def _track_to_meta(track: dict) -> Optional[dict]:
    if not track:
        return None
    images = track.get("album", {}).get("images", [])
    artists = track.get("artists", [])
    uri = track.get("uri")
    if not uri:
        return None
    return {
        "uri": uri,
        "name": track.get("name", "Unknown track"),
        "artist": ", ".join(a.get("name", "Unknown artist") for a in artists) or "Unknown artist",
        "album": track.get("album", {}).get("name", ""),
        "album_art": images[0]["url"] if images else None,
        "spotify_url": track.get("external_urls", {}).get("spotify", ""),
    }


def _fallback_track_meta(uri: str) -> dict:
    return {
        "uri": uri,
        "name": uri.split(":")[-1],
        "artist": "Metadata unavailable",
        "album": "",
        "album_art": None,
        "spotify_url": f"https://open.spotify.com/track/{uri.split(':')[-1]}",
    }


def fetch_playlist_tracks(sp, playlist_url_or_id: str) -> List[dict]:
    """Fetch every track in a Spotify playlist, following pagination."""
    playlist_id = parse_playlist_id(playlist_url_or_id)
    results = sp.playlist_items(
        playlist_id,
        additional_types=("track",),
        fields="items(track(uri,name,artists(name),album(name,images),external_urls)),next",
        limit=100,
    )
    tracks: List[dict] = []
    while results:
        for item in results.get("items", []):
            meta = _track_to_meta(item.get("track"))
            if meta:
                tracks.append(meta)
        next_url = results.get("next")
        if not next_url:
            break
        results = sp.next(results)
    return tracks


def fetch_track_metadata(sp, uris: Iterable[str]) -> Dict[str, dict]:
    """Batch-fetch metadata for Spotify track URIs."""
    unique = []
    seen = set()
    for uri in uris:
        if uri not in seen:
            unique.append(uri)
            seen.add(uri)

    metadata: Dict[str, dict] = {}
    if sp is None:
        for uri in unique:
            metadata[uri] = _fallback_track_meta(uri)
        return metadata

    for start in range(0, len(unique), 50):
        batch = unique[start:start + 50]
        ids = [uri.split(":")[-1] for uri in batch]
        try:
            response = sp.tracks(ids)
        except Exception:
            for uri in batch:
                metadata[uri] = _fallback_track_meta(uri)
            continue
        for track in response.get("tracks", []):
            meta = _track_to_meta(track)
            if meta:
                metadata[meta["uri"]] = meta
        for uri in batch:
            metadata.setdefault(uri, _fallback_track_meta(uri))
    return metadata


def resolve_track_aliases(sp, uris: Sequence[str], market: str = "US") -> Dict[str, List[str]]:
    """Return Spotify relink aliases for input URIs.

    MPD was built from older Spotify track IDs. When Spotify relinks a track,
    the pasted/current URI may differ from the older canonical URI in MPD.
    """
    unique = []
    seen = set()
    for uri in uris:
        if uri not in seen:
            unique.append(uri)
            seen.add(uri)

    aliases: Dict[str, List[str]] = {}
    if sp is None:
        return aliases

    for start in range(0, len(unique), 50):
        batch = unique[start:start + 50]
        ids = [uri.split(":")[-1] for uri in batch]
        try:
            response = sp.tracks(ids, market=market)
        except Exception:
            continue
        for requested_uri, track in zip(batch, response.get("tracks", [])):
            if not track:
                continue
            candidates = []
            current_uri = track.get("uri")
            linked_uri = track.get("linked_from", {}).get("uri")
            for candidate in (current_uri, linked_uri):
                if candidate and candidate != requested_uri and SPOTIFY_URI_RE.fullmatch(candidate):
                    candidates.append(candidate)
            if candidates:
                aliases[requested_uri] = candidates
    return aliases
