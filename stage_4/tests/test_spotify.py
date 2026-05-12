import pytest

from stage_4.spotify import fetch_track_metadata, parse_playlist_id, parse_track_uris, resolve_track_aliases


def test_parse_playlist_id_from_url():
    assert (
        parse_playlist_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc")
        == "37i9dQZF1DXcBWIGoYBM5M"
    )


def test_parse_playlist_id_from_uri():
    assert parse_playlist_id("spotify:playlist:abc123") == "abc123"


def test_parse_playlist_id_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_playlist_id("https://open.spotify.com/playlist/not valid")


def test_parse_track_uris_from_mixed_text():
    text = """
    spotify:track:abc123
    https://open.spotify.com/track/def456?si=1
    spotify:track:abc123
    not-a-track
    """
    assert parse_track_uris(text) == [
        "spotify:track:abc123",
        "spotify:track:def456",
    ]


class FakeSpotify:
    def tracks(self, ids, market="US"):
        assert market == "US"
        return {
            "tracks": [
                {
                    "uri": "spotify:track:new123",
                    "linked_from": {"uri": "spotify:track:old123"},
                }
            ]
        }


def test_resolve_track_aliases_from_linked_track():
    aliases = resolve_track_aliases(FakeSpotify(), ["spotify:track:new123"])

    assert aliases == {"spotify:track:new123": ["spotify:track:old123"]}


class BrokenSpotify:
    def tracks(self, ids, market=None):
        raise RuntimeError("spotify unavailable")


def test_track_metadata_falls_back_when_spotify_lookup_fails():
    metadata = fetch_track_metadata(BrokenSpotify(), ["spotify:track:abc123"])

    assert metadata["spotify:track:abc123"]["artist"] == "Metadata unavailable"


def test_resolve_track_aliases_ignores_spotify_lookup_failures():
    assert resolve_track_aliases(BrokenSpotify(), ["spotify:track:new123"]) == {}
