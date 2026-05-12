import pytest

from stage_4.spotify import parse_playlist_id, parse_track_uris


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
