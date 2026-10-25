from pathlib import Path

import pytest
from cli import IndexedLibrary

pytestmark = [pytest.mark.model]


def _track_entries(text: str) -> list[str]:
    return [line for line in text.splitlines() if line != "" and not line.startswith("#")]


def test_create_writes_playlist_from_indexed_cache(indexed_library: IndexedLibrary) -> None:
    context = indexed_library.context
    playlist = context.playlists_dir / "energetic.m3u8"
    result = context.run("create", "energetic", "--music-cache", str(indexed_library.music_cache), "-t", "0.1", "-o", str(playlist))
    assert result.exit_code == 0, result.stderr

    text = playlist.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "#EXTM3U"
    entries = _track_entries(text)
    assert len(entries) > 0, text
    track_names = {path.name for path in indexed_library.tracks}
    for entry in entries:
        entry_path = Path(entry)
        track = entry_path if entry_path.is_absolute() else context.playlists_dir / entry
        assert track.is_file(), entry
        assert track.name in track_names, track


def test_create_without_output_prints_absolute_entries(indexed_library: IndexedLibrary) -> None:
    result = indexed_library.context.run("create", "energetic", "--music-cache", str(indexed_library.music_cache), "-t", "0.1")
    assert result.exit_code == 0, result.stderr

    entries = _track_entries(result.stdout)
    assert len(entries) > 0, result.stdout
    assert all(Path(entry).is_absolute() for entry in entries)
    track_names = {path.name for path in indexed_library.tracks}
    assert {Path(entry).name for entry in entries} <= track_names
