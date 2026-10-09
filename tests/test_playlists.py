"""Tests for portable M3U playlist rendering."""

from pathlib import Path

from local_smart_playlist.playlists import RankedTrack, render_m3u8


def test_render_m3u8_paths_relative_to_output_playlist(tmp_path: Path) -> None:
    library = tmp_path / "Music"
    audio_path = library / "Artist" / "song.mp3"
    output_path = library / "Playlists" / "calm.m3u8"

    playlist = render_m3u8([RankedTrack(audio_path, 0.8)], output_path=output_path)

    entry = playlist.splitlines()[2]
    assert entry == "../Artist/song.mp3"
    assert (output_path.parent / entry).resolve() == audio_path.resolve()


def test_render_m3u8_paths_relative_across_sibling_trees(tmp_path: Path) -> None:
    audio_path = tmp_path / "Music" / "Artist" / "song.mp3"
    output_path = tmp_path / "Playlists" / "calm.m3u8"

    playlist = render_m3u8([RankedTrack(audio_path, 0.8)], output_path=output_path)

    entry = playlist.splitlines()[2]
    assert not Path(entry).is_absolute()
    assert (output_path.parent / entry).resolve() == audio_path.resolve()


def test_render_m3u8_stdout_paths_remain_absolute(tmp_path: Path) -> None:
    audio_path = tmp_path / "Music" / "song.mp3"

    playlist = render_m3u8([RankedTrack(audio_path, 0.8)])

    assert playlist.splitlines()[2] == audio_path.resolve().as_posix()
