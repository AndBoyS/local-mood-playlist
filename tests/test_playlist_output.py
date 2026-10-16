import sys
from pathlib import Path

import pytest

from local_mood_playlist.main import _resolve_playlist_output
from local_mood_playlist.paths import ResolvedPath


def test_m3u8_path_used_as_is() -> None:
    resolved = _resolve_playlist_output("lists/list.m3u8", ["calm"])
    assert resolved == ResolvedPath(Path("lists/list.m3u8"))


def test_m3u8_suffix_match_is_case_insensitive() -> None:
    resolved = _resolve_playlist_output("lists/LIST.M3U8", ["calm"])
    assert resolved == ResolvedPath(Path("lists/LIST.M3U8"))


def test_directory_receives_moods_filename() -> None:
    resolved = _resolve_playlist_output("lists", ["calm", "dreamy"])
    assert resolved == ResolvedPath(Path("lists") / "calm-dreamy.m3u8")


def test_directory_receives_non_default_mode_and_threshold() -> None:
    resolved = _resolve_playlist_output("lists", ["calm", "dreamy"], mode="or", threshold=0.75)
    assert resolved == ResolvedPath(Path("lists") / "calm-dreamy-or-t0.75.m3u8")


def test_directory_keeps_default_parameters_out_of_filename() -> None:
    resolved = _resolve_playlist_output("lists", ["calm"], mode="and", threshold=0.5)
    assert resolved == ResolvedPath(Path("lists") / "calm.m3u8")


def test_directory_with_trailing_slash() -> None:
    resolved = _resolve_playlist_output("lists/", ["calm"])
    assert resolved == ResolvedPath(Path("lists") / "calm.m3u8")


def test_moods_placeholder_expands_to_file() -> None:
    resolved = _resolve_playlist_output("lists/{moods}.m3u8", ["calm", "dreamy"])
    assert resolved == ResolvedPath(Path("lists/calm-dreamy.m3u8"))


def test_moods_placeholder_names_the_file() -> None:
    resolved = _resolve_playlist_output("lists/{moods}", ["calm", "dreamy"])
    assert resolved == ResolvedPath(Path("lists/calm-dreamy.m3u8"))


def test_bare_moods_placeholder_becomes_filename() -> None:
    resolved = _resolve_playlist_output("{moods}", ["calm"])
    assert resolved == ResolvedPath(Path("calm.m3u8"))


def test_moods_placeholder_keeps_existing_suffix() -> None:
    resolved = _resolve_playlist_output("lists/{moods}.M3U8", ["calm"])
    assert resolved == ResolvedPath(Path("lists/calm.M3U8"))


def test_moods_placeholder_with_dotted_basename_appends_not_replaces() -> None:
    resolved = _resolve_playlist_output("lists/v1.2-{moods}", ["calm"])
    assert resolved == ResolvedPath(Path("lists/v1.2-calm.m3u8"))


def test_mode_pasted_into_moods_expansion_when_non_default() -> None:
    resolved = _resolve_playlist_output("lists/{moods}", ["calm"], mode="or")
    assert resolved == ResolvedPath(Path("lists/calm-or.m3u8"))


def test_explicit_default_parameters_not_pasted_into_moods_expansion() -> None:
    resolved = _resolve_playlist_output("{moods}", ["calm"], mode="and", threshold=0.5)
    assert resolved == ResolvedPath(Path("calm.m3u8"))


def test_threshold_pasted_into_moods_expansion_when_non_default() -> None:
    resolved = _resolve_playlist_output("{moods}", ["calm"], threshold=1 / 3)
    assert resolved == ResolvedPath(Path("calm-t0.33.m3u8"))
    resolved = _resolve_playlist_output("{moods}", ["calm"], threshold=0.3)
    assert resolved == ResolvedPath(Path("calm-t0.30.m3u8"))


def test_mode_and_threshold_both_pasted_into_moods_expansion() -> None:
    resolved = _resolve_playlist_output("{moods}", ["calm", "happy"], mode="mean", threshold=0.75)
    assert resolved == ResolvedPath(Path("calm-happy-mean-t0.75.m3u8"))


@pytest.mark.expects_stderr
def test_bare_drive_letter_warns_only_on_windows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    resolved = _resolve_playlist_output("C:", ["calm"])
    assert resolved == ResolvedPath(Path("C:") / "calm.m3u8")
    assert "bare drive letter" in capsys.readouterr().err


def test_bare_drive_letter_is_normal_directory_on_posix() -> None:
    resolved = _resolve_playlist_output("C:", ["calm"])
    assert resolved == ResolvedPath(Path("C:") / "calm.m3u8")
