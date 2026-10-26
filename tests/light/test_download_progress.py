from local_mood_playlist.assets import AssetDownloadProgress
from local_mood_playlist.main import _print_download_pbar

MB = 1 << 20


def test_phase_lines_from_yielded_transitions() -> None:
    """Yields happen only at file start and file complete; the consumer
    renders exactly those two transitions, branchable off state alone.
    Per-chunk byte progress is deliberately unobserved.
    """
    start = AssetDownloadProgress(
        files_processed=0,
        files_total=2,
        current_file_name="btc_model_large_voca.safetensors",
        done_bytes=0,
        file_complete=False,
    )
    assert _print_download_pbar(start) == "Downloading 1/2 btc_model_large_voca.safetensors..."
    complete = AssetDownloadProgress(
        files_processed=0,
        files_total=2,
        current_file_name="btc_model_large_voca.safetensors",
        done_bytes=100 * MB,
        file_complete=True,
    )
    assert _print_download_pbar(complete) == "Downloaded 1/2 btc_model_large_voca.safetensors (100.0 MB)"

    second_start = AssetDownloadProgress(
        files_processed=1,
        files_total=2,
        current_file_name="model.safetensors.zst",
        done_bytes=0,
        file_complete=False,
    )
    assert _print_download_pbar(second_start) == "Downloading 2/2 model.safetensors.zst..."
    second_complete = AssetDownloadProgress(
        files_processed=1,
        files_total=2,
        current_file_name="model.safetensors.zst",
        done_bytes=125 * MB,
        file_complete=True,
    )
    assert _print_download_pbar(second_complete) == "Downloaded 2/2 model.safetensors.zst (125.0 MB)"


def test_zero_length_file_renders_zero_mb() -> None:
    start = AssetDownloadProgress(
        files_processed=0,
        files_total=1,
        current_file_name="empty.bin",
        done_bytes=0,
        file_complete=False,
    )
    assert _print_download_pbar(start) == "Downloading 1/1 empty.bin..."
    complete = AssetDownloadProgress(
        files_processed=0,
        files_total=1,
        current_file_name="empty.bin",
        done_bytes=0,
        file_complete=True,
    )
    assert _print_download_pbar(complete) == "Downloaded 1/1 empty.bin (0.0 MB)"
