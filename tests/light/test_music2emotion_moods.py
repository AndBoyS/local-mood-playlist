"""Alignment between the CLI mood vocabulary and the vendored model tag list.

Reads ``inference/data/tag_list.npy`` (see ``vendor/music2emo.py``) straight
from disk, so these tests never download model bundles, touch the network, or
import torch.
"""

import numpy as np
import pytest
from const import REPO_DIR

from local_mood_playlist.moods import MOOD_NAMES
from local_mood_playlist.music2emotion import TAG_PREFIX, to_cli_moods

TAG_LIST_PATH = REPO_DIR / "src/local_mood_playlist/vendor/inference/data/tag_list.npy"


def model_theme_tags() -> list[str]:
    """Every ``mood/theme---`` tag from the vendored tag list."""
    if not TAG_LIST_PATH.exists():
        pytest.skip(f"vendored tag list not present on disk: {TAG_LIST_PATH}", allow_module_level=True)
    loaded = np.load(TAG_LIST_PATH)
    return sorted(str(tag) for tag in loaded if str(tag).startswith(TAG_PREFIX))


def test_dream_override_maps_to_dreamy() -> None:
    assert to_cli_moods("mood/theme---dream") == "dreamy"


def test_theme_tags_map_onto_the_cli_vocabulary() -> None:
    resolved = {to_cli_moods(tag) for tag in model_theme_tags()}
    assert resolved == set(MOOD_NAMES)
