"""Fixed mood vocabulary."""

from typing import Final, Literal, NamedTuple, get_args

MoodName = Literal[
    "calm",
    "dark",
    "dreamy",
    "emotional",
    "energetic",
    "happy",
    "hopeful",
    "melancholic",
    "meditative",
    "positive",
    "relaxing",
    "romantic",
    "sad",
    "sexy",
    "uplifting",
    "cool",
    "deep",
    "epic",
    "fun",
    "funny",
    "groovy",
    "heavy",
    "inspiring",
    "love",
    "motivational",
    "party",
    "powerful",
    "soft",
    "upbeat",
    "action",
    "adventure",
    "advertising",
    "background",
    "ballad",
    "children",
    "christmas",
    "commercial",
    "corporate",
    "documentary",
    "drama",
    "dramatic",
    "fast",
    "film",
    "game",
    "holiday",
    "melodic",
    "movie",
    "nature",
    "retro",
    "slow",
    "soundscape",
    "space",
    "sport",
    "summer",
    "trailer",
    "travel",
]

MOOD_NAMES: Final[frozenset[MoodName]] = frozenset(get_args(MoodName))

BlendMode = Literal["mean", "and", "or"]
"""How selected mood probabilities combine into one track score.

Declaration order is the blend-mode cycle order used by the interactive
picker's ``m`` key: ``mean`` averages the scores, ``and`` keeps tracks
matching every selected mood (minimum score), ``or`` accepts any selected
mood (maximum score).
"""


class BlendModeDetail(NamedTuple):
    """User-facing wording describing a blend mode's score and threshold."""

    matching: str
    threshold_label: str


BLEND_MODE_DETAIL: Final[dict[BlendMode, BlendModeDetail]] = {
    "mean": BlendModeDetail("mean of selected mood probabilities", "mean mood probability"),
    "and": BlendModeDetail(
        "minimum of selected mood probabilities (every mood matched)", "minimum (weakest) mood probability"
    ),
    "or": BlendModeDetail(
        "maximum of selected mood probabilities (any mood matched)", "maximum (strongest) mood probability"
    ),
}
