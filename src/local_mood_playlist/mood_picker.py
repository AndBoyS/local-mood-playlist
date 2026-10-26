"""Checkbox mood picker with a live-blend-mode toggle key (InquirerPy)."""

from collections.abc import Sequence
from typing import Final, NamedTuple, override

from InquirerPy.base.control import Choice
from InquirerPy.prompts.list import InquirerPyListControl, ListPrompt
from InquirerPy.utils import InquirerPyKeybindings, get_style
from prompt_toolkit.key_binding import KeyPressEvent

from local_mood_playlist.moods import BLEND_MODES, BlendMode, MoodName

_INSTRUCTION: Final[str] = "(↑↓/jk move, Space toggle, <m> blend mode, Enter confirm)"
_EMPTY_SELECTION_ERROR: Final[str] = "select at least one mood"

_POINTER: Final[str] = "›"
_MARKER: Final[str] = "◉"
_MARKER_PLACEHOLDER: Final[str] = "○"

PICKER_STYLE: Final = get_style(
    {
        "questionmark": "bold",
        "question": "bold",
        "pointer": "fg:ansicyan bold",
        "marker": "fg:ansigreen",
        "instruction": "fg:ansibrightblack italic",
        "mode-mean": "fg:#8db9d8 bold",
        "mode-and": "fg:#d99a6c bold",
        "mode-or": "fg:#d6a0c2 bold",
    }
)

MOVE_KEYBINDINGS: Final[InquirerPyKeybindings] = {
    "down": [{"key": "down"}, {"key": "c-n"}, {"key": "j"}],
    "up": [{"key": "up"}, {"key": "c-p"}, {"key": "k"}],
    "skip": [],
}


class MoodSelection(NamedTuple):
    """Moods chosen in the picker and the blend mode active when confirmed."""

    moods: list[MoodName]
    mode: BlendMode


def _mood_color_name_enabled(choice: dict[str, object]) -> tuple[str, str, bool]:
    value = choice["value"]
    name = choice["name"]
    enabled = choice["enabled"]
    if not isinstance(value, str) or not isinstance(name, str) or not isinstance(enabled, bool):
        raise TypeError(f"InquirerPy choice has an unexpected shape: {choice!r}")
    return value, name, enabled


class MoodListControl(InquirerPyListControl):
    """List control rendering each mood in its family color."""

    def __init__(self, *, colors: dict[str, str], choices: list[Choice]) -> None:
        self._colors = colors
        super().__init__(
            choices=choices,
            default=None,
            pointer=_POINTER,
            marker=_MARKER,
            session_result=None,
            multiselect=True,
            marker_pl=_MARKER_PLACEHOLDER,
        )

    @override
    def _get_hover_text(self, choice: dict[str, object]) -> list[tuple[str, str]]:
        mood, name, enabled = _mood_color_name_enabled(choice)
        color = self._colors[mood]
        return [
            ("class:pointer", self._pointer),
            (f"bold fg:{color}", self._marker if enabled else self._marker_pl),
            ("[SetCursorPosition]", ""),
            (f"bold fg:{color}", name),
        ]

    @override
    def _get_normal_text(self, choice: dict[str, object]) -> list[tuple[str, str]]:
        mood, name, enabled = _mood_color_name_enabled(choice)
        color = self._colors[mood]
        return [
            ("", " " * len(self._pointer)),
            ("class:marker", self._marker if enabled else self._marker_pl),
            (f"fg:{color}", name),
        ]


class MoodPrompt(ListPrompt):
    """Multiselect mood prompt whose ``m`` key cycles the blend mode badge."""

    def __init__(self, *, message: str, choices: list[Choice], colors: dict[str, str], default_mode: BlendMode) -> None:
        self._mode = default_mode
        self.content_control = MoodListControl(colors=colors, choices=choices)
        super().__init__(
            message=message,
            choices=choices,
            multiselect=True,
            instruction=_INSTRUCTION,
            style=PICKER_STYLE,
            cycle=False,
            validate=lambda vals: isinstance(vals, list) and len(vals) > 0,
            invalid_message=_EMPTY_SELECTION_ERROR,
            keybindings=MOVE_KEYBINDINGS,
        )
        _ = self.register_kb("m")(self._cycle_mode)

    @override
    def _get_prompt_message(self) -> list[tuple[str, str]]:
        message = super()._get_prompt_message()
        message.insert(len(message) - 1, (f"class:mode-{self._mode}", f" [{self._mode}]"))
        return message

    def _cycle_mode(self, _event: KeyPressEvent) -> None:
        self._mode = _next_mode(self._mode)


def _next_mode(mode: BlendMode) -> BlendMode:
    return BLEND_MODES[(BLEND_MODES.index(mode) + 1) % len(BLEND_MODES)]


MoodAndColor = tuple[MoodName, str]


def ask_moods_with_mode(
    entries: Sequence[MoodAndColor],
    *,
    message: str,
    default_mode: BlendMode,
) -> MoodSelection | None:
    """Show a mood multiselect; ``m`` cycles and/or/mean with a colored mode badge.

    ``entries`` are ``(mood, title_color)`` pairs. The mode badge starts at
    ``default_mode``. Returns the selection, or ``None`` on cancel.
    """
    prompt = MoodPrompt(
        message=message,
        choices=[Choice(value=mood) for mood, _ in entries],
        colors={mood: color for mood, color in entries},
        default_mode=default_mode,
    )
    try:
        result = prompt.execute()
    except KeyboardInterrupt:
        return None
    if not isinstance(result, list):
        raise TypeError(f"InquirerPy returned an unexpected picker result: {result!r}")
    moods = [mood for mood, _ in entries if mood in result]
    return MoodSelection(moods=moods, mode=prompt._mode)
