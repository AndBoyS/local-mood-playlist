"""Checkbox mood picker with a live-blend-mode toggle key."""

from collections.abc import Sequence
from typing import Final, NamedTuple, get_args

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from questionary import Choice
from questionary.constants import DEFAULT_QUESTION_PREFIX, DEFAULT_SELECTED_POINTER, INVALID_INPUT
from questionary.prompts import common
from questionary.question import Question
from questionary.styles import merge_styles_default

from local_mood_playlist.moods import BlendMode, MoodName

_MODE_CYCLE: Final[tuple[BlendMode, ...]] = get_args(BlendMode)

_INSTRUCTION: Final[str] = "(arrows/jk move, Space toggle, <m> blend mode, Enter confirm)"

_MODE_STYLE: Final = Style(
    [
        ("mode-mean", "fg:#8db9d8 bold"),
        ("mode-and", "fg:#d99a6c bold"),
        ("mode-or", "fg:#d6a0c2 bold"),
    ]
)


class MoodSelection(NamedTuple):
    """Moods chosen in the picker and the blend mode active when confirmed."""

    moods: list[MoodName]
    mode: BlendMode


def _next_mode(mode: BlendMode) -> BlendMode:
    return _MODE_CYCLE[(_MODE_CYCLE.index(mode) + 1) % len(_MODE_CYCLE)]


MoodAndColor = tuple[MoodName, str]


def ask_moods_with_mode(
    entries: Sequence[MoodAndColor],
    *,
    message: str,
    style: Style,
) -> MoodSelection | None:
    """Show a mood checkbox; ``m`` cycles mean/and/or with a colored mode badge.

    ``entries`` are ``(mood, title_color)`` pairs. Returns the selection, or
    ``None`` on cancel.
    """
    mode: BlendMode = "mean"
    choices = [Choice(title=[(f"fg:{color}", mood)], value=mood) for mood, color in entries]
    ic = common.InquirerControl(choices, pointer=DEFAULT_SELECTED_POINTER)
    merged_style = merge_styles_default(
        [Style([("bottom-toolbar", "noreverse")]), style, _MODE_STYLE],
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = [
            ("class:qmark", DEFAULT_QUESTION_PREFIX),
            ("class:question", f" {message} "),
        ]
        if ic.is_answered:
            selected = len(ic.selected_options)
            tokens.append(("class:answer", "done" if selected == 0 else f"done ({selected} selections)"))
            tokens.append((f"class:mode-{mode}", f" [{mode}]"))
            return tokens
        tokens.append((f"class:mode-{mode}", f"[{mode}] "))
        tokens.append(("class:instruction", _INSTRUCTION))
        return tokens

    def get_selected_values() -> list[object]:
        return [choice.value for choice in ic.get_selected_values()]

    def perform_validation(selected_values: list[object]) -> bool:
        if len(selected_values) == 0 and ic.submission_attempted:
            ic.error_message = FormattedText([("class:validation-toolbar", INVALID_INPUT)])
        else:
            ic.error_message = None
        return len(selected_values) > 0 or not ic.submission_attempted

    layout = common.create_inquirer_layout(ic, get_prompt_tokens)

    bindings = KeyBindings()

    @bindings.add(Keys.ControlC, eager=True)
    def abort(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(" ", eager=True)
    def toggle(_event: KeyPressEvent) -> None:
        pointed_choice = ic.get_pointed_at().value
        if pointed_choice in ic.selected_options:
            ic.selected_options.remove(pointed_choice)
        else:
            ic.selected_options.append(pointed_choice)
        _ = perform_validation(get_selected_values())

    @bindings.add("m", eager=True)
    def cycle_mode(_event: KeyPressEvent) -> None:
        nonlocal mode
        mode = _next_mode(mode)

    @bindings.add(Keys.Down, eager=True)
    @bindings.add("j", eager=True)
    @bindings.add(Keys.ControlN, eager=True)
    def move_cursor_down(_event: KeyPressEvent) -> None:
        ic.select_next()
        while not ic.is_selection_valid():
            ic.select_next()

    @bindings.add(Keys.Up, eager=True)
    @bindings.add("k", eager=True)
    @bindings.add(Keys.ControlP, eager=True)
    def move_cursor_up(_event: KeyPressEvent) -> None:
        ic.select_previous()
        while not ic.is_selection_valid():
            ic.select_previous()

    @bindings.add(Keys.ControlM, eager=True)
    def set_answer(event: KeyPressEvent) -> None:
        selected_values = get_selected_values()
        ic.submission_attempted = True
        if perform_validation(selected_values):
            ic.is_answered = True
            event.app.exit(result=selected_values)

    @bindings.add(Keys.Any)
    def other(_event: KeyPressEvent) -> None:
        """Disallow inserting other text."""

    question = Question(Application(layout=layout, key_bindings=bindings, style=merged_style))
    # Suppress questionary's own "Cancelled by user"; the caller owns cancel messaging.
    if question.ask(kbi_msg="") is None:
        return None
    # questionary surfaces selection values as `Any`; rebuild the answer from the
    # typed entries instead of trusting that pipeline. The choice values were set
    # from `entries` above and only our own key bindings write `selected_options`.
    selected: set[object] = set(ic.selected_options)
    moods = [mood for mood, _ in entries if mood in selected]
    return MoodSelection(moods=moods, mode=mode)
