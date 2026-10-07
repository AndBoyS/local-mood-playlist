# local-smart-playlist

CLI shell for creating local playlists from a fixed vocabulary of mood tags.
The interface shows one unified, mood-ordered label list; audio tagging,
playlist ranking, and playlist writing are not wired yet.

## Setup

```console
uv sync
uv run sp moods
uv run sp create --dry-run
uv run sp create dreamy calm -n 30 --dry-run
```

Multiple positional labels blend. With no labels, `sp create --dry-run` opens a flat,
multi-select picker; related labels appear near each other. Muted pastel colors
give a rough visual cue: blue for subdued/reflective, pink for affectionate,
yellow for bright, orange for active, and red for intense/action-like. Colors
are UI heuristics, not model output or strict categories. Non-interactive use
requires explicit labels.

## Commands

- `sp moods` — list selectable labels in a flat, mood-ordered list.
- `sp create MOOD [MOOD ...] [-n COUNT] [-o PATH] [--dry-run]` — validate and
  preview playlist request. Multiple labels blend by default.
- `sp --help` — show CLI help.

`sp create` requires `--dry-run` until a tag-prediction backend is connected. The
preview does not scan audio, rank tracks, or write a playlist. Picker uses
Questionary; arrows navigate, Space toggles, Enter confirms, and Ctrl-C cancels.

