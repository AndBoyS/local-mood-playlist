# local-smart-playlist

CLI shell for local mood playlists. It exposes one unified, mood-ordered label
list and can index audio with vendored Music2Emotion, caching raw mood scores.
Playlist ranking and playlist writing are not wired yet.

## Setup

```console
uv sync
uv run sp moods
uv run sp create --dry-run
uv run sp index /path/to/music
```

Multiple positional labels blend. With no labels, `sp create --dry-run` opens a flat,
multi-select picker; related labels appear near each other. Muted pastel colors
give a rough visual cue: blue for subdued/reflective, pink for affectionate,
yellow for bright, orange for active, and red for intense/action-like. Colors
are UI heuristics, not model output or strict categories. Non-interactive use
requires explicit labels.

## Commands

- `sp moods` — list selectable labels in a flat, mood-ordered list.
- `sp index DIRECTORY [--cache PATH] [--force]` — predict missing or changed
  `.mp3`, `.m4a`, `.wav`, and `.flac` files and cache raw canonical mood scores.
- `sp create MOOD [MOOD ...] [-n COUNT] [-o PATH] [--dry-run]` — validate and
  preview playlist request. Multiple labels blend by default.
- `sp --help` — show CLI help.

Music2Emotion runs in-process using the main Python 3.13 environment and
PyTorch/MPS when available, else CPU. Set `SP_MUSIC2EMOTION_DEVICE` to `cpu`,
`mps`, `cuda`, or `auto` to override. First inference downloads pinned MERT
weights from Hugging Face and may take time. M4A files require FFmpeg on
`PATH`. MERT is licensed CC BY-NC 4.0, so this integration is limited to
non-commercial use unless you obtain separate permission.

Predictions cache under the OS user cache directory by default. Cache records
invalidate when audio metadata or model version changes. Cache is a local pickle;
never load or share one from an untrusted source.

`sp create` still requires `--dry-run`: it previews only and does not scan,
rank, or write playlists. Ranking will consume cached scores in a later step.
Picker uses Questionary; arrows navigate, Space toggles, Enter confirms, and
Ctrl-C cancels.

