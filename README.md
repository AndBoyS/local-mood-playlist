# local-mood-playlist

CLI for local mood playlists. It exposes one unified, mood-ordered label list,
indexes audio with vendored Music2Emotion, and creates M3U8 playlists from cached
mood scores.

## Setup

```console
uv sync
uv run moodpl moods
uv run moodpl index /path/to/music
uv run moodpl create calm dreamy --library /path/to/music -o playlists/calm-dreamy.m3u8
```

Multiple positional labels blend. With no labels, `moodpl create --dry-run` opens a flat,
multi-select picker; related labels appear near each other. Muted pastel colors
give a rough visual cue: blue for subdued/reflective, pink for affectionate,
yellow for bright, orange for active, and red for intense/action-like. Colors
are UI heuristics, not model output or strict categories. Non-interactive use
requires explicit labels.

## Commands

- `moodpl moods` — list selectable labels in a flat, mood-ordered list.
- `moodpl index DIRECTORY [--cache PATH] [--force]` — predict missing or changed
  `.mp3`, `.m4a`, `.wav`, and `.flac` files and cache raw canonical mood scores.
- `moodpl create MOOD [MOOD ...] [-n COUNT] [-o PATH] [--library DIRECTORY | --cache PATH] [--dry-run]` — rank fresh cached predictions by the mean probability of selected moods and emit an M3U8 playlist. With `-o`, track entries are relative to playlist location so playlist and library can move together while keeping their directory layout. Without `-o`, playlist text goes to stdout with absolute track paths.
- `moodpl --help` — show CLI help.

Music2Emotion runs in-process using the main Python 3.13 environment and
PyTorch/MPS when available, else CPU. Set `SP_MUSIC2EMOTION_DEVICE` to `cpu`,
`mps`, `cuda`, or `auto` to override. First inference downloads pinned MERT
weights from Hugging Face and may take time. M4A files require FFmpeg on
`PATH`. MERT is licensed CC BY-NC 4.0, so this integration is limited to
non-commercial use unless you obtain separate permission.

Predictions cache under the OS user cache directory by default. Cache records
invalidate when audio metadata or model version changes. Cache is a local pickle;
never load or share one from an untrusted source.

`moodpl create` consumes only fresh predictions already in cache; it does not run
inference. `--cache` selects the exact cache file, while `--library` selects its
default cache. With a cache, `--dry-run` shows ranked tracks without writing;
without one, it previews only the request. Multiple selected moods receive
equal weight. CLI label `dreamy` maps to model's canonical `dream` tag. Picker
uses Questionary; arrows navigate, Space toggles, Enter confirms, and Ctrl-C
cancels.

