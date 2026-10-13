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
- `moodpl setup [--library DIRECTORY] [--cache-dir PATH] [-m MODE] [-e]` — write the
  user config interactively (prompts for library, cache dir, blend mode, and the
  extended-moods setting, each prefilled from the current config or flag; missing
  directories are re-prompted). Non-interactive use requires `--library`. An
  existing config is only overwritten after confirmation.
- `moodpl index [DIRECTORY] [--cache PATH] [--force]` — predict missing or changed
  `.mp3`, `.m4a`, `.wav`, and `.flac` files and cache raw canonical mood scores.
  Without `DIRECTORY`, the configured `library_dir` is used.
- `moodpl create MOOD [MOOD ...] [-n COUNT] [-o PATH] [-t SCORE] [-e] [--library DIRECTORY | --cache PATH] [--dry-run]` — rank fresh cached predictions by the minimum probability of selected moods and emit an M3U8 playlist. `-n`, `-t`, `-o`, and `-e` default to the config keys `count`, `threshold`, `output`, and `show_extended`. `-e` also offers extended model-only tags (e.g. `children`, `sport`, `film`, `game`, `trailer`) in the picker. With `-o`, track entries are relative to playlist location so playlist and library can move together while keeping their directory layout. Without `-o` (and with `output` unset), playlist text goes to stdout with absolute track paths. Note: config values have no per-invocation "unset" switch — a configured `count` can only be overridden to another size, not back to "all" (same for `show_extended` vs `-e`), so omit these keys from `config.toml` unless you want them as standing defaults.
- `moodpl --help` — show CLI help.

Music2Emotion runs in-process using the main Python 3.13 environment and
PyTorch/MPS when available, else CPU. Set `SP_MUSIC2EMOTION_DEVICE` to `cpu`,
`mps`, `cuda`, or `auto` to override. First inference downloads pinned MERT
weights from Hugging Face and may take time. M4A files require FFmpeg on
`PATH`. MERT is licensed CC BY-NC 4.0, so this integration is limited to
non-commercial use unless you obtain separate permission.

Predictions cache under `~/.cache/moodpl/predictions` by default, keyed by the
resolved library path. The user config file (`~/.config/moodpl/config.toml`,
created by `moodpl setup`) may set:

- `cache_dir` — cache root (default `~/.cache/moodpl`)
- `library_dir` — default music library directory
- `mode` — default blend mode: `and`, `or`, or `mean`
- `threshold` — default minimum blend score in (0, 1] (default `0.5`)
- `count` — default playlist size cap (default: all ranked tracks)
- `output` — default playlist destination; `{moods}` expands to selected labels
- `show_extended` — offer extended model-only mood tags in the picker by default

CLI flags keep precedence over the config file. Cache records
invalidate when audio metadata or model version changes. Cache is a local pickle;
never load or share one from an untrusted source.

`moodpl create` consumes only fresh predictions already in cache; it does not run
inference. `--cache` selects the exact cache file, while `--library` selects its
default cache; without either, the configured `library_dir` applies. With a
cache, `--dry-run` shows ranked tracks without writing;
without a usable cache (none selected, or the selected file does not exist
yet), it previews only the request. Multiple selected moods receive
equal weight. CLI label `dreamy` maps to model's canonical `dream` tag. Picker uses InquirerPy; arrows (or j/k) navigate, Space toggles, `m` cycles
blend mode, Enter confirms, and Ctrl-C cancels.

