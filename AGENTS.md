# AGENTS.md

## Environment

- Python 3.13, managed with uv. Run Python through `uv run ...`, never bare
  `python` or `pip`.
- Lint: `uv run ruff check --fix`.
- Types: `uv run pyrefly check` — strict preset. See [`docs/typing.md`](docs/typing.md).

## Current scope

- CLI exposes a fixed mood vocabulary through `moodpl moods` and
  `moodpl create MOOD [MOOD ...]`; `moodpl create` ranks cached predictions by the
  minimum probability of selected moods and writes M3U8 output to `--output` or
  stdout. With a cache, `--dry-run` previews ranked tracks without writing;
  without a cache, it previews the request only. With no labels in a terminal,
  Questionary opens a flat multi-select picker.
- `moodpl index DIRECTORY` runs vendored Music2Emotion in-process in the main
  Python 3.13 environment and caches canonical tag scores. `moodpl create` reads
  fresh cached predictions; it does not run model inference.
- Keep mood labels mapped to the model's canonical tag names. User-facing
  aliases belong at the CLI boundary.

## Conventions

- Use `pathlib` for paths.
- Type public functions and methods; prefer simple useful contracts over
  unnecessary abstractions.
- Validate values at input boundaries. Prefer `object` over `Any` for unknown
  data; use `Any` only for inherently dynamic APIs.
- Don't do mock tests
- Use git naming conventions
- We don't do env vars as configs
- no walrus
