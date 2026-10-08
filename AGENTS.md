# AGENTS.md

Guidelines for coding agents working in this repo.

## Environment

- Python 3.13, managed with uv. Run Python through `uv run ...`, never bare
  `python` or `pip`.
- Lint: `uv run ruff check .`.
- Types: `uv run pyrefly check` — strict preset. See [`docs/typing.md`](docs/typing.md).

## Current scope

- CLI exposes a fixed mood vocabulary through `sp moods` and
  `sp create MOOD [MOOD ...]`; `sp create --dry-run` with no labels opens a
  flat Questionary multi-select picker in a terminal.
- `sp index DIRECTORY` runs vendored Music2Emotion in separate Python 3.10
  environment and caches canonical tag scores. Playlist ranking and writing
  are not implemented; `sp create --dry-run` previews only.
- Keep mood labels mapped to the model's canonical tag names. User-facing
  aliases belong at the CLI boundary.
- Keep CLI parsing and validation independent from future model dependencies.

## Conventions

- Use `pathlib` for paths.
- Type public functions and methods; prefer simple useful contracts over
  unnecessary abstractions.
- Validate values at input boundaries. Prefer `object` over `Any` for unknown
  data; use `Any` only for inherently dynamic APIs.
