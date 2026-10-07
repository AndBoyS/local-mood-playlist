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
- Playlist inference and audio processing are not implemented yet. `--dry-run`
  previews a request; do not claim it generated or wrote a playlist.
- Keep mood labels mapped to the model's canonical tag names. User-facing
  aliases belong at the CLI boundary.
- Keep CLI parsing and validation independent from future model dependencies.

## Conventions

- Use `pathlib` for paths.
- Type public functions and methods; prefer simple useful contracts over
  unnecessary abstractions.
- Validate values at input boundaries. Prefer `object` over `Any` for unknown
  data; use `Any` only for inherently dynamic APIs.
