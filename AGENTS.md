# AGENTS.md

## Environment

- Python 3.13, managed with uv. Run Python through `uv run ...`, never bare
  `python` or `pip`.
- Lint: `uv run ruff check --fix` (fix is safe, you can you use it every time).
- Types: `uv run pyrefly check` — strict preset. See [`docs/typing.md`](docs/typing.md).

## Current scope

- CLI exposes a fixed mood vocabulary through `moodpl moods` and
  `moodpl create MOOD [MOOD ...]`; `moodpl create` ranks cached predictions by the
  minimum probability of selected moods and writes M3U8 output to `--output` or
  stdout; `{moods}` in `--output`/`output` names the playlist file (`.m3u8`
  appended when missing), a value ending in `.m3u8` is used as-is, anything
  else is a directory receiving `<moods>.m3u8`.
  With a cache, `--dry-run` previews ranked tracks without writing;
  without a cache, it previews the request only. With no labels in a terminal,
  InquirerPy opens a multi-select picker with a blend-mode toggle key.
- `moodpl index DIRECTORY` runs vendored Music2Emotion in-process in the main
  Python 3.13 environment and caches canonical tag scores. `moodpl create` reads
  fresh cached predictions; it does not run model inference.
- Keep mood labels mapped to the model's canonical tag names. User-facing
  aliases belong at the CLI boundary.
- Right now alpha - don't need backwards compat when changing the contract

## Conventions

- Use `pathlib` for paths.
- Type public functions and methods; prefer simple useful contracts over
  unnecessary abstractions.
- Validate values at input boundaries. Prefer `object` over `Any` for unknown
  data; use `Any` only for inherently dynamic APIs.
- Don't do mock tests
- Don't need to test things which are protected by design (e.g. that funcs operating on ResolvedPath are absolute and expanded)
- When asking for review, also point to check if tests are not redundant
- Use git naming conventions
- We don't do env vars as configs
- no walrus
