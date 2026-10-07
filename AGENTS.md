# AGENTS.md

Guidelines for coding agents working in this repo.

## Environment

- Python 3.13, managed with uv. Run everything through `uv run ...`, never
  bare `python`/`pip`.
- Lint: `uv run ruff check .`, config in `pyproject.toml`.
- Types: `uv run pyrefly check` — strict preset;
  See [`docs/typing.md`](docs/typing.md) for typing patterns and checker workflow.
- Tests: `uv run pytest` (model tests deselected by default; `uv run pytest
-m model` runs the MuQ-MuLan sanity suite locally, requires network).

## Conventions

- Paths via `pathlib`.
- Audio: mono float32, 48 kHz everywhere after `decode.decode_mono`.
- Track identity: POSIX relative path from library root (stable across
  re-indexes). Never store absolute paths in the DB.
- Playlist files: UTF-8 m3u8, `#EXTM3U` + `#EXTINF` headers, paths relative
  to the playlist file location.
- DB access only through `index/store.py`; schema lives there.
- Embedding math: vectors are L2-normalized before storage; aggregation
  helpers in `embed/aggregate.py` re-normalize after mean/p90.
- Python typing: choose the simplest type that preserves a useful contract.
  Prefer dataclasses for records, `TypedDict` for fixed dictionary payloads,
  `Protocol` for structural interfaces, and `Literal` for finite choices; use
  generics when they preserve a real input/output type relationship. Narrow
  dynamic values at boundaries with validation or `isinstance`; prefer
  `object` to `Any` for unknown values, reserving `Any` for inherently dynamic
  APIs. Do not contort runtime logic to satisfy a checker limitation. Use
  `reveal_type` to inspect inference; see `docs/typing.md` for NumPy shapes,
  casts, suppressions, and worked patterns.
- NumPy dtype: no gratuitous wrapping. `np.asarray(x, dtype=...)` only at
  real boundaries (embedder/torch/soxr output, public API args that may be
  lists). Between typed ndarrays use `.astype()` for casts and bare ops
  otherwise; dtype is guaranteed by the boundaries, not re-asserted at
  every intermediate step.
- NumPy shape types: preserve important shape contracts at function and data
  boundaries in the existing shape-typed modules. Do not add shape annotations
  to every local or propagate them through unsupported operations merely for
  completeness; use inference or a focused helper when that is clearer. Follow
  [`docs/typing.md`](docs/typing.md) for supported patterns and known stubs.
  Every Pyrefly suppression must name its error code, stay as narrow as
  possible, and explain non-obvious checker gaps. Remove stale suppressions;
  `unused-type-ignore = true` detects them.
- Keep torch/transformers imports inside `embed/` and `audio/features.py`;
  everything else stays pure numpy/sqlite so tests run without the model.
