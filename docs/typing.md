# Python typing guide

General typing guidance for Python 3.13 and Pyrefly. Keep signatures precise
and implementation straightforward.

## Everyday decisions

- Annotate public function and method parameters and return types.
- Let Pyrefly infer local variables by default. Add local annotations only to
  clarify ambiguous inference or document non-obvious intent.
- Choose the simplest type that preserves a useful contract:
  - `dataclass` for a stable record;
  - `TypedDict` for a fixed dictionary payload;
  - `Protocol` for behavior shared by unrelated implementations;
  - `Literal` for a finite set of valid options;
  - generics when an input/output type relationship matters.
- Represent absence explicitly with `T | None`; narrow before use.
- Validate or narrow dynamic values at input boundaries. Prefer `object` for
  unknown data; reserve `Any` for inherently dynamic APIs.
- Pass filesystem inputs at CLI/config/prompt boundaries through
  [`local_mood_playlist.paths`](../src/local_mood_playlist/paths.py).
  `ResolvedPath` canonicalizes on construction (``~`` expanded, absolute,
  symlinks resolved) — there is no way to build a non-canonical instance, so
  wrap raw strings in `Path(...)` at the boundary. Signatures carry
  `ResolvedPath` when canonical form is part of the contract (config paths,
  library roots, prediction-cache audio keys); functions that additionally
  need existence check for it with `.path` at their own boundary (e.g.
  `require_library_dir`, `index_library`). M3U8 playlist entries are
  `RelativePath` (relative to the playlist file's own directory, POSIX form),
  so the playlist and the library can move together to another device — the
  playlist may live anywhere as long as the layout relationship is kept
  (Windows: different drive from the library yields absolute entries).
  This is boundary guidance, not a blanket ban on `pathlib.Path`: internal
  plumbing (`filesystem_utils`, derived config/cache paths), vendored code,
  and dev scripts take plain `Path` when canonical form is not load-bearing.
- Avoid `typing.cast`: it does not validate values. Prefer `isinstance`, an
  assertion for a verified invariant, or a small typed adapter. Use a narrow
  cast only when an external API contract guarantees the type but the checker
  cannot express it; explain that guarantee nearby.

## Handling checker errors

1. Decide whether the code violates its declared contract. If so, fix the code
   or validate/narrow at the correct boundary.
2. If inference is unclear, use `reveal_type(expr)` temporarily, then remove it.
3. Prefer a standard typed method, then a small adapter around an untyped API.
4. If valid code hits a known stub gap, use the narrowest
   `# pyrefly: ignore[<error-code>]` and explain why. Never use blanket ignores
   or duplicate suppressions through a call chain.
