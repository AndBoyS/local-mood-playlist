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
  cannot express it; explain that guarantee nearby. In a runtime-evaluated
  position a `cast` cannot carry a shape subscript; a deferred local
  annotation (`x: Tensor[[...]] = ...` under `from __future__ import
  annotations`) is the runtime-neutral equivalent for re-establishing shapes.

## Tensor shape types

`mert_model.py` (and any future inference-time model code) is annotated with
Pyrefly's tensor shape types. The shape machinery is dev-only:

- `pyrefly-torch-stubs` (dev dep) installs the shape-aware torch overlay and
  pulls `pyrefly-shape-extensions`; shape tracking activates automatically
  once pyrefly resolves `shape_extensions`.
- Runtime code imports shape symbols only under `TYPE_CHECKING` and relies on
  `from __future__ import annotations`, so the published package never needs
  them (PEP 695 type-param bounds are also unevaluated at import).
- Known constraint: `Module.forward` loops and per-instance kernel/stride
  variants cannot all be tracked statically; where the shape is a pure
  function of pinned config constants, re-establish it with one documented
  annotation fallback at the boundary (see `_FeatureEncoder.forward`).
- Shape-typed `forward` overrides pass strict's override check as of
  pyrefly 1.4.0 ([fixed][pyrefly-140]); no diagnostic suppression is needed.
  The port itself carries zero in-code ignores; the repo-wide 8 suppressed
  diagnostics are one deliberate ignore in `config.py` plus stub-level
  suppressions pyrefly does not itemize.

[pyrefly-140]: https://github.com/facebook/pyrefly/releases/tag/1.4.0

## Handling checker errors

1. Decide whether the code violates its declared contract. If so, fix the code
   or validate/narrow at the correct boundary.
2. If inference is unclear, use `reveal_type(expr)` temporarily, then remove it.
3. Prefer a standard typed method, then a small adapter around an untyped API.
4. If valid code hits a known stub gap, use the narrowest
   `# pyrefly: ignore[<error-code>]` and explain why. Never use blanket ignores
   or duplicate suppressions through a call chain.
