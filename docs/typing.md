# Python typing guide

Project-specific typing patterns for Python 3.13 and Pyrefly. Keep function
and method signatures precise; keep implementation code straightforward.

## Everyday decisions

- Annotate every function and method parameter and return type.
- Let Pyrefly infer local variables by default. Add local annotations only to
  resolve ambiguous inference, document non-obvious intent, or preserve a
  useful NumPy shape.
- Choose the least complex type that protects a real contract:
  - `dataclass` for a stable record;
  - `TypedDict` for a fixed-shape dictionary crossing a boundary;
  - `Protocol` for behavior shared by unrelated implementations;
  - `Literal` for a finite set of valid options;
  - generics when a type relationship must survive a call (for example,
    input element type equals output element type).
- Represent absence explicitly with `T | None`; narrow before using the value.
- Validate or narrow dynamic data once at its boundary. Prefer `object` when
  accepting an unknown value; use `Any` only where an external or reflective
  API is inherently dynamic. Existing `verify_type` / `verify_not_none`
  helpers provide runtime-checked narrowing when appropriate.
- Avoid `typing.cast`: it returns the value unchanged and performs no runtime
  check. First try `isinstance`, an assertion for a verified invariant, or a
  small typed adapter. A localized cast is acceptable only when a third-party
  API contract proves the type but the checker cannot express it; add a short
  comment explaining that guarantee. Never use a cast to validate untrusted
  input or paper over an actual mismatch.

## NumPy dtype and shape

- NumPy arrays crossing real boundaries (embedder, Torch, soxr, public API
  inputs that may be lists) may use `np.asarray(x, dtype=...)`. Between typed
  arrays, use `.astype()` for an actual cast and ordinary NumPy operations
  otherwise. Do not reassert dtype at every intermediate step.
- Keep shape annotations where they communicate important contracts—especially
  embedding dimensions and matrix/vector APIs—in the existing shape-typed
  modules. Prefer normal inference or an unshaped `np.ndarray` for internal
  operations where carrying exact dimensions adds noise or hits unsupported
  stubs. Do not weaken project-wide checker settings to avoid one such error.
- Import `IntVar` from `shape_extensions`; declare each module-level dimension
  on its own line. Use shape variables in function signatures, not local or
  class-body annotations. Example:

  ```python
  import numpy as np
  from shape_extensions import IntVar

  N = IntVar("N")
  D = IntVar("D")


  def normalize(vectors: np.ndarray[[N, D]]) -> np.ndarray[[N, D]]: ...
  ```

- A dataclass holding shaped arrays should be a PEP 695 generic class, e.g.
  `class TrackRow[D: IntVar]` with `mean_vec: np.ndarray[[D]]`; parameterize
  each use (`TrackRow[D]`). Bare generic use can introduce implicit `Any`.
  Explicit PEP 695 parameters work in function and class definitions.
- For an unshaped array that needs a shape claim, use
  `shape_extensions.assert_shape` at the point the invariant is established.
  For dimension expressions, import `D` as `S` to avoid collision with a local
  dimension variable, and wrap dimensions in the claim: `S[V]`, `S[D]`, or
  `S[V * D]`. For `np.concatenate` / `np.frombuffer`, one assertion at the
  boundary should be enough. Do not add wrapper functions solely to placate
  inference.
- For reshape with a desired output shape, use
  `numpy_helpers.reshape(arr, (n, dim))`; it preserves the supplied shape type.
- For shaped `np.empty`, a literal zero dimension may remain `Literal[0]` and
  fail to unify. If zero comes from an array length, use `len(array)`; otherwise
  bind it to a typed `int` first.
- Prefer NumPy function forms such as `np.greater`, `np.greater_equal`, and
  `np.equal` when comparison operators lose shape information. Prefer tracked
  methods such as `x.mean(axis=...)` and `x.argmax()` when available.

## Handling checker and stub gaps

When Pyrefly reports an error:

1. Decide whether code violates the declared contract. If yes, fix the code or
   narrow/validate the value at the correct boundary.
2. If inference is unclear, use `reveal_type(expr)` temporarily; remove it after
   diagnosis. Use `typing.assert_type` sparingly when a generic helper's inferred
   type is itself a contract worth pinning down.
3. Prefer a standard typed method, then a small adapter/helper local to the
   untyped library boundary.
4. If valid code still hits a known stub gap, add the narrowest
   `# pyrefly: ignore[<error-code>]` and explain the reason on that line. Never
   use a blanket ignore or duplicate suppressions throughout the call chain.
5. Re-run `uv run pyrefly check`. Remove suppressions that are no longer needed;
   `unused-type-ignore = true` catches stale ones.

Known gaps in current stubs:

- `unsupported-operation`: `__setitem__` on shape-typed NumPy arrays. Slice
  assignment is valid at runtime; isolate suppression to assignment line.
- `unknown-argument-type`: `np.where` when fed `np.linalg.norm` output,
  `np.log` / `np.exp` / `soundfile.write` with unknown-typed arguments, and
  JSON config values passed into third-party constructors. Narrow or validate
  external JSON values where practical; do not wrap intermediate arrays in
  `np.asarray` solely to silence NumPy diagnostics. Follow dtype boundary rules
  above.
- Prefer `np.greater`, `np.greater_equal`, and `np.equal` over shaped-array
  comparison dunders rather than suppressing loss of shape information.

## Useful examples in this repository

- Runtime-checked boundary helpers: `src/local_smart_playlist/type_utils.py`
- Generic shape-preserving Torch-to-NumPy adapter: `src/local_smart_playlist/embed/model.py`
- Shaped records and database boundary validation:
  `src/local_smart_playlist/index/store.py`
