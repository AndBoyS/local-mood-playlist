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
