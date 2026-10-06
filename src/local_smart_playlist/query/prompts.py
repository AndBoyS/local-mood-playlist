"""Mood query → LLM correction (OpenAI-compatible endpoint).

Free text becomes a small set of comma-attribute query variants; existing
comma-attribute lists pass through unchanged (typos/translation only).
"""

import json
import logging
import os
import re
import time
from collections import Counter

from local_smart_playlist.query.axis_prompts import ALL_AXIS_PROMPTS, AXIS_TO_PROMPTS, QueryAxis
from local_smart_playlist.type_utils import NonEmptyTuple

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"  # OpenCode Go subscription
DEFAULT_MODEL = "deepseek-v4-flash"


def calibration_axis_ranges() -> dict[QueryAxis, tuple[int, int]]:
    """Prompt-bank row ranges for each axis in ``calibration_prompt_bank``."""
    offset = 0
    ranges: dict[QueryAxis, tuple[int, int]] = {}
    for axis, prompts in AXIS_TO_PROMPTS.items():
        end = offset + len(prompts)
        ranges[axis] = (offset, end)
        offset = end
    return ranges


def calibration_prompt_bank() -> NonEmptyTuple[str]:
    """Single-axis reference prompts used for text-query calibration."""
    return ALL_AXIS_PROMPTS


ADAPT_PROMPT = (
    "You adapt a user's music request for MuQ-MuLan, a music-text embedding "
    "model whose captions use short, comma-separated lowercase attributes.\n\n"
    "A separate user message may provide supplementary Context; use it to "
    "interpret the query's intent, without replacing or contradicting the query.\n\n"
    "For an input that is already a comma-separated attribute list (e.g. "
    "'dreamy, melancholic'), return exactly one line, unchanged except for "
    "translation or typo fixes. Never expand or reorder it.\n\n"
    "For other inputs, preserve every explicitly stated musical detail. For "
    "purpose or situation requests (e.g. 'music to concentrate'), infer only "
    "the central, broad qualities directly implied by that intent. Return a "
    "distinct alternative for each meaningfully different interpretation; "
    "there is no fixed count. Avoid near-paraphrases. Keep each alternative "
    "to 1 or 2 broad attributes; do not build a full "
    "imagined track description or combine many independent dimensions. When "
    "meanings overlap, prefer fewer alternatives over near-paraphrases. Do not "
    "guess genre, instruments, vocals, tempo, production, or detailed "
    "arrangement/dynamics unless the request states them. For example, 'to "
    "concentrate' can become 'focused, unobtrusive' or 'calm, non-distracting'; "
    "do not add 'instrumental', 'slow', 'repetitive', or 'sparse'. 'Music to "
    "train' may imply 'energetic' or 'driving', but not a specific genre or "
    "instrument. Each variant must be a concise, comma-separated attribute "
    "caption ending with a period.\n\n"
    "Output one query per line, no bullets, numbering, quotes, or commentary."
)


class LlmError(Exception):
    """Raised when the LLM endpoint is unreachable or returns unusable output."""


def _settings() -> tuple[str, str, str]:
    base_url = os.environ.get("SP_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("SP_LLM_MODEL", DEFAULT_MODEL)
    sp_key = os.environ.get("SP_LLM_API_KEY")
    oc_key = os.environ.get("OPENCODE_API_KEY")
    if sp_key is not None:
        api_key = sp_key
    elif oc_key is not None:
        api_key = oc_key
    else:
        api_key = ""
    return base_url, model, api_key


def adapt_mood(mood: str, *, context: str | None = None) -> list[str]:
    """Adapt *mood* into embedding-friendly query variants; raises LlmError on failure."""
    import uuid

    import httpx

    base_url, model, api_key = _settings()
    headers = {"x-opencode-session": uuid.uuid4().hex}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
    messages = [
        {"role": "system", "content": ADAPT_PROMPT},
        {"role": "user", "content": f"Query: {mood}"},
    ]
    if context is not None:
        messages.append({"role": "user", "content": f"Context: {context}"})
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    try:
        started = time.perf_counter()
        resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=60.0)
        _ = resp.raise_for_status()
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        assert isinstance(content, str)
    except Exception as exc:
        logger.warning("llm adapt failed for %r via %s/%s: %s", mood, base_url, model, exc)
        msg = f"LLM adaptation failed: {exc}"
        raise LlmError(msg) from exc

    lines = [re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", line.strip()).strip("`\"'") for line in content.splitlines()]
    lines = list(dict.fromkeys(line for line in lines if line != ""))
    if len(lines) == 0:
        logger.warning("llm adapt returned no text for %r via %s/%s", mood, base_url, model)
        raise LlmError("LLM adaptation returned no text")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("llm adapt: %r -> %r (%s, %.0f ms)", mood, lines, model, elapsed_ms)
    return lines


def decompose_query_axes(query: str) -> dict[QueryAxis, str] | None:
    """Ask LLM to split one adapted query into axis-labelled source phrases.

    Values must preserve wording from *query*; unrecognized or unsupported
    attributes are omitted. Returns ``None`` when no supported axis is present.
    """
    import uuid

    import httpx

    base_url, model, api_key = _settings()
    headers = {"x-opencode-session": uuid.uuid4().hex}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
    axis_names = ", ".join(axis.value for axis in AXIS_TO_PROMPTS)
    system_prompt = (
        "Split one music query into its explicit musical components. Use only these axis keys: "
        f"{axis_names}. Return one JSON object mapping each present axis to a short phrase copied from the query. "
        "Do not invent, translate, expand, or infer attributes; keep multiple terms on one axis together. "
        "Omit absent or ambiguous axes. Return {} if no supported component exists. No markdown or commentary."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"},
        ],
        "temperature": 0.0,
    }
    try:
        started = time.perf_counter()
        resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=60.0)
        _ = resp.raise_for_status()
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        assert isinstance(content, str)
    except Exception as exc:
        logger.warning("llm axis decomposition failed for %r via %s/%s: %s", query, base_url, model, exc)
        msg = f"LLM axis decomposition failed: {exc}"
        raise LlmError(msg) from exc

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        msg = f"LLM axis decomposition returned invalid JSON: {exc}"
        raise LlmError(msg) from exc
    if not isinstance(decoded, dict):
        raise LlmError("LLM axis decomposition must return a JSON object")

    components: dict[QueryAxis, str] = {}
    for raw_axis, raw_component in decoded.items():
        if not isinstance(raw_axis, str) or not isinstance(raw_component, str):
            raise LlmError("LLM axis decomposition keys and values must be strings")
        try:
            axis = QueryAxis(raw_axis)
        except ValueError as exc:
            msg = f"LLM axis decomposition returned unsupported axis {raw_axis!r}"
            raise LlmError(msg) from exc
        component = raw_component.strip().strip(".,; ")
        if component != "":
            source_terms = Counter(re.findall(r"\w+", query.casefold()))
            component_terms = Counter(re.findall(r"\w+", component.casefold()))
            if len(component_terms - source_terms) > 0:
                raise LlmError(f"LLM axis decomposition invented terms for {axis.value!r}")
            components[axis] = component

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("llm axis decomposition: %r -> %r (%s, %.0f ms)", query, components, model, elapsed_ms)
    return components if len(components) > 0 else None
