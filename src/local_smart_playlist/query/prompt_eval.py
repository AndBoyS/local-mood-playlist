"""Prompt variants and label metrics for read-only mood evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, sqrt

EXTRA_MOOD_TERMS = (
    "joyful",
    "energetic",
    "excited",
    "lively",
    "vibrant",
    "exuberant",
    "buoyant",
    "bright",
)

PROMPT_TEMPLATES: dict[str, str] = {
    "mood": "{term} mood.",
    "paper": "A {term} track.",
    "natural": "The music sounds {term}.",
}
DEFAULT_TEMPLATE_IDS = ("mood", "paper", "natural")
MIN_CORRELATION_SAMPLES = 2


@dataclass(frozen=True)
class MoodPromptVariant:
    """One wording of one mood term."""

    term: str
    template_id: str
    prompt: str


def build_mood_prompt_variants(
    anchor_prompts: Sequence[str],
    *,
    terms: Sequence[str] | None = None,
    template_ids: Sequence[str] | None = None,
) -> list[MoodPromptVariant]:
    """Build controlled text variants from current mood anchors and extras."""
    if terms is None:
        mood_terms: list[str] = []
        for prompt in anchor_prompts:
            if not prompt.casefold().endswith(" mood."):
                raise ValueError(f"mood anchor must end with ' mood.': {prompt!r}")
            mood_terms.append(prompt[: -len(" mood.")])
        mood_terms.extend(EXTRA_MOOD_TERMS)
    else:
        mood_terms = list(terms)

    unique_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in mood_terms:
        cleaned = term.strip()
        if cleaned == "":
            raise ValueError("mood terms must be non-empty")
        key = cleaned.casefold()
        if key not in seen_terms:
            unique_terms.append(cleaned)
            seen_terms.add(key)

    selected_templates = DEFAULT_TEMPLATE_IDS if template_ids is None else tuple(template_ids)
    if len(selected_templates) == 0:
        raise ValueError("at least one prompt template is required")

    variants: list[MoodPromptVariant] = []
    for template_id in selected_templates:
        template = PROMPT_TEMPLATES.get(template_id)
        if template is None:
            raise ValueError(f"unknown prompt template: {template_id!r}")
        variants.extend(
            MoodPromptVariant(
                term=term,
                template_id=template_id,
                prompt=template.format(term=term),
            )
            for term in unique_terms
        )
    return variants


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Pearson r, or None when sample size/variance makes it undefined."""
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have equal lengths")
    if len(xs) < MIN_CORRELATION_SAMPLES:
        return None

    mean_x = fsum(xs) / len(xs)
    mean_y = fsum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    sum_xx = fsum(x * x for x in centered_x)
    sum_yy = fsum(y * y for y in centered_y)
    if sum_xx == 0.0 or sum_yy == 0.0:
        return None
    covariance = fsum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    return covariance / sqrt(sum_xx * sum_yy)
