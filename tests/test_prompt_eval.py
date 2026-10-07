"""Tests for mood prompt evaluation helpers."""

import pytest

from local_smart_playlist.query.prompt_eval import (
    EXTRA_MOOD_TERMS,
    build_mood_prompt_variants,
    pearson_correlation,
)


def test_build_mood_prompt_variants_includes_controlled_templates_and_extras() -> None:
    variants = build_mood_prompt_variants(("happy mood.", "upbeat mood."))

    assert len(variants) == (2 + len(EXTRA_MOOD_TERMS)) * 3
    assert variants[0].prompt == "happy mood."
    assert any(variant.prompt == "The music sounds happy." for variant in variants)
    assert any(variant.prompt == "A upbeat track." for variant in variants)
    assert any(variant.prompt == "exuberant mood." for variant in variants)


def test_build_mood_prompt_variants_deduplicates_terms_case_insensitively() -> None:
    variants = build_mood_prompt_variants(
        (),
        terms=("Happy", "happy", " upbeat "),
        template_ids=("mood",),
    )

    assert [variant.prompt for variant in variants] == ["Happy mood.", "upbeat mood."]


def test_build_mood_prompt_variants_rejects_invalid_mood_anchor() -> None:
    with pytest.raises(ValueError, match="mood anchor must end"):
        _ = build_mood_prompt_variants(("happy",))


def test_build_mood_prompt_variants_rejects_unknown_template() -> None:
    with pytest.raises(ValueError, match="unknown prompt template"):
        _ = build_mood_prompt_variants((), terms=("happy",), template_ids=("unknown",))


def test_pearson_correlation() -> None:
    assert pearson_correlation((1.0, 2.0, 3.0), (2.0, 4.0, 6.0)) == pytest.approx(1.0)
    assert pearson_correlation((1.0, 2.0, 3.0), (6.0, 4.0, 2.0)) == pytest.approx(-1.0)
    assert pearson_correlation((1.0, 1.0), (2.0, 3.0)) is None
    assert pearson_correlation((1.0,), (2.0,)) is None


def test_pearson_correlation_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        _ = pearson_correlation((1.0, 2.0), (1.0,))
