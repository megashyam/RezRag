"""Covers evaluation.py's fuzzy name matching and retrieval metrics — the
numbers that feed the README's reported MRR@5/Hit@K/P@K, so they need to be right."""

import pytest

from evaluation import (
    _normalize,
    name_match,
    in_relevant,
    mrr_at_k,
    hit_at_k,
    precision_at_k,
    location_accuracy,
)


def test_normalize_strips_accents_and_branch_suffixes():
    assert _normalize("Café Du Monde") == "cafe du monde"
    assert _normalize("Hattie B's Hot Chicken - Nashville") == "hattie b's hot chicken"
    assert _normalize("Shells Seafood Restaurant- Carrollwood") == "shells seafood restaurant"


def test_name_match_exact_and_prefix():
    assert name_match("Gumbo Shop", "Gumbo Shop")
    assert name_match("Hattie B's Hot Chicken - Nashville", "Hattie B's Hot Chicken")


def test_name_match_prefix_guard_requires_word_boundary():
    # "South" must not match "Southwark" just because it's a leading
    # character-substring — the len>=3 guard alone doesn't catch this (both
    # clear length 3); the prefix match needs a real word boundary.
    assert not name_match("South", "Southwark Restaurant And Bar")


def test_name_match_word_boundary_prefix_matches():
    assert name_match("Hattie B's Hot Chicken", "Hattie B's Hot Chicken Shack")


def test_name_match_distinguishes_different_branches():
    assert not name_match("Prince's Hot Chicken Shack", "Prince's Hot Chicken South")


def test_name_match_rejects_unrelated_names():
    assert not name_match("Vedge", "Bern's Steak House")


def test_in_relevant_checks_restaurant_or_name_key():
    relevant = {"Gumbo Shop", "Mr. B's Bistro"}
    assert in_relevant({"restaurant": "Gumbo Shop"}, relevant)
    assert in_relevant({"name": "Mr. B's Bistro"}, relevant)
    assert not in_relevant({"restaurant": "Some Other Place"}, relevant)


def test_mrr_at_k_rewards_earlier_hits():
    relevant = {"Gumbo Shop"}
    first = [{"restaurant": "Gumbo Shop"}, {"restaurant": "Other"}]
    second = [{"restaurant": "Other"}, {"restaurant": "Gumbo Shop"}]
    assert mrr_at_k(first, relevant, k=5) == 1.0
    assert mrr_at_k(second, relevant, k=5) == 0.5


def test_mrr_at_k_zero_when_no_hit_in_k():
    relevant = {"Gumbo Shop"}
    results = [{"restaurant": "Other"}] * 5
    assert mrr_at_k(results, relevant, k=5) == 0.0


def test_hit_at_k():
    relevant = {"Gumbo Shop"}
    assert hit_at_k([{"restaurant": "Gumbo Shop"}], relevant, k=5) == 1.0
    assert hit_at_k([{"restaurant": "Other"}], relevant, k=5) == 0.0


def test_precision_at_k_denominator_is_always_k():
    relevant = {"A", "B"}
    results = [{"restaurant": "A"}, {"restaurant": "B"}]
    assert precision_at_k(results, relevant, k=5) == pytest.approx(2 / 5)


def test_precision_at_k_empty_results():
    assert precision_at_k([], {"A"}, k=5) == 0.0


def test_location_accuracy_metro_area_matching():
    results = [{"city": "cherry hill"}, {"city": "philadelphia"}, {"city": "chicago"}]
    acc = location_accuracy(results, "philadelphia")
    assert acc == pytest.approx(2 / 3)


def test_location_accuracy_none_when_no_expected_city():
    assert location_accuracy([{"city": "philadelphia"}], None) is None
