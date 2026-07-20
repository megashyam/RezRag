"""Covers cache.py's key derivation and basic get/set/evict semantics."""

import importlib

import pytest


@pytest.fixture()
def cache_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "query_cache"))
    import cache

    importlib.reload(cache)  # pick up the patched CACHE_DIR
    yield cache
    cache.evict_all()


def test_key_is_deterministic_and_case_insensitive(cache_module):
    k1 = cache_module._key("Best Tacos in Philadelphia", 5, True)
    k2 = cache_module._key("best tacos in philadelphia  ", 5, True)
    assert k1 == k2


def test_key_differs_by_top_k_and_rerank_flag(cache_module):
    base = cache_module._key("sushi", 5, True)
    assert base != cache_module._key("sushi", 10, True)
    assert base != cache_module._key("sushi", 5, False)


def test_key_differs_by_retrieval_params(cache_module):
    base = cache_module._key("sushi", 5, True, k_rrf=60, initial_k=100, max_duplicates=2)
    assert base != cache_module._key("sushi", 5, True, k_rrf=30, initial_k=100, max_duplicates=2)
    assert base != cache_module._key("sushi", 5, True, k_rrf=60, initial_k=50, max_duplicates=2)


def test_set_then_get_roundtrips(cache_module):
    results = [{"restaurant": "Test Place", "score": 0.9}]
    cache_module.set_cached("best tacos", 5, True, results)
    hit = cache_module.get_cached("best tacos", 5, True)
    assert hit == results


def test_get_cached_miss_returns_none(cache_module):
    assert cache_module.get_cached("nonexistent query xyz", 5, True) is None


def test_evict_all_clears_entries(cache_module):
    cache_module.set_cached("q1", 5, True, [{"a": 1}])
    cache_module.set_cached("q2", 5, True, [{"b": 2}])
    n = cache_module.evict_all()
    assert n == 2
    assert cache_module.get_cached("q1", 5, True) is None
