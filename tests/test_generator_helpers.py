"""
Covers the generator_groq.py bugs fixed in this pass:
  - _safe_excerpt used to crash (IndexError) on any chunk whose text didn't
    contain a literal "--", which happens for every business-profile/
    attribute/vibe chunk now that those are actually indexed.
  - is_out_of_coverage matched the 2-letter Indiana code "in" case-
    insensitively against the whole query, so the ubiquitous preposition
    "in" made almost any query read as "covered".

Note: only generator_groq.py is imported here (not generator_local.py) —
both call attach_prometheus() against the shared global Prometheus registry,
and instrumenting more than one FastAPI app per process raises a "Duplicated
timeseries" error. Keep FastAPI-app-creating modules to one per test process.
"""

import pytest

from generator_groq import _safe_excerpt, is_non_food_query, is_out_of_coverage


class TestSafeExcerpt:
    def test_extracts_review_body_after_double_dash(self):
        text = "passage: positive reviews for X mention:\n -- Great tacos here!"
        assert _safe_excerpt(text) == "Great tacos here!"

    def test_does_not_crash_on_text_without_double_dash(self):
        text = "passage: X is a Italian restaurant located at 123 Main St in Testville, TS."
        assert _safe_excerpt(text) == text

    def test_does_not_crash_on_none(self):
        assert _safe_excerpt(None) == ""

    def test_does_not_crash_on_empty_string(self):
        assert _safe_excerpt("") == ""

    def test_truncates_to_max_len(self):
        text = "-- " + ("a" * 1000)
        assert len(_safe_excerpt(text, max_len=50)) == 50


class TestNonFoodQuery:
    @pytest.mark.parametrize(
        "query",
        ["breakfast", "brunch", "dinner", "burgers", "sashimi",
         "best hot dog places in philadelphia", "sushi restaurants near me",
         "restaurant with parking in boise", "cheapest restaurants in tampa"],
    )
    def test_legitimate_food_queries_are_not_blocked(self, query):
        assert not is_non_food_query(query)

    @pytest.mark.parametrize(
        "query",
        ["hi", "hello", "thanks", "asdfgh", "what is the capital of france",
         "write a python script", "who won the super bowl"],
    )
    def test_off_topic_queries_are_blocked(self, query):
        assert is_non_food_query(query)


class TestOutOfCoverage:
    def test_covered_city_is_not_out_of_coverage(self):
        assert not is_out_of_coverage("best tacos in Philadelphia")

    def test_uncovered_city_is_out_of_coverage(self):
        assert is_out_of_coverage("best pizza in Chicago")

    def test_query_with_no_location_is_in_coverage(self):
        assert not is_out_of_coverage("upscale steakhouse")

    def test_philly_nickname_is_covered(self):
        assert not is_out_of_coverage("best ramen in Philly")
