"""
Covers the chunking bugs fixed in this pass:
  - a single over-long review used to be able to blow CHUNK_MAX_TOKENS on its
    own (bug: chunk token budget only loosely enforced)
  - chunks are now measured with the actual embedding model's tokenizer, not
    an unrelated one, and must stay under CHUNK_MAX_TOKENS
  - business-profile / attribute / vibe chunks must actually be produced
    (these used to be computed and then dropped before reaching the embedder)
"""

import pytest

import config
from data_pipeline.chunker import YelpChunking


@pytest.fixture(scope="module")
def chunker():
    return YelpChunking()


def test_review_batches_respect_token_budget(chunker):
    reviews = [f"This place was great, the {i}th time we visited." for i in range(200)]
    header = "passage: positive customer reviews for Test Place in Testville, TS mention:\n "

    chunks = chunker._create_review_batches(reviews, header)

    assert chunks, "expected at least one chunk"
    for c in chunks:
        n_tokens = len(chunker.enc.encode(c, add_special_tokens=True))
        assert n_tokens <= config.EMBED_MAX_SEQ_LENGTH, (
            f"chunk has {n_tokens} tokens, exceeds the embedding model's "
            f"max_seq_length ({config.EMBED_MAX_SEQ_LENGTH})"
        )


def test_single_over_long_review_is_truncated_not_overflowed(chunker):
    huge_review = "amazing food and service. " * 400
    header = "passage: positive customer reviews for Test Place in Testville, TS mention:\n "

    chunks = chunker._create_review_batches([huge_review], header)

    assert len(chunks) == 1
    n_tokens = len(chunker.enc.encode(chunks[0], add_special_tokens=True))
    assert n_tokens <= config.CHUNK_MAX_TOKENS


def test_empty_and_whitespace_reviews_are_dropped(chunker):
    header = "passage: positive customer reviews for Test Place in Testville, TS mention:\n "
    chunks = chunker._create_review_batches(["", "   ", None], header)
    assert chunks == []


def test_profile_chunk_includes_attributes_and_vibes(chunker):
    row = {
        "name": "Test Place",
        "city": "testville",
        "state": "TS",
        "categories": "Restaurants, Italian",
        "address": "123 Main St",
        "attributes": {
            "RestaurantsTakeOut": "True",
            "OutdoorSeating": "False",
            "Alcohol": "u'full_bar'",
            "Ambience": "{'romantic': True, 'casual': False}",
        },
        "positive": [],
        "neutral": [],
        "negative": [],
    }
    import pandas as pd

    result = chunker.process_row(pd.Series(row))
    profile_chunks, chunked_pos, chunked_neu, chunked_neg = result

    joined = " ".join(profile_chunks)
    assert "Test Place" in joined
    assert "Takeout Service" in joined
    assert "Serves Alcohol" in joined
    assert "romantic" in joined
    assert "Outdoor Seating" not in joined


def test_parse_attributes_and_vibes_do_not_double_parse():
    attrs_dict = YelpChunking._parse_attribute_dict(
        "{'GoodForKids': 'True', 'Ambience': \"{'trendy': True}\"}"
    )
    assert attrs_dict == {"GoodForKids": "True", "Ambience": "{'trendy': True}"}
    assert YelpChunking._parse_attributes(attrs_dict) == ["Good For Kids"]
    assert YelpChunking._parse_vibes(attrs_dict) == ["trendy"]


def test_parse_attribute_dict_handles_malformed_input():
    assert YelpChunking._parse_attribute_dict("not a dict") == {}
    assert YelpChunking._parse_attribute_dict(None) == {}
    assert YelpChunking._parse_attribute_dict(float("nan")) == {}
