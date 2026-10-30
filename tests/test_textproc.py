"""Tests for text processing."""

from kbforge.textproc import TextProcessor, tokenize, tokenize_query


def test_tokenization_normalization():
    proc = TextProcessor()
    tokens = proc.tokenize("Hello, World! This is a TEST.")
    assert "hello" in tokens
    assert "world" in tokens


def test_punctuation_handling():
    proc = TextProcessor()
    tokens = proc.tokenize("it's test-case, example.")
    assert len(tokens) > 0


def test_unicode_handling():
    proc = TextProcessor()
    tokens = proc.tokenize("café naïve")
    assert tokens == ["café", "naïve"]


def test_stopwords_optional():
    proc = TextProcessor(remove_stopwords=True)
    tokens = proc.tokenize("the cat is on the mat")
    assert "the" not in tokens


def test_stemming_optional():
    proc = TextProcessor(stem=True)
    tokens = proc.tokenize("running runs ran")
    assert len(tokens) > 0


def test_query_document_consistency():
    q = tokenize_query("Hello World")
    d = tokenize("Hello World")
    assert q == d
