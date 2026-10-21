import math

import pytest

from dmc.canonical import canonical_json, content_hash


def test_key_order_and_whitespace_do_not_matter():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert canonical_json({"a": 1}) == '{"a":1}'


def test_integral_floats_collapse_to_ints():
    assert canonical_json({"x": 2.0}) == canonical_json({"x": 2})
    assert content_hash([1.0, 2.5]) == content_hash([1, 2.5])


def test_tuples_and_lists_are_equal():
    assert content_hash((1, "a")) == content_hash([1, "a"])


def test_non_finite_floats_rejected():
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_unicode_is_preserved_not_escaped():
    assert canonical_json("é") == '"é"'


def test_unsupported_types_rejected():
    with pytest.raises(TypeError):
        canonical_json({"x": object()})
