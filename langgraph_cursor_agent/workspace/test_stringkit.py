import pytest

from stringkit import normalize_spaces, truncate


def test_normalize_spaces_collapses_and_strips() -> None:
    assert normalize_spaces("  hello   world  ") == "hello world"


def test_truncate_keeps_short_text_unchanged() -> None:
    assert truncate("hello", 10) == "hello"


def test_truncate_result_never_exceeds_max_length() -> None:
    result = truncate("hello world", 8)
    assert len(result) <= 8
    assert result.endswith("...")


def test_truncate_rejects_negative_length() -> None:
    with pytest.raises(ValueError):
        truncate("hello", -1)
