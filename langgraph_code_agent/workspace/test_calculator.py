import pytest

from calculator import divide


def test_divide() -> None:
    assert divide(8, 2) == 4


def test_divide_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
