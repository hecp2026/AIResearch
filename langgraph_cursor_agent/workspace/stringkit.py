"""Small string utilities used as the demo target for the coding agent.

The ``truncate`` function below contains a deliberate bug so the agent has a
real, test-detectable problem to find and fix.
"""

from __future__ import annotations


def normalize_spaces(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return " ".join(text.split())


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate ``text`` so the result is at most ``max_length`` characters.

    When the text is cut, ``suffix`` is appended to signal truncation.
    """
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    if len(text) <= max_length:
        return text
    # BUG: appending suffix without reserving room makes the result longer
    # than max_length. It should keep room for the suffix instead.
    return text[:max_length] + suffix
