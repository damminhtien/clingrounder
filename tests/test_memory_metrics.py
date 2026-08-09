"""Cross-platform RSS unit normalization tests."""

import pytest

from clingrounder.evaluation.memory_metrics import rss_bytes_from_ru_maxrss


@pytest.mark.parametrize(
    ("platform", "raw", "expected"),
    [
        ("darwin", 10 * 1024 * 1024, 10 * 1024 * 1024),
        ("linux", 10 * 1024, 10 * 1024 * 1024),
        ("freebsd", 10 * 1024, 10 * 1024 * 1024),
    ],
)
def test_rss_units_are_normalized_to_bytes(platform: str, raw: int, expected: int) -> None:
    assert rss_bytes_from_ru_maxrss(raw, platform=platform) == expected


def test_negative_rss_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        rss_bytes_from_ru_maxrss(-1, platform="darwin")
