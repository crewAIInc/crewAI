"""Test datetime compatibility module."""
from datetime import datetime, timedelta, timezone

from crewai.utilities.datetime_compat import UTC


def test_utc_timezone_compatibility():
    """Test that UTC timezone is compatible with both Python 3.10 and 3.11+"""
    assert UTC == timezone.utc
    assert UTC.tzname(None) == "UTC"
    # Verify it works with datetime.now()
    dt = datetime.now(UTC)
    assert dt.tzinfo == timezone.utc


def test_utc_timezone_edge_cases():
    """Test UTC timezone handling in edge cases."""
    # Test with leap year
    leap_date = datetime(2024, 2, 29, tzinfo=UTC)
    assert leap_date.tzinfo == timezone.utc
    
    # Test DST transition dates
    dst_date = datetime(2024, 3, 10, 2, 0, tzinfo=UTC)  # US DST start
    assert dst_date.tzinfo == timezone.utc
    
    # Test with minimum/maximum dates
    min_date = datetime.min.replace(tzinfo=UTC)
    max_date = datetime.max.replace(tzinfo=UTC)
    assert min_date.tzinfo == timezone.utc
    assert max_date.tzinfo == timezone.utc
    
    # Test timezone offset calculations
    dt = datetime(2024, 1, 1, tzinfo=UTC)
    offset = dt.utcoffset()
    assert offset == timedelta(0)  # UTC should always have zero offset
