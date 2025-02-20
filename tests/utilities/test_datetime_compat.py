"""Test datetime compatibility module."""
from datetime import timezone

from crewai.utilities.datetime_compat import UTC


def test_utc_timezone_compatibility():
    """Test that UTC timezone is compatible with both Python 3.10 and 3.11+"""
    assert UTC == timezone.utc
    assert UTC.tzname(None) == "UTC"
    # Verify it works with datetime.now()
    from datetime import datetime
    dt = datetime.now(UTC)
    assert dt.tzinfo == timezone.utc
