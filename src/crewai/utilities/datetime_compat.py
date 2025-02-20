"""Compatibility module for datetime functionality across Python versions.

This module provides timezone constants that work consistently across different
Python versions, particularly focusing on maintaining compatibility between
Python 3.10 and newer versions.

Notes:
    - In Python 3.10, datetime.UTC is not available, so we use timezone.utc
    - In Python 3.11+, this provides equivalent functionality to datetime.UTC
    - This implementation maintains consistent behavior across versions for
      timezone-aware datetime operations
    - No known limitations or edge cases between versions
    - Safe to use with DST transitions and leap years
    - Maintains exact timezone offset (always UTC+00:00)

Example:
    >>> from datetime import datetime
    >>> from crewai.utilities.datetime_compat import UTC
    >>> dt = datetime.now(UTC)  # Creates timezone-aware datetime with UTC
"""
from datetime import timezone

UTC = timezone.utc  # Equivalent to datetime.UTC (Python 3.11+)
