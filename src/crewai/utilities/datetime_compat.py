"""Compatibility module for datetime functionality across Python versions."""
from datetime import timezone

# Provide UTC timezone constant that works in Python 3.10+
# This is equivalent to datetime.UTC in Python 3.11+
UTC = timezone.utc
