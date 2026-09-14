"""Deprecated compatibility alias for :mod:`crewai.flow.conversational`."""

import sys

from crewai.flow import conversational as _real


sys.modules[__name__] = _real
